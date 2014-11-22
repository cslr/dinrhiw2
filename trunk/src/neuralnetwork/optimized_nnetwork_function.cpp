
#include "optimized_nnetwork_function.h"
#include "vertex.h"
#include "matrix.h"


namespace whiteice
{
  template <typename T>
  optimized_nnetwork_function<T>::optimized_nnetwork_function(nnetwork<T>& network, 
							      dataset<T>& data) : 
    nn(network), ds(data)
  {
    // this->nn = network;
    // this->ds = data;
    
    if(ds.getNumberOfClusters() != 2)
      throw std::out_of_range("Number of clusters in ds must be 2 (input and output)");
    
    if(ds.size(0) != ds.size(1))
      throw std::out_of_range("Number of data points are not same in training clusters");
    
    if(ds.size(0) == 0)
      throw std::out_of_range("Data points cannot be zero");
    
    // FIXME needs more checks to see that NN and dataset are compatible
  }

  template <typename T>
  optimized_nnetwork_function<T>::~optimized_nnetwork_function()
  {
  }
  
  
  // calculates value of function
  template <typename T>
  T optimized_nnetwork_function<T>::operator() 
    (const math::vertex<T>& x) const 
  {
    // we will calculate squared error in the ds
    
    return this->calculate(x);
  }
  
  
  // calculates value (squared_error)
  template <typename T>
  T optimized_nnetwork_function<T>::calculate(const math::vertex<T>& x) const 
  {
    T error = T(0.0f);
    math::vertex<T> err;
    
    nn.importdata(x);
	
    for(unsigned int i=0;i<ds.size(0);i++){
      nn.input() = ds.access(0, i);
      nn.calculate();
      err = ds.access(1,i) - nn.output();
      
      for(unsigned int i=0;i<err.size();i++)
	error += (err[i]*err[i]) / T((float)err.size());
      
    }
    
    error /= T((float)ds.size());
    
    return error;
  }
  
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  template <typename T>
  void optimized_nnetwork_function<T>::calculate(const math::vertex<T>& x, T& y) const
  {
    y = this->calculate(x);
  }
  
  
  // creates copy of object
  template <typename T>
  function<math::vertex<T>, T>* optimized_nnetwork_function<T>::clone() const
  {
    return new optimized_nnetwork_function<T>(nn, ds);
  }
  
  
  // returns input vectors dimension (weight vector length)
  template <typename T>
  unsigned int optimized_nnetwork_function<T>::dimension() const throw() 
  {
    math::vertex<T> err, grad;
	
    nn.input() = ds.access(0, 0);
    nn.calculate(true);
    err = ds.access(1, 0) - nn.output();
    
    if(nn.gradient(err, grad) == false)
      return 0;
    else
      return grad.size();
  }
  
  
  template <typename T>
  bool optimized_nnetwork_function<T>::hasGradient() const throw() 
  {
    math::vertex<T> err, grad;
	
    nn.input() = ds.access(0, 0);
    nn.calculate(true);
    err = ds.access(1, 0) - nn.output();
    
    if(nn.gradient(err, grad) == false)
      return false;
    else
      return true;
  }
  
  
  // gets gradient at given point
  template <typename T>
  math::vertex<T> optimized_nnetwork_function<T>::grad(math::vertex<T>& x) const 
  {
    math::vertex<T> err, grad, sumgrad;
    unsigned int N = 0;
    
    nn.importdata(x); // import weight parameters
    
    // next we calculate the gradient for the whole DS
    // (sum of gradients for the whole ds)
    
    for(unsigned int i=0;i<ds.size(0);i++){
      nn.input() = ds.access(0, i);
      nn.calculate(true);
      err = ds.access(1,i) - nn.output();
	    
      if(nn.gradient(err, grad) == false)
	std::cout << "gradient failed." << std::endl;
      
      if(i == 0){
	sumgrad = grad;
	N = N + 1;
      }
      else{
	sumgrad = sumgrad + grad;
	N = N + 1;
      }
    }
    
    sumgrad /= T((float)N);
    
    return sumgrad;
  }
  
  
  // gets gradient at given point (faster)
  template <typename T>
  void optimized_nnetwork_function<T>::grad(math::vertex<T>& x, math::vertex<T>& y) const
  {
    y = this->grad(x);
  }
  
  
  template <typename T>
  bool optimized_nnetwork_function<T>::hasHessian() const throw() 
  {
    return false;
  }
  
  
  // gets hessian at given point
  template <typename T>
  math::matrix<T> optimized_nnetwork_function<T>::hessian(math::vertex<T>& x) const 
  {
    unsigned int D = this->dimension();
    
    math::matrix<T> H;
    H.resize(D, D);
    H.identity();
    
    return H;
  }
  
  
  // gets hessian at given point (faster)
  template <typename T>
  void optimized_nnetwork_function<T>::hessian(math::vertex<T>& x, math::matrix<T>& y) const
  {
    unsigned int D = this->dimension();
    
    y.resize(D, D);
    y.identity();
  }
  
  
  // explicit template instantations
  
  template class optimized_nnetwork_function< float >;
  template class optimized_nnetwork_function< double >;
  template class optimized_nnetwork_function< math::blas_real<float> >;
  template class optimized_nnetwork_function< math::blas_real<double> >;
  
};
