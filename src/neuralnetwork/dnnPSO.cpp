
#include "dnnPSO.h"
#include "neuralnetwork.h"


namespace whiteice
{
  template class dnnPSO_optimized_function<float>;
  template class dnnPSO_optimized_function<double>;
  template class dnnPSO_optimized_function< math::blas_real<float> >;
  template class dnnPSO_optimized_function< math::blas_real<double> >;
  
  
  

  template <typename T>
  dnnPSO_optimized_function<T>::dnnPSO_optimized_function(neuralnetwork<T>& nn,
							  const dataset<T>* input,
							  const function<std::vector< math::vertex<T> >,T>& dist)
  {
    this->testnet = new neuralnetwork<T>(nn);
    this->input   = input;
    this->pseudodist = dist.clone();
    
    {
      math::vertex<T> v;
      
      if(this->testnet->exportdata(v) == false)
	throw std::logic_error("bad neural network parameter/copy");
      
      this->fvector_dimension = v.size();
    }
    
  }

  
  template <typename T>
  dnnPSO_optimized_function<T>::dnnPSO_optimized_function(const dnnPSO_optimized_function<T>& nnpsof)
  {
    this->testnet = new neuralnetwork<T>(*nnpsof.testnet);
    this->input  = nnpsof.input;
    this->pseudodist = nnpsof.pseudodist->clone();
    
    this->fvector_dimension = nnpsof.fvector_dimension;
  }

  
  template <typename T>
  dnnPSO_optimized_function<T>::~dnnPSO_optimized_function()
  {
    if(this->testnet)
      delete this->testnet;
    
    if(this->pseudodist)
      delete this->pseudodist;
  }
  
  
  
  // calculates value of function
  template <typename T>
  T dnnPSO_optimized_function<T>::operator()(const math::vertex<T>& x) const
  {
    return calculate(x);
  }
  
  
  
  // calculates value
  template <typename T>
  T dnnPSO_optimized_function<T>::calculate(const math::vertex<T>& x) const
  {
    // uses even samples to calculate error    
    
    std::vector< math::vertex<T> > n;
    typename dataset<T>::const_iterator i;    
    T error = T(0.0), counter = T(0.0);
    T e;
    
    i = input->begin();            
    n.resize(2);
    n[0].resize(fvector_dimension);
    n[1].resize(fvector_dimension);
    
    
    if(testnet->importdata(x) == false)
      goto bigerror;
    
    
    while(i != input->end()){
      typename dataset<T>::const_iterator j = input->begin();
      math::vertex<T> i_out;
      math::vertex<T> j_out;
      
      
      testnet->input() = *i;
      testnet->calculate();
      n[0]  = *i;
      i_out = testnet->output();
      
      while(j != input->end()){
	testnet->input() = *j;
	testnet->calculate();
	j_out = testnet->output();
	
	n[1] = *j;
	j_out = j_out - i_out;
	
	input->invpreprocess(n);
	
	e = j_out.norm() - pseudodist->calculate(n);
	error += e*e;
	
	j++;
      }
      
      counter += T(1.0);
      
      i++;
    }
    
    
    if(counter > T(0.0))
      error /= counter;
    
    return error;
    
  bigerror: // "exception handling with goto"
    std::cout << "calculate error" << std::endl;
    
    error = T(10.0);
    for(unsigned int i=0;i<100;i++)
      error *= error;
    
    return error; // very big number
  }
  
  
  
  template <typename T>
  void dnnPSO_optimized_function<T>::calculate(const math::vertex<T>& x, T& y) const
  {
    y = calculate(x);
  }
  
  
  template <typename T>
  unsigned int dnnPSO_optimized_function<T>::dimension() const 
  {
    return (this->fvector_dimension);
  }
  
  
  // creates copy of object
  template <typename T>
  function<math::vertex<T>,T>* dnnPSO_optimized_function<T>::clone() const
  {
    return new dnnPSO_optimized_function<T>(*this);
  }
  
  
  
  
  //////////////////////////////////////////////////////////////////////
  
  
  template <typename T>
  bool dnnPSO_optimized_function<T>::hasGradient() const {
    return false;
  }
  
  // gets gradient at given point (faster)
  template <typename T>
  math::vertex<T> dnnPSO_optimized_function<T>::grad(math::vertex<T>& x) const{
    return x;
  }
  
  
  template <typename T>
  void dnnPSO_optimized_function<T>::grad(math::vertex<T>& x, math::vertex<T>& y) const{
    return;
  }
  
  
  template <typename T>
  bool dnnPSO_optimized_function<T>::hasHessian() const {
    return false;
  }
  
  
  // gets gradient at given point (faster)
  template <typename T>
  math::matrix<T> dnnPSO_optimized_function<T>::hessian(math::vertex<T>& x) const{
    return math::matrix<T>(1,1);
    
  }
  
  
  template <typename T>
  void dnnPSO_optimized_function<T>::hessian(math::vertex<T>& x, math::matrix<T>& y) const{
    return;
  }
  
  
}

