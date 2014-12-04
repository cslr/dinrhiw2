
#include "nnetwork_function.h"


namespace whiteice
{
  template <typename T>  
  nnetwork_function<T>::nnetwork_function(nnetwork<T>& nnet) : net(nnet)
  {
  }
  
  template <typename T>  
  nnetwork_function<T>::~nnetwork_function()
  {
  }
  
  // returns input vectors dimension
  template <typename T>  
  unsigned int nnetwork_function<T>::dimension() const throw()
  {
    return net.input_size();
  }
  
  // calculates value of function
  template <typename T>  
  T nnetwork_function<T>::operator() (const math::vertex<T>& x) const
  {
    if(x.size() != net.input_size())
      return T(+1000000.0);
    
    net.input() = x;
    net.calculate();
    return net.output()[0];
  }
  
  // calculates value
  template <typename T>  
  T nnetwork_function<T>::calculate(const math::vertex<T>& x) const
  {
    if(x.size() != net.input_size())
      return T(+1000000.0);
    
    net.input() = x;
    net.calculate();
    return net.output()[0];
  }
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  template <typename T>  
  void nnetwork_function<T>::calculate(const math::vertex<T>& x, T& y) const
  {
    if(x.size() != net.input_size()){
      y = T(+1000000.0);
      return;
    }
    
    net.input() = x;
    net.calculate();
    y = net.output()[0];
  }
  
  // creates copy of object
  template <typename T>  
  function< math::vertex<T>, T>* nnetwork_function<T>::clone() const
  {
    return new nnetwork_function<T>(net);
  }
  
  template <typename T>  
  bool nnetwork_function<T>::hasGradient() const throw()
  {
    return false;
  }
  
  // gets gradient at given point
  template <typename T>  
  math::vertex<T> nnetwork_function<T>::grad(math::vertex<T>& x) const
  {
    return x;
  }
  
  // gets gradient at given point (faster)
  template <typename T>  
  void nnetwork_function<T>::grad(math::vertex<T>& x, math::vertex<T>& y) const
  {
    return;
  }
  
  template class nnetwork_function< float >;
  template class nnetwork_function< double >;  
  template class nnetwork_function< math::blas_real<float> >;
  template class nnetwork_function< math::blas_real<double> >;  
};
