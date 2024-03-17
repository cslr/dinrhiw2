
#include <new>
#include <cmath>
#include <exception>

#ifndef gaussian_cpp
#define gaussian_cpp


namespace whiteice
{
  template class gaussian<float>;
  template class gaussian<double>;
  template class gaussian< math::blas_real<float> >;
  template class gaussian< math::blas_real<double> >;
  
  
  
  template <typename T>
  gaussian<T>::gaussian(T mean_, T var_)
  {
    this->mean_val = mean_;
    this->var_val = var_;
  }
  
  
  // calculates value of function
  template <typename T>
  T gaussian<T>::operator() (const T& x) const
  {
    return T(exp( (-((double)(x - mean_val))/(2*var_val)) )); // e^(-x/2var)
  }
  
  
  // calculates value
  template <typename T>
  T gaussian<T>::calculate(const T& x) const
  {
    return T(exp( (-((double)(x - mean_val))/(2*var_val)) )); // e^(-x/2var)
  }
  
  
  // creates copy of object
  template <typename T>
  gaussian<T>* gaussian<T>::clone() const 
  {
    gaussian<T> *g = new gaussian(mean_val, var_val);
    return g;
  }
  
  
  template <typename T>
  T& gaussian<T>::mean() { return mean_val; }
  
  
  template <typename T>
  const T& gaussian<T>::mean() const { return mean_val; }
  
  
  template <typename T>
  T& gaussian<T>::variance(){ return var_val; }
  
  
  template <typename T>
  const T& gaussian<T>::variance() const{ return var_val; }
  
}

  
#endif
  

  
