
#ifndef activation_function_cpp
#define activation_function_cpp

#include "activation_function.h"
#include <assert.h>

namespace whiteice
{
  
  template <typename T>
  activation_function<T>::activation_function(){ }
  
  template <typename T>
  activation_function<T>::~activation_function(){ }
  
  
  // calculates derivate of activation function
  template <typename T>
  T activation_function<T>::derivate(const T& x) const
  { assert(0); return x; }
  
  template <typename T>
  bool activation_function<T>::has_max() const
  { assert(0); return false; } // has maximum
  
  template <typename T>
  bool activation_function<T>::has_min() const
  { assert(0); return false; } // has minimum
  
  template <typename T>
  bool activation_function<T>::has_zero() const
  { assert(0); return false; } // has uniq. zero
  
  
  template <typename T>
  T activation_function<T>::max() const
  { assert(0); return T(0); } // gives maximum value of activation function
  
  template <typename T>
  T activation_function<T>::min() const
  { assert(0); return T(0); } // gives minimum value of activation function
  
  template <typename T>
  T activation_function<T>::zero() const
  { assert(0); return T(0); } // gives zero location of activation function  
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class activation_function<float>;
  template class activation_function<double>;
  template class activation_function< math::blas_real<float> >;
  template class activation_function< math::blas_real<double> >;
  template class activation_function< math::vertex<float> >;
  template class activation_function< math::vertex<double> >;
  template class activation_function< math::vertex<math::blas_real<float> > >;
  template class activation_function< math::vertex<math::blas_real<double> > >;
  
}



#endif
