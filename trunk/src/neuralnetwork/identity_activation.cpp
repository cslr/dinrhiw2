
#include "identity_activation.h"


#ifndef identity_activation_cpp
#define identity_activation_cpp


namespace whiteice
{
  
  
  template <typename T>
  identity_activation<T>::identity_activation(){ }
  
  template <typename T>
  identity_activation<T>::~identity_activation(){ }
  
  // calculates value
  template <typename T>
  T identity_activation<T>::calculate(const T& x) const{ return x; }
  
  // calculates value
  template <typename T>
  void identity_activation<T>::calculate(const T& x, T& y) const { y = x; }
  
  // calculates value of activation function
  template <typename T>
  T identity_activation<T>::operator() (const T& x) const{ return x; }
  
  // calculates derivate of activation function
  template <typename T>
  T identity_activation<T>::derivate(const T& x)  const
  {
    return static_cast<T>(x);
  }
  
  // has maximum
  template <typename T>
  bool identity_activation<T>::has_max() const { return false; }
  
  // has minimum
  template <typename T>
  bool identity_activation<T>::has_min() const { return false; }
  
  // has unique zero ( = true (assumes T has one) )
  template <typename T>
  bool identity_activation<T>::has_zero() const{ return true; }
  
  // gives maximum value of activation function
  template <typename T>
  T identity_activation<T>::max() const{ return static_cast<T>(0); }
  
  // gives minimum value of activation function
  template <typename T>
  T identity_activation<T>::min() const{ return static_cast<T>(0); }
  
  // gives zero location of activation function
  template <typename T>
  T identity_activation<T>::zero() const{ return static_cast<T>(0); }
  
  // creates copy of object
  template <typename T>
  function<T,T>* identity_activation<T>::clone() const
  {
    return static_cast< function<T,T>* >
    (new identity_activation);
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class identity_activation<float>;
  template class identity_activation<double>;
  template class identity_activation< math::atlas_real<float> >;
  template class identity_activation< math::atlas_real<double> >;
  
}

#endif

