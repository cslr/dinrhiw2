

#ifndef odd_sigmoid_cpp
#define odd_sigmoid_cpp

#include <cmath>
#include <cassert>
#include "odd_sigmoid.h"
#include "blade_math.h"


namespace whiteice
{
  
  template <typename T>
  odd_sigmoid<T>::odd_sigmoid(const T& s, const T& a)
  {
    assert(s > 0 && a > 0);
    
    scale = s;
    alpha = a;
  }
  
  
  template <typename T>
  odd_sigmoid<T>::~odd_sigmoid(){ }
  
  
  template <typename T>
  T odd_sigmoid<T>::calculate(const T& x) const  // calculates value of activation function
  {
    if(x < T(-10.0f)) return scale;
    else if(x > T(10.0f)) return scale;
    
    T eav = static_cast<T>(whiteice::math::exp(-alpha * x));
    
    return ( scale * (T(1.0f) - eav)/(T(1.0f) + eav) );
  }
  
  
  template <typename T>
  void odd_sigmoid<T>::calculate(const T& x, T& r) const // calculates value
  {
    r = calculate(x);
  }
  
  
  template <typename T>
  T odd_sigmoid<T>::operator() (const T& x) const  // calculates value of activation function
  {
    return calculate(x);
  }
  
  
  template <typename T>
  T odd_sigmoid<T>::derivate(const T& x) const  // calculates derivate of activation function
  {
    if(x < T(-10.0f)) return T(0.00001f);
    else if(x > T(10.0f)) return T(0.00001f);
    
    T eav = static_cast<T>(exp(-alpha * x));
    T v = (T(1.0f) - eav)/(T(1.0f) + eav);
    
    return ((scale*alpha/2) * ( T(1.0f) - v*v ) );
  }
  
  
  template <typename T>
  bool odd_sigmoid<T>::has_max() const   // has maximum
  {
    return true;
  }
  
  
  template <typename T>
  bool odd_sigmoid<T>::has_min() const   // has minimum
  {
    return true;
  }


  template <typename T>
  bool odd_sigmoid<T>::has_zero() const  // has unique zero
  {
    return true;
  }
  
  
  template <typename T>
  T odd_sigmoid<T>::max() const   // gives maximum value of activation function
  {
    return  scale;
  }
  
  
  template <typename T>
  T odd_sigmoid<T>::min() const   // gives minimum value of activation function
  {
    return -scale;
  }
  
  
  template <typename T>
  T odd_sigmoid<T>::zero() const  // gives zero location of activation function
  {
    return  0.0;
  }
  
  
  template <typename T>
  function<T,T>* odd_sigmoid<T>::clone() const  // creates copy of object
  {
    return static_cast< function<T,T>* >(new odd_sigmoid<T>(scale, alpha));
  }
  
  
  template <typename T>
  T odd_sigmoid<T>::get_scale() const throw(){
    return scale;
  }
  
  
  template <typename T>
  T odd_sigmoid<T>::get_alpha() const throw(){
    return alpha;
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class odd_sigmoid<float>;
  template class odd_sigmoid<double>;
  template class odd_sigmoid< math::blas_real<float> >;
  template class odd_sigmoid< math::blas_real<double> >;
  
}
  
#endif






