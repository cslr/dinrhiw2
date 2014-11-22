/*
 * make function easier for minimization process by
 * function stretching
 * 
 */

#ifndef stretched_function_cpp
#define stretched_function_cpp

#include <new>
#include <stdexcept>
#include "stretched_function.h"
#include "blade_math.h"
#include <math.h>


namespace whiteice
{
  
  template class stretched_function<float>;
  template class stretched_function<double>;
  template class stretched_function< math::blas_real<float> >;
  template class stretched_function< math::blas_real<double> >;
  
  
  
  // clones local copy of function
  template <typename T>
  stretched_function<T>::stretched_function(optimized_function<T>& f,
					    const math::vertex<T>& minima)
  {
    this->f = dynamic_cast< optimized_function<T>* >(f.clone());
    this->minima = minima;
    minima_value = f.calculate(this->minima);
  }
  
  
  // uses given pointer
  template <typename T>
  stretched_function<T>::stretched_function(optimized_function<T>* f,
					    const math::vertex<T>& minima)
  {
    this->f = f;
    this->minima = minima;
    minima_value = f->calculate(this->minima);
  }
  
  template <typename T>
  stretched_function<T>::~stretched_function()
  {
    if(f) delete f;
  }
  
  // calculates value of function
  template <typename T>
  T stretched_function<T>::operator() (const math::vertex<T>& x) const
  {
    return calculate(x);
  }
  
  template <typename T>
  T stretched_function<T>::calculate(const math::vertex<T>& x) const
  {
    // uses
    // g1 = 5000
    // g2 = 0.5
    // mu = 10^-10;
    
    // calculates G(x)
    T fx = f->calculate(x);
    
    T distance = (x - minima).norm();
    
    int signplus = sign(fx - minima_value) + 1;
    
    T Gx = fx;
    Gx  += T(5000.0f) * distance * signplus;
    
        
    // calculates tanh divider
    T divider;
    
    {
      double d;
      
      if(math::convert(d, Gx - fx) == false)
	throw std::domain_error("function stretching error: conversion between types failed");
      
      divider = T(tanh(0.0000000001*d));
    }
    
    
    return ( Gx + T(0.5 * signplus)/divider );
  }
  
  
  template <typename T>
  void stretched_function<T>::calculate(const math::vertex<T>& x, T& y) const {
    y = calculate(x);
  }

  
  // creates copy of object
  template <typename T>
  function<math::vertex<T>,T>* stretched_function<T>::clone() const
  {
    optimized_function<T>* fptr =
      dynamic_cast<optimized_function<T>*>(f->clone());
    
    return new stretched_function<T>(fptr, minima);
  }
  
  
  template <typename T>
  unsigned int stretched_function<T>::dimension() const throw(){
    return f->dimension();
  }
  
  
  template <typename T>
  int stretched_function<T>::sign(const T& x) const throw()
  {
    if(x > T(0.0)) return 1;
    else if(x == T(0.0)) return 0;
    else return -1;
  }
  
  
  
  ////////////////////////////////////////////////////////////
  
  template <typename T>
  bool stretched_function<T>::hasGradient() const throw()
  {
    return false;
  }
  
  
  template <typename T>
  math::vertex<T> stretched_function<T>::grad(math::vertex<T>& x) const
  {
    return x;
  }
  
  
  template <typename T>
  void stretched_function<T>::grad(math::vertex<T>& x, math::vertex<T>& y) const
  {
  }
  
  
  template <typename T>
  bool stretched_function<T>::hasHessian() const throw()
  {
    return false;
  }
  
  
  template <typename T>
  math::matrix<T> stretched_function<T>::hessian(math::vertex<T>& x) const
  {
    return math::matrix<T>(1,1);
  }
  
  
  template <typename T>
  void stretched_function<T>::hessian(math::vertex<T>& x, math::matrix<T>& y) const
  {
  }
  
}

#endif
