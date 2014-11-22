/*
 * neural networks activation function interface
 *
 * implementations must override all methods
 * calling any other functions except ctor/dtor leads
 * to program crash.
 */

#include "function.h"
#include "dinrhiw_blas.h"
#include "vertex.h"
#include <vector>

#ifndef activation_function_h
#define activation_function_h

namespace whiteice
{
  
  template <typename T=double>
    class activation_function : public function<T,T>
  {
    public:
    
    activation_function();
    virtual ~activation_function();
    
    // calculates derivate of activation function
    virtual T derivate(const T& x) const;
    
    virtual bool has_max() const;  // has maximum
    virtual bool has_min() const;  // has minimum
    virtual bool has_zero() const; // has uniq. zero
    
    virtual T max() const;  // gives maximum value of activation function
    virtual T min() const;  // gives minimum value of activation function
    virtual T zero() const; // gives zero location of activation function
    
  };
  
  
  extern template class activation_function<float>;
  extern template class activation_function<double>;
  extern template class activation_function< math::blas_real<float> >;
  extern template class activation_function< math::blas_real<double> >;
  extern template class activation_function< math::vertex<float> >;
  extern template class activation_function< math::vertex<double> >;
  extern template class activation_function< math::vertex<math::blas_real<float> > >;
  extern template class activation_function< math::vertex<math::blas_real<double> > >;
  
}

#endif





