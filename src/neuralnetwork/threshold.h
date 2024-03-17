/*
 * threshold activation function
 *
 * non-continuous => no derivate everywhere
 *
 */

#include "activation_function.h"
#include "blade_math.h"


#ifndef threshold_h
#define threshold_h

namespace whiteice
{

  template <typename T>
    class threshold : public activation_function<T>
    {
    public:
      
      threshold();
      virtual ~threshold();
      
      T calculate(const T& x)    const; // calculates value
      T operator() (const T& x)  const; // calculates value of activation function
      T derivate(const T& x)  const; // calculates derivate of activation function
      
      bool has_max() const;  // has maximum
      bool has_min() const;  // has minimum
      bool has_zero() const; // has unique zero
      
      T max() const;  // gives maximum value of activation function
      T min() const;  // gives minimum value of activation function
      T zero() const; // gives zero location of activation function
      
      function<T,T>* clone() const; // creates copy of object
      
    private:
      
    };

  
  extern template class threshold<float>;
  extern template class threshold<double>;
  extern template class threshold< math::blas_real<float> >;
  extern template class threshold< math::blas_real<double> >;
  
}
  



#endif

