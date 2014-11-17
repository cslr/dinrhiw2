/*
 * identity (no) activation function
 */

#include "activation_function.h"
#include "blade_math.h"


#ifndef identity_activation_h
#define identity_activation_h

namespace whiteice
{

  template <typename T>
    class identity_activation : public activation_function<T>
    {
    public:
      
      identity_activation();
      virtual ~identity_activation();
      
      T calculate(const T& x) const; // calculates value
      void calculate(const T& x, T& y) const; // calculates value
      T operator() (const T& x)  const; // calculates value of activation function
      T derivate(const T& x)  const; // calculates derivate of activation function
      
      bool has_max() const;  // has maximum
      bool has_min() const;  // has minimum
      bool has_zero() const; // has unique zero
      
      T max() const;  // gives maximum value of activation function
      T min() const;  // gives minimum value of activation function
      T zero() const; // gives zero location of activation function
      
      function<T,T>* clone() const; // creates copy of object
      
    };
  
  
  extern template class identity_activation<float>;
  extern template class identity_activation<double>;
  extern template class identity_activation< math::atlas_real<float> >;
  extern template class identity_activation< math::atlas_real<double> >;
    
  
}
  

#endif
