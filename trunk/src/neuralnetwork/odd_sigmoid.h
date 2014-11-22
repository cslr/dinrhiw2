/*
 * odd sigmoid activation function
 */


#include "blade_math.h"
#include "activation_function.h"

#ifndef odd_sigmoid_h
#define odd_sigmoid_h

namespace whiteice
{
  
  
  template <typename T=double>
    class odd_sigmoid : public activation_function<T>
  {
    public:
    
    odd_sigmoid(const T& s = 1.0, const T& b = 1.333);
    virtual ~odd_sigmoid();
    
    T calculate(const T& x)    const; // calculates value
    void calculate(const T& x, T& r) const; // calculates value
    T operator() (const T& x)  const; // calculates value of activation function
    T derivate(const T& x)  const; // calculates derivate of activation function
    
    bool has_max() const;  // has maximum
    bool has_min() const;  // has minimum
    bool has_zero() const; // has unique zero
    
    T max() const;  // gives maximum value of activation function
    T min() const;  // gives minimum value of activation function
    T zero() const; // gives zero location of activation function
    
    function<T,T>* clone() const; // creates copy of object
    
    
    // returns parameters
    T get_scale() const throw();
    T get_alpha() const throw();
    
    private:
    
    T scale;
    T alpha;
    
  };
  
  
  extern template class odd_sigmoid<float>;
  extern template class odd_sigmoid<double>;
  extern template class odd_sigmoid< math::blas_real<float> >;
  extern template class odd_sigmoid< math::blas_real<double> >;
  
}
  

#endif


