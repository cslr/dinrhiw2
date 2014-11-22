
#include "ga3_test_function.h"

namespace whiteice
{
  template <typename T>
  ga3_test_function<T>::ga3_test_function()
  {
  }

  // calculates value of function
  template <typename T>
  T ga3_test_function<T>::operator() (const math::vertex<T>& x) const
  {
    return T(20.0)*x[0]*sin(T(40.0)*x[0]) + T(11.0)*x[1]*sin(T(20.0)*x[1]);
  }
  
  // calculates value
  template <typename T>
  T ga3_test_function<T>::calculate(const math::vertex<T>& x) const
  {
    return T(20.0)*x[0]*sin(T(40.0)*x[0]) + T(11.0)*x[1]*sin(T(20.0)*x[1]);
  }
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  template <typename T>
  void ga3_test_function<T>::calculate(const math::vertex<T>& x, T& y) const
  {
    y = T(20.0)*x[0]*sin(T(40.0)*x[0]) + T(11.0)*x[1]*sin(T(20.0)*x[1]);
  }
  
  // creates copy of object
  template <typename T>
  function< math::vertex<T>,T>* ga3_test_function<T>::clone() const
  {
    return new ga3_test_function<T>();
  }
};


namespace whiteice
{
  template class ga3_test_function< float >;
  template class ga3_test_function< double >;
  template class ga3_test_function< math::blas_real<float> >;
  template class ga3_test_function< math::blas_real<double> >;    
};
