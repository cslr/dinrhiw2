
#include "ga3_test_function.h"

namespace whiteice
{
  template <typename T>
  ga3_test_function<T>::ga3_test_function()
  {
  }
  
  template <typename T>
  unsigned int ga3_test_function<T>::dimension() const 
  {
    return 2;
  }

  // calculates value of function
  template <typename T>
  T ga3_test_function<T>::operator() (const math::vertex<T>& x) const
  {
    if(x.size() != 2) return T(0.0);
    if(x[0] < T(0.0f) || x[0] > T(1.0)) return T(0.0);
    else if(x[1] < T(0.0f) || x[1] > T(1.0)) return T(0.0);
    
    return T(20.0)*x[0]*sin(T(40.0)*x[0]) + T(11.0)*x[1]*sin(T(20.0)*x[1]);
  }
  
  // calculates value
  template <typename T>
  T ga3_test_function<T>::calculate(const math::vertex<T>& x) const
  {
    if(x.size() != 2) return T(0.0);
    if(x[0] < T(0.0f) || x[0] > T(1.0)) return T(0.0);
    else if(x[1] < T(0.0f) || x[1] > T(1.0)) return T(0.0);

    return T(20.0)*x[0]*sin(T(40.0)*x[0]) + T(11.0)*x[1]*sin(T(20.0)*x[1]);
  }
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  template <typename T>
  void ga3_test_function<T>::calculate(const math::vertex<T>& x, T& y) const
  {
    if(x.size() != 2){ y = T(0.0); return; }
    if(x[0] < T(0.0f) || x[0] > T(1.0)){ y = T(0.0); return; }
    else if(x[1] < T(0.0f) || x[1] > T(1.0)){ y = T(0.0); return; }

    y = T(20.0)*x[0]*sin(T(40.0)*x[0]) + T(11.0)*x[1]*sin(T(20.0)*x[1]);
  }
  
  // creates copy of object
  template <typename T>
  function< math::vertex<T>,T>* ga3_test_function<T>::clone() const
  {
    return new ga3_test_function<T>();
  }

  
  template <typename T>
  bool ga3_test_function<T>::hasGradient() const 
  {
    return false;
  }
  
  // gets gradient at a given point
  template <typename T>
  math::vertex<T> ga3_test_function<T>::grad(math::vertex<T>& x) const
  {
    return x;
  }
  
  // gets gradient at given point (faster)
  template <typename T>
  void ga3_test_function<T>::grad(math::vertex<T>& x, math::vertex<T>& y) const
  {
    return;
  }
  
};


namespace whiteice
{
  template class ga3_test_function< float >;
  template class ga3_test_function< double >;
  template class ga3_test_function< math::blas_real<float> >;
  template class ga3_test_function< math::blas_real<double> >;    
};
