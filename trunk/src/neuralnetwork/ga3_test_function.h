
#ifndef ga3_test_function_h
#define ga3_test_function_h

#include "function.h"
#include "optimized_function.h"
#include "dinrhiw_blas.h"
#include "vertex.h"
#include <math.h>

namespace whiteice
{
  // f(x,y) = 10*x*sin(40x) + 11*y*sin(20*y) [0,1]
  
  template <typename T=math::blas_real<float> >
  class ga3_test_function : public optimized_function< T >
  {
  public:

  ga3_test_function();

  // returns input vectors dimension
  virtual unsigned int dimension() const throw() PURE_FUNCTION;
  
  // calculates value of function
  T operator() (const math::vertex<T>& x) const PURE_FUNCTION;
  
  // calculates value
  T calculate(const math::vertex<T>& x) const PURE_FUNCTION;
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  void calculate(const math::vertex<T>& x, T& y) const;
  
  // creates copy of object
  virtual function< math::vertex<T>,T>* clone() const;
  
  virtual bool hasGradient() const throw() PURE_FUNCTION;
  
  // gets gradient at given point
  virtual math::vertex<T> grad(math::vertex<T>& x) const PURE_FUNCTION;
  
  // gets gradient at given point (faster)
  virtual void grad(math::vertex<T>& x, math::vertex<T>& y) const;
  
  
  };
  
};

namespace whiteice
{
  extern template class ga3_test_function< float >;
  extern template class ga3_test_function< double >;
  extern template class ga3_test_function< math::blas_real<float> >;
  extern template class ga3_test_function< math::blas_real<double> >;    
};


#endif
