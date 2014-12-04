/*
 * (to be) optimized function interface
 * useful for passing functions for optimization
 * algorithms where input is always multidimensional
 * and final goodness value is single number
 */

#include "function.h"
#include "vertex.h"
#include "matrix.h"
#include "dinrhiw_blas.h"

#ifndef optimized_function_h
#define optimized_function_h

namespace whiteice
{

  template <typename T>
    class optimized_function : public function<math::vertex<T>, T>
    {
    public:
      // function call definitions inherited from
      // function interface.      
      
      // returns input vectors dimension
      virtual unsigned int dimension() const throw() PURE_FUNCTION = 0;
      
      
      
      virtual bool hasGradient() const throw() PURE_FUNCTION = 0;
      
      // gets gradient at given point
      virtual math::vertex<T> grad(math::vertex<T>& x) const PURE_FUNCTION = 0;
      
      // gets gradient at given point (faster)
      virtual void grad(math::vertex<T>& x, math::vertex<T>& y) const  = 0;
      
    };
  
}

  
#endif


