/*
 * test function for optimization
 */

#include "optimized_function.h"
#include "matrix.h"
#include "vertex.h"

#ifndef test_function_h
#define test_function_h

namespace whiteice
{

  class test_function : public optimized_function< math::blas_real<float> >
    {
    public:
      
      // calculates value of function
      math::blas_real<float>
	operator() (const math::vertex< math::blas_real<float> >& x)
	const PURE_FUNCTION;
      
      // calculates value
      math::blas_real<float> 
	calculate(const math::vertex< math::blas_real<float> >& x)
	const PURE_FUNCTION;
      
      // calculates value
      void calculate(const math::vertex< math::blas_real<float> >& x,
		     math::blas_real<float>& y) const;
	
      
      // creates copy of object
      function<math::vertex< math::blas_real<float> >, math::blas_real<float> >*
	clone() const;
      
      // returns input vectors dimension
      unsigned int dimension() const  PURE_FUNCTION;
      
      
      bool hasGradient() const  PURE_FUNCTION;
      math::vertex< math::blas_real<float> > grad(math::vertex< math::blas_real<float> >& x) const PURE_FUNCTION;
      void grad(math::vertex< math::blas_real<float> >& x, math::vertex< math::blas_real<float> >& y) const;
      
      
      bool hasHessian() const  PURE_FUNCTION;
      math::matrix< math::blas_real<float> > hessian(math::vertex< math::blas_real<float> >& x) const PURE_FUNCTION;
      void hessian(math::vertex< math::blas_real<float> >& x, math::matrix< math::blas_real<float> >& y) const;
      
    };
}


#endif
