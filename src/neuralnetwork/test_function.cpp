

#include "test_function.h"
#include <new>


namespace whiteice
{
  
  // calculates value of function
  math::atlas_real<float>
  test_function::operator() (const whiteice::math::vertex< whiteice::math::atlas_real<float> >& x) const
  {
    return calculate(x);
  }
  
  
  // calculates value
  math::atlas_real<float>
  test_function::calculate(const whiteice::math::vertex<whiteice::math::atlas_real<float> >& x) const
  {
    whiteice::math::atlas_real<float> xf = x[0] / whiteice::math::atlas_real<float>(1000.0);
    
    // min at x = 1, value 9.0, max is unlimited  x-> inf.
    return (xf*xf*xf*xf + xf*xf - whiteice::math::atlas_real<float>(2.0f)*xf + 
	    whiteice::math::atlas_real<float>(10.0));
  }
  
  
  
  void test_function::calculate(const math::vertex< math::atlas_real<float> >& x,
				math::atlas_real<float>& y) const
  {
    y = calculate(x);
  }
  
  
  // creates copy of object
  function<math::vertex<whiteice::math::atlas_real<float> >, whiteice::math::atlas_real<float> >*
  test_function::clone() const
  {
    return new test_function;
  }
  
  
  // returns input vectors dimension
  unsigned int test_function::dimension() const throw()
  {
    return 1;
  }
  
  
  
  
  bool test_function::hasGradient() const throw()
  { return false; }
  
  math::vertex< math::atlas_real<float> > test_function::grad(math::vertex< math::atlas_real<float> >& x) const
  { return x; }
  
  void test_function::grad(math::vertex< math::atlas_real<float> >& x, math::vertex< math::atlas_real<float> >& y) const
  { y = calculate(x); }
  
  
  bool test_function::hasHessian() const throw()
  { return false; }
  
  math::matrix< math::atlas_real<float> > test_function::hessian(math::vertex< math::atlas_real<float> >& x) const
  { return math::matrix< math::atlas_real<float> >(1,1); }
  
  void test_function::hessian(math::vertex< math::atlas_real<float> >& x, math::matrix< math::atlas_real<float> >& y) const
  { y = calculate(x); }
  
}
