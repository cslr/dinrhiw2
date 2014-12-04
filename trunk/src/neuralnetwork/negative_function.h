/*
 * negative function for transforming maximization task
 * optimized_function into
 * minimization task or other way around.
 * neg(x) = -x
 *
 */


#include "optimized_function.h"


#ifndef negative_function_h
#define negative_function_h

namespace whiteice
{

  template <typename T>
    class negative_function : public optimized_function<T>
    {
    public:  
      negative_function(const optimized_function<T>& f); // clones local copy
      negative_function(const optimized_function<T>* f); // uses ptr and deletes it later
      virtual ~negative_function();
      
      // calculates value of function
      virtual T operator() (const math::vertex<T>& x) const PURE_FUNCTION;
      
      // calculates value
      virtual T calculate(const math::vertex<T>& x) const PURE_FUNCTION;
      
      virtual void calculate(const math::vertex<T>& x, T& y) const;
      
      // creates copy of object  
      virtual function<math::vertex<T>,T>* clone() const;
      
      virtual unsigned int dimension() const throw() PURE_FUNCTION;
      
      virtual bool hasGradient() const throw() PURE_FUNCTION;
      virtual math::vertex<T> grad(math::vertex<T>& x) const PURE_FUNCTION;
      virtual void grad(math::vertex<T>& x, math::vertex<T>& y) const;
      
    private:
      
      optimized_function<T>* f;
    };
  
  
  
  extern template class negative_function< float >;
  extern template class negative_function< double >;  
  extern template class negative_function< math::blas_real<float> >;
  extern template class negative_function< math::blas_real<double> >;
  
}


#endif
