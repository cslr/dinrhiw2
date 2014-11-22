/*
 * creates "stretched function" as described in
 *   Recent approaches to global optimization problems
 *   through Particle Swarm Optimization. 
 *   Parsopoulos, K.E.  Vrahatis, M.N. 2002
 *   [referes to another older paper]
 *
 * from given to_be_minimized_ optimized_function f
 */


#ifndef stretched_function_h
#define stretched_function_h

#include "optimized_function.h"



namespace whiteice
{
  template <typename T>
    class stretched_function : public optimized_function<T>
    {
    public:
      // clones local copy of function
      stretched_function(optimized_function<T>& f,
			 const math::vertex<T>& minima);
      
      // uses given pointer, function will delete this pointer
      stretched_function(optimized_function<T>* f,
			 const math::vertex<T>& minima);
      
      virtual ~stretched_function();
      
      // calculates value of function
      virtual T operator() (const math::vertex<T>& x) const PURE_FUNCTION;
      virtual T calculate(const math::vertex<T>& x) const PURE_FUNCTION;
      virtual void calculate(const math::vertex<T>& x, T& y) const;
      
      virtual function<math::vertex<T>,T>* clone() const; // creates copy of object  
      
      virtual unsigned int dimension() const throw() PURE_FUNCTION;
      
      
      //////////////////////////////////////////////////
      // not implemented
      
      virtual bool hasGradient() const throw() PURE_FUNCTION;
      virtual math::vertex<T> grad(math::vertex<T>& x) const PURE_FUNCTION;
      virtual void grad(math::vertex<T>& x, math::vertex<T>& y) const;
      
      virtual bool hasHessian() const throw() PURE_FUNCTION;
      virtual math::matrix<T> hessian(math::vertex<T>& x) const PURE_FUNCTION;
      virtual void hessian(math::vertex<T>& x, math::matrix<T>& y) const;
      
    private:
      optimized_function<T>* f; // original function
      math::vertex<T> minima;
      T minima_value;
      
      // sign() function for function streching
      int sign(const T& x) const throw();
    };
  
  
  
  extern template class stretched_function<float>;
  extern template class stretched_function<double>;
  extern template class stretched_function< math::blas_real<float> >;
  extern template class stretched_function< math::blas_real<double> >;
  
}

  
#endif


