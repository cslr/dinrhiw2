/*
 * (to be) optimized function interface
 * useful for passing functions for optimization
 * algorithms where input is always multidimensional
 * and final goodness value is single number
 */

#include "function.h"
#include "vertex.h"
#include "matrix.h"
#include "atlas.h"
#include "optimized_function.h"
#include "nnetwork.h"
#include "dataset.h"

#ifndef optimized_nnetwork_function_h
#define optimized_nnetwork_function_h

namespace whiteice
{
  // function for BFGS optimization (minimization) of squared error
  // of neural network errors (nnetwork)

  template <typename T=math::atlas_real<float> >
    class optimized_nnetwork_function : public optimized_function<T>
    {
    public:
      // function call definitions inherited from
      // function interface.
      
      optimized_nnetwork_function(nnetwork<T>& network, dataset<T>& ds);

      virtual ~optimized_nnetwork_function();
      
      // calculates value of function
      virtual T operator() (const math::vertex<T>& x) const PURE_FUNCTION;
      
      // calculates value
      virtual T calculate(const math::vertex<T>& x) const PURE_FUNCTION;
      
      // calculates value 
      // (optimized version, this is faster because output value isn't copied)
      virtual void calculate(const math::vertex<T>& x, T& y) const;
      
      // creates copy of object
      virtual function<math::vertex<T>, T>* clone() const;
      
      // returns input vectors dimension
      virtual unsigned int dimension() const throw() PURE_FUNCTION;
      
      virtual bool hasGradient() const throw() PURE_FUNCTION;
      
      // gets gradient at given point
      virtual math::vertex<T> grad(math::vertex<T>& x) const PURE_FUNCTION;
      
      // gets gradient at given point (faster)
      virtual void grad(math::vertex<T>& x, math::vertex<T>& y) const;
      
      virtual bool hasHessian() const throw() PURE_FUNCTION;
      
      // gets hessian at given point
      virtual math::matrix<T> hessian(math::vertex<T>& x) const PURE_FUNCTION;
      
      // gets hessian at given point (faster)
      virtual void hessian(math::vertex<T>& x, math::matrix<T>& y) const;
      
    private:
      
      nnetwork<T>& nn;
      dataset<T>&  ds;
    };
};


namespace whiteice
{
  extern template class optimized_nnetwork_function< float >;
  extern template class optimized_nnetwork_function< double >;
  extern template class optimized_nnetwork_function< math::atlas_real<float> >;
  extern template class optimized_nnetwork_function< math::atlas_real<double> >;

};

  
#endif


