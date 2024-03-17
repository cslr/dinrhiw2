/*
 * function interface to continuous Genetic Algorithm 3 (GA3)
 * that will be used to optimize neural network function.
 * (NOT learning or minimizing MSE but finding the best value
 *  from pretrained neural network returning SINGLE value)
 */

#include "optimized_function.h"
#include "vertex.h"
#include "dinrhiw_blas.h"
#include "nnetwork.h"

namespace whiteice
{
  
  template < typename T = math::blas_real<float> >
    class nnetwork_function : public optimized_function<T>
    {
    public:
      nnetwork_function(nnetwork<T>& nnet);
      virtual ~nnetwork_function();
      
      // returns input vectors dimension
      virtual unsigned int dimension() const ;

      // calculates value of function
      virtual T operator() (const math::vertex<T>& x) const;
      
      // calculates value
      virtual T calculate(const math::vertex<T>& x) const;
      
      // calculates value 
      // (optimized version, this is faster because output value isn't copied)
      virtual void calculate(const math::vertex<T>& x, T& y) const;
      
      // creates copy of object
      virtual function< math::vertex<T>, T>* clone() const;

      
      virtual bool hasGradient() const ;
      
      // gets gradient at given point
      virtual math::vertex<T> grad(math::vertex<T>& x) const;
      
      // gets gradient at given point (faster)
      virtual void grad(math::vertex<T>& x, math::vertex<T>& y) const;
      
    private:
      
      nnetwork<T>& net;
      
    };
  
  extern template class nnetwork_function< math::blas_real<float> >;
  extern template class nnetwork_function< math::blas_real<double> >;
};


