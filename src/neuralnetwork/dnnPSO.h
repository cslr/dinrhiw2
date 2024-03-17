/*
 * OUTDATED CODE: DOES NOT COMPILE
 *
 * dnnPSO
 * 
 * distance learning PSO learner
 * 
 * This teaches NN to learn mapping from data to feature distance
 * based euclidean space (functional form of d(x,y) isn't necessarily
 * known, only examples, although it can be known if data is attempted
 * to be mapped in a D-dimensional space with same distances as calculated
 * by d(x,y))
 * 
 * Let d(x,y) be (non-euclidean)
 * distance function between two data points.
 * PSO minimizes 
 * E[0.5*(d(x,y) - ||f(x) - f(y)||)^2]
 *
 * where f() is neural network to
 * f() may also map data to higher all
 * lower dimensional spaces than original
 * data.
 * 
 * This is somewhat related to SOM but
 * but in this case name of approach could
 * be "pseudodistance based organizing map".
 * 
 * d(x,y) > 0 here maybe for example in a form
 * |g(x) - g(y)|, g(x) > 0.
 *
 * For example, g(x) maybe chess board goodness
 * evaluation function (heuristical) and
 * f is neural network which maps data to a
 * equal number of dimensions than original
 * data. 
 * 
 * (Problem may need some sort of regularization:
 *  f(x) must be bijective and/or all data space should
 *  be used. (f(x) shouldn't converge to [g(x) 0 0 0..])
 * 
 */


#include "neuralnetwork.h"
#include "dataset.h"
#include "optimized_function.h"
#include "vertex.h"
#include "PSO.h"


#ifndef dnnPSO_h
#define dnnPSO_h


namespace whiteice
{
  
  template <typename T>
    class dnnPSO_optimized_function : public optimized_function<T>
    {
    public:
      
      dnnPSO_optimized_function(neuralnetwork<T>& nn,
				const dataset<T>* input,
				const function<std::vector< math::vertex<T> >,T>& dist);
      
      dnnPSO_optimized_function(const dnnPSO_optimized_function<T>& nnpsof);
      
      ~dnnPSO_optimized_function();
      
      
      // calculates value of function
      virtual T operator() (const math::vertex<T>& x) const PURE_FUNCTION;
      
      // calculates value
      virtual T calculate(const math::vertex<T>& x) const PURE_FUNCTION;
      
      virtual void calculate(const math::vertex<T>& x, T& y) const;
      
      virtual unsigned int dimension() const  PURE_FUNCTION;
      
      // creates copy of object
      virtual function<math::vertex<T>,T>* clone() const;
      
      
      
      //////////////////////////////////////////////////////////////////////
      
      bool hasGradient() const  PURE_FUNCTION;
      
      // gets gradient at given point (faster)
      math::vertex<T> grad(math::vertex<T>& x) const PURE_FUNCTION;
      void grad(math::vertex<T>& x, math::vertex<T>& y) const;
      
      bool hasHessian() const  PURE_FUNCTION;
      
      // gets gradient at given point (faster)
      math::matrix<T> hessian(math::vertex<T>& x) const PURE_FUNCTION;
      void hessian(math::vertex<T>& x, math::matrix<T>& y) const; 
      
    private:
      
      neuralnetwork<T>* testnet; // calculates errors with testnet
      const dataset<T>* input;   // dataset
      const function<std::vector< math::vertex<T> >, T>* pseudodist;
      
      unsigned int fvector_dimension;
      
    };
  
  
  extern template class dnnPSO_optimized_function<float>;
  extern template class dnnPSO_optimized_function<double>;
  extern template class dnnPSO_optimized_function< math::blas_real<float> >;
  extern template class dnnPSO_optimized_function< math::blas_real<double> >;
  
};



#endif

