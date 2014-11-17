/*
 * nnPSO - neural network particle swarm optimizer
 * 
 * 
 */
#include "neuralnetwork.h"
#include "dataset.h"
#include "optimized_function.h"
#include "vertex.h"
#include "PSO.h"


#ifndef nnPSO_h
#define nnPSO_h


namespace whiteice
{
  
  template <typename T>
    class nnPSO
    {
    public:
      
      // neural network which is target of this nnPSO
      // (only this nn can be improved)
      nnPSO(neuralnetwork<T>* nn,
	    const dataset<T>* input,
	    const dataset<T>* output,
	    unsigned int swarmsize);
      
      nnPSO(neuralnetwork<T>* nn,
	    const dataset<T>* data, // both in&out
	    unsigned int swarmsize);
      
      nnPSO(optimized_function<T>* nnfun,
	    unsigned int swarmsize);
	    
      ~nnPSO();
      
      bool improve(unsigned int niters = 50);
      
      T getError();
      T getCurrentError();
      bool getSolution(math::vertex<T>& v);
      
      // validation set error
      // in practice minimization should be
      // only continued as long as validation
      // set error becomes smaller
      // (I know that this is somewhat incorrect..)
      T getCurrentValidationError();
      
      bool verbosity(bool v) throw();
      
      // samples NN weight vector from PSO swarm
      // samplerate of particle is proportional to goodness
      const math::vertex<T>& sample();
      
      bool enableOvertraining() throw();
      bool disableOvertraining() throw();
      
    private:
      
      PSO<T>* pso;
      optimized_function<T>* nn_error;
      
      // validation(?) set - uses odd samples
      const dataset<T>* input;
      const dataset<T>* output;
      
      bool firsttime;
      unsigned int swarmsize;
      
      bool verbose; // be talkative?
    };
  
  
  
  template <typename T>
    class nnPSO_optimized_function : public optimized_function<T>
    {
    public:
      
      nnPSO_optimized_function(neuralnetwork<T>* nn,
			       const dataset<T>* input,
			       const dataset<T>* output);
      
      nnPSO_optimized_function(neuralnetwork<T>* nn,
			       const dataset<T>* data);
      
      nnPSO_optimized_function(const nnPSO_optimized_function<T>& nnpsof);
      
      ~nnPSO_optimized_function();
      
      
      // calculates value of function
      virtual T operator() (const math::vertex<T>& x) const PURE_FUNCTION;
      
      // calculates value
      virtual T calculate(const math::vertex<T>& x) const PURE_FUNCTION;
      
      virtual void calculate(const math::vertex<T>& x, T& y) const;
      
      virtual unsigned int dimension() const throw() PURE_FUNCTION;
      
      // creates copy of object
      virtual function<math::vertex<T>,T>* clone() const;
      
      // changes error calculation method
      bool getUseAllData() const throw();
      void enableUseAllData() throw();
      void disableUseAllData() throw();
      
      // calculations validation error
      // if all samples are used returns exactly same error 
      // (uses all samples) as calculate()
      T getValidationError(const math::vertex<T>& x) const PURE_FUNCTION;
      
      //////////////////////////////////////////////////////////////////////
      
      bool hasGradient() const throw() PURE_FUNCTION;
      
      // gets gradient at given point (faster)
      math::vertex<T> grad(math::vertex<T>& x) const PURE_FUNCTION;
      void grad(math::vertex<T>& x, math::vertex<T>& y) const;
      
      bool hasHessian() const throw() PURE_FUNCTION;
      
      // gets gradient at given point (faster)
      math::matrix<T> hessian(math::vertex<T>& x) const PURE_FUNCTION;
      void hessian(math::vertex<T>& x, math::matrix<T>& y) const;
      
    private:
      // if true error is calculated from all samples
      bool useAllSamples;
      
      neuralnetwork<T>* testnet; // calculates errors with testnet
      
      // uses even samples (error minimization set)
      const dataset<T>* input;
      const dataset<T>* output;
      
      unsigned int fvector_dimension;
      
    };
  
  
  
  extern template class nnPSO<float>;
  extern template class nnPSO<double>;
  extern template class nnPSO< math::atlas_real<float> >;
  extern template class nnPSO< math::atlas_real<double> >;
  extern template class nnPSO_optimized_function<float>;
  extern template class nnPSO_optimized_function<double>;
  extern template class nnPSO_optimized_function< math::atlas_real<float> >;
  extern template class nnPSO_optimized_function< math::atlas_real<double> >;
  
};



#endif

