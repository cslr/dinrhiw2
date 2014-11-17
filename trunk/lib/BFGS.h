/*
 * Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer
 */


#include <pthread.h>
#include "optimized_function.h"
#include "atlas.h"

#ifndef BFGS_h
#define BFGS_h


namespace whiteice
{
  namespace math
  {
    
    template <typename T=atlas_real<float> >
      class BFGS
      {
      public:
	BFGS();
	~BFGS();
	
	bool minimize(whiteice::optimized_function<T>* f,
		      vertex<T>& x0);
	
	bool getSolution(vertex<T>& x, T& y);
	
	// continues, pauses, stops computation
	bool continueComputation();
	bool pauseComputation();
	bool stopComputation();
	
      private:
	whiteice::optimized_function<T>* f;
	
	// current solution
	vertex<T> bestx; 
	T besty;
	
	matrix<T> H; // initial (positive definite) hessian matrix
	
	
	bool sleep_mode, thread_running;
	pthread_t optimizer_thread;
	pthread_mutex_t sleep_lock, thread_lock, solution_lock;
	
	
      public:
	void __optimizerloop();
	
      };
    
  };
};



namespace whiteice
{
  namespace math
  {
    
    extern template class BFGS< float >;
    extern template class BFGS< double >;
    extern template class BFGS< atlas_real<float> >;
    extern template class BFGS< atlas_real<double> >;
    
  };
};

#endif
