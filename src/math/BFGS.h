/*
 * Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer
 * minimizes the target error function.
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
      
      protected:
        /* optimized function */

        virtual T U(const vertex<T>& x) const = 0;
        virtual vertex<T> Ugrad(const vertex<T>& x) const = 0;

      public:
	
        bool minimize(vertex<T>& x0);
	
        bool getSolution(vertex<T>& x, T& y, unsigned int& iterations) const;
	
	// continues, pauses, stops computation
	bool continueComputation();
	bool pauseComputation();
	bool stopComputation();
	
      private:
        void linesearch(vertex<T>& xn,
			const vertex<T>& x, const vertex<T>& d) const;
      
        // current solution
	vertex<T> bestx; 
	T besty;
        volatile unsigned int iterations;
	
        volatile bool sleep_mode, thread_running;
	pthread_t optimizer_thread;
        mutable pthread_mutex_t sleep_lock, thread_lock, solution_lock;
	
	
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
