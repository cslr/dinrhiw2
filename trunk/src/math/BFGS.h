/*
 * Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer
 * minimizes the target error function.
 */


#include <pthread.h>
#include "optimized_function.h"
#include "dinrhiw_blas.h"

#ifndef BFGS_h
#define BFGS_h


namespace whiteice
{
  namespace math
  {
    
    template <typename T=blas_real<float> >
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
        /* 
	 * error function we are (indirectly) optimizing)
	 * can be same as U(x) if there are no uncertainties,
	 * but can be different if we are optimizing 
	 * statistical model and want to have early stopping
	 * in order to prevent optimizing to statistical
	 * noise.
	 */
        virtual T getError(const vertex<T>& x) const = 0;
      
      
	
        bool minimize(vertex<T>& x0);
	
        bool getSolution(vertex<T>& x, T& y, unsigned int& iterations) const;
	
	// continues, pauses, stops computation
        bool continueComputation();
        bool pauseComputation();
	bool stopComputation();

        // returns true if solution converged and we cannot
        // find better solution
        bool solutionConverged() const;

        // returns true if optimization thread is running
        bool isRunning() const;
	
      private:
        bool linesearch(vertex<T>& xn,
			const vertex<T>& x, const vertex<T>& d) const;

        bool wolfe_conditions(const vertex<T>& x0,
			      const T& alpha,
			      const vertex<T>& p) const;
      
        // current solution
	vertex<T> bestx; 
	T besty;
        volatile unsigned int iterations;
	
        volatile bool sleep_mode, thread_running, solution_converged;
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
    extern template class BFGS< blas_real<float> >;
    extern template class BFGS< blas_real<double> >;
    
  };
};

#endif
