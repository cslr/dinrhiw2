/*
 * Limited Memory-Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer
 * minimizes the target error function.
 */

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "dinrhiw_blas.h"
#include "vertex.h"


#ifndef LBFGS_h
#define LBFGS_h


namespace whiteice
{
  namespace math
  {
    
    template <typename T=blas_real<float> >
      class LBFGS
      {
      public:
        LBFGS(bool overfit=false); // overfit: do not use early stopping via getError() function
        virtual ~LBFGS();
      
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
			const vertex<T>& x,
			const vertex<T>& d) const;

        bool wolfe_conditions(const vertex<T>& x0,
			      const T& alpha,
			      const vertex<T>& p) const;
      
        // current solution
	vertex<T> bestx; 
	T besty;
        volatile unsigned int iterations;
      
        bool overfit;  
	
        volatile bool sleep_mode, thread_running, solution_converged;
      
        volatile int thread_is_running;
        mutable std::mutex thread_is_running_mutex;
        mutable std::condition_variable thread_is_running_cond;
        
        std::thread* optimizer_thread;
        mutable std::mutex sleep_mutex, thread_mutex, solution_mutex;
	
      private:
	void optimizer_loop();
	
      };
    
  };
};



namespace whiteice
{
  namespace math
  {
    
    extern template class LBFGS< float >;
    extern template class LBFGS< double >;
    extern template class LBFGS< blas_real<float> >;
    extern template class LBFGS< blas_real<double> >;
    
  };
};

#endif
