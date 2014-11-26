
#include "BFGS.h"
#include "linear_equations.h"
#include <iostream>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#ifdef WINNT
#include <windows.h>
#endif


extern "C" { static void* __bfgs_optimizer_thread_init(void *optimizer_ptr); };


namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    BFGS<T>::BFGS(bool overfit)
    {
      thread_running = false;
      sleep_mode = false;
      solution_converged = false;
      pthread_mutex_init(&thread_lock, 0);
      pthread_mutex_init(&sleep_lock, 0);
      pthread_mutex_init(&solution_lock, 0);

      this->overfit = overfit;
    }
    
    
    template <typename T>
    BFGS<T>::~BFGS()
    {
      pthread_mutex_lock( &thread_lock );
      if(thread_running){
	// pthread_cancel( optimizer_thread );
	thread_running = false;
	
	// waits for thread to stop
	pthread_join( optimizer_thread, NULL);
      }
      pthread_mutex_unlock( &thread_lock );

      
      pthread_mutex_destroy(&thread_lock);
      pthread_mutex_destroy(&sleep_lock);
      pthread_mutex_destroy(&solution_lock);
    }
    
    
    template <typename T>
    bool BFGS<T>::minimize(vertex<T>& x0)
    {
      pthread_mutex_lock( &thread_lock );
      if(thread_running){
	pthread_mutex_unlock( &thread_lock );
	return false;
      }
      
      // calculates initial solution
      pthread_mutex_lock( &solution_lock );
      this->bestx = x0;
      this->besty = U(x0);
      iterations  = 0;
      pthread_mutex_unlock( &solution_lock );

      thread_running = true;
      sleep_mode = false;
      solution_converged = false;
      
      pthread_create(&optimizer_thread, 0,
		     __bfgs_optimizer_thread_init,
		     (void*)this);
      // pthread_detach( optimizer_thread);
      
      pthread_mutex_unlock( &thread_lock );
      
      return true;
    }
    
    
    template <typename T>
    bool BFGS<T>::getSolution(vertex<T>& x, T& y, unsigned int& iterations) const
    {
      // gets current solution
      pthread_mutex_lock( &solution_lock );
      x = bestx;
      y = besty;
      iterations = this->iterations;
      pthread_mutex_unlock( &solution_lock );
      
      return true;
    }
    
    
    // continues, pauses, stops computation
    template <typename T>
    bool BFGS<T>::continueComputation()
    {
      pthread_mutex_lock( &sleep_lock );
      sleep_mode = false;
      pthread_mutex_unlock( &sleep_lock );
      
      return true;
    }
    
    
    template <typename T>
    bool BFGS<T>::pauseComputation()
    {
      pthread_mutex_lock( &sleep_lock );
      sleep_mode = true;
      pthread_mutex_unlock( &sleep_lock );
      
      return false;
    }
    
    
    template <typename T>
    bool BFGS<T>::stopComputation()
    {
      pthread_mutex_lock( &thread_lock );
      if(!thread_running){
	pthread_mutex_unlock( &thread_lock );
	return false;
      }

      thread_running = false;
      pthread_join( optimizer_thread, NULL);
      // pthread_cancel( optimizer_thread );
      pthread_mutex_unlock( &thread_lock );
      
      return true;
    }


    // returns true if solution converged and we cannot
    // find better solution
    template <typename T>
    bool BFGS<T>::solutionConverged() const
    {
      return solution_converged;
    }
    
    // returns true if optimization thread is running
    template <typename T>
    bool BFGS<T>::isRunning() const
    {
      return thread_running;
    }
    

    template <typename T>
    bool BFGS<T>::linesearch(vertex<T>& xn,
			     const vertex<T>& x,
			     const vertex<T>& d) const
    {
      // finds the correct scale first
      // (exponential search)

      math::vertex<T> localbestx = x + d;
      T localbest  = T(1000000000.0f);
      // T best_alpha = T(1.0f);
      unsigned int found = 0;

      // best_alpha = 0.0f;
      localbestx = x;
      localbest = U(localbestx);
      

      unsigned int k = 0;

      while(found <= 0 && k <= 30){ // min 2**(-30) = 10e-9 step length

	T alpha = T(0.0f);
	T tvalue;
	
	alpha  = T(powf(2.0f, (float)k));
	tvalue = U(x + alpha*d);

	if(tvalue < localbest){
	  if(wolfe_conditions(x, alpha, d)){
	    // std::cout << "NEW SOLUTION FOUND" << std::endl;
	    localbest = tvalue;
	    localbestx = x + alpha*d;
	    // best_alpha = alpha;
	    found++;
	  }
	}

	alpha  = T(1.0f)/alpha;
	tvalue = U(x + alpha*d);

	if(tvalue < localbest){
	  if(wolfe_conditions(x, alpha, d)){
	    // std::cout << "NEW SOLUTION FOUND" << std::endl;
	    localbest = tvalue;
	    localbestx = x + alpha*d;
	    // best_alpha = alpha;
	    found++;
	  }
	}
	
	k++;
      }
      
      
      /*
      if(found <= 0)
	std::cout << "NO NEW SOLUTIONS FOUND: " << k 
		  << std::endl;
      else
	std::cout << "BEST ALPHA= " << best_alpha << std::endl;
      */
      
      xn = localbestx;

      return (found > 0);
    }
    
    
    template <typename T>
    bool BFGS<T>::wolfe_conditions(const vertex<T>& x0,
				   const T& alpha,
				   const vertex<T>& p) const
    {
      T c1 = T(0.0001f);
      T c2 = T(0.9f);
      
      bool cond1 = (U(x0 + alpha*p) <= (U(x0) + c1*alpha*(p*Ugrad(x0))[0]));
      bool cond2 = ((p*Ugrad(x0 + alpha*p))[0] >= c2*(p*Ugrad(x0))[0]);

      return (cond1 && cond2);
    }
    
    
    template <typename T>
    void BFGS<T>::__optimizerloop()
    {
      vertex<T> d, g; // gradient
      vertex<T> x(bestx), xn;
      vertex<T> s, q;
      T y;
      
      matrix<T> H; // H is INVERSE of hessian matrix
      H.resize(bestx.size(), bestx.size());
      H.identity();

      T prev_error = T(1000.0f);
      T error      = T(1000.0f);
      T ratio      = T(1000.0f);
      
      
      while(thread_running){
	try{
	  // we keep iterating until we converge (later) or
	  // the real error starts to increase
	  if(overfit == false){
	    prev_error = error;
	    error = getError(x);
	    ratio = (prev_error - error)/prev_error;
	    
	    if(ratio < T(0.0f)){
	      break;
	    }
	  }
	    

	  ////////////////////////////////////////////////////////////
	  g = Ugrad(x);
	  d = -H*g; // linsolve(H, d, -g);
	
	  // linear search finds xn = x + alpha*d
	  // so that U(xn) is minimized
	  if(linesearch(xn, x, d) == false){
	    solution_converged = true;
	    break; // we stop computation as we cannot find better solution
	  }
	  
	  y = U(xn);

	  // std::cout << "xn = " << xn << std::endl;
	  // std::cout << "H = " << H << std::endl;
	  // std::cout << "y = " << y << std::endl;
	  
	  if(y < besty){
	    pthread_mutex_lock( &solution_lock );
	    bestx = xn;
	    besty = y;
	    pthread_mutex_unlock( &solution_lock );
	  }
	  
	  // updates hessian approximation (BFGS method)
	  s = xn - x;
	  q = Ugrad(xn) - g; // Ugrad(xn) - Ugrad(x)
	  
	  T r = T(1.0f)/(s*q)[0];
	  
	  matrix<T> A;
	  A.resize(x.size(), x.size());
	  A.identity();
	  A = A - r*s.outerproduct(q);
	  
	  matrix<T> B;
	  B.resize(x.size(), x.size());
	  B.identity();
	  B = B - r*q.outerproduct(s);
	  
	  H = A*H*B;
	  H += r*s.outerproduct();
	  
	  x = xn;
	  
	  iterations++;
	}
	catch(std::exception& e){
	  std::cout << "ERROR: Unexpected exception: "
		    << e.what() << std::endl;
	}

	////////////////////////////////////////////////////////////
	// checks if thread has been cancelled.
	// pthread_testcancel();
	
	// checks if thread has been ordered to sleep
	while(sleep_mode){
	  sleep(1);
	}
      }
      
      
      // everything done. time to quit
      // THIS IS NOT THREAD-SAFE ?!?!
      
      if(pthread_mutex_trylock( &thread_lock ) == 0){
	thread_running = false;
	pthread_mutex_unlock( &thread_lock );
      }
      else{
	// cannot get the mutex
	// [something is happenind to thread_running]
	// so we just exit and let the mutex owner decide
	// what to do
      }
      
    }
    
    
    // explicit template instantations
    
    template class BFGS< float >;
    template class BFGS< double >;
    template class BFGS< blas_real<float> >;
    template class BFGS< blas_real<double> >;    
    
  };
};




extern "C" {
  void* __bfgs_optimizer_thread_init(void *optimizer_ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(optimizer_ptr)
      ((whiteice::math::BFGS< whiteice::math::blas_real<float> >*)optimizer_ptr)->__optimizerloop();
    
    pthread_exit(0);

    return 0;
  }
};
