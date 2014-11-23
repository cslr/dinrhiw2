
#include "LBFGS.h"
#include "linear_equations.h"
#include <iostream>
#include <list>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#ifdef WINNT
#include <windows.h>
#endif


extern "C" { static void* __lbfgs_optimizer_thread_init(void *optimizer_ptr); };


namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    LBFGS<T>::LBFGS()
    {
      thread_running = false;
      sleep_mode = false;
      solution_converged = false;
      pthread_mutex_init(&thread_lock, 0);
      pthread_mutex_init(&sleep_lock, 0);
      pthread_mutex_init(&solution_lock, 0);
    }
    
    
    template <typename T>
    LBFGS<T>::~LBFGS()
    {
      pthread_mutex_lock( &thread_lock );
      if(thread_running){
	pthread_cancel( optimizer_thread );
	thread_running = false;
      }
      pthread_mutex_unlock( &thread_lock );
      
      pthread_mutex_destroy(&thread_lock);
      pthread_mutex_destroy(&sleep_lock);
      pthread_mutex_destroy(&solution_lock);
    }
    
    
    template <typename T>
    bool LBFGS<T>::minimize(vertex<T>& x0)
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
		     __lbfgs_optimizer_thread_init,
		     (void*)this);
      pthread_detach( optimizer_thread);
      
      pthread_mutex_unlock( &thread_lock );
      
      return true;
    }
    
    
    template <typename T>
    bool LBFGS<T>::getSolution(vertex<T>& x, T& y, unsigned int& iterations) const
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
    bool LBFGS<T>::continueComputation()
    {
      pthread_mutex_lock( &sleep_lock );
      sleep_mode = false;
      pthread_mutex_unlock( &sleep_lock );
      
      return true;
    }
    
    
    template <typename T>
    bool LBFGS<T>::pauseComputation()
    {
      pthread_mutex_lock( &sleep_lock );
      sleep_mode = true;
      pthread_mutex_unlock( &sleep_lock );
      
      return false;
    }
    
    
    template <typename T>
    bool LBFGS<T>::stopComputation()
    {
      pthread_mutex_lock( &thread_lock );
      if(!thread_running){
	pthread_mutex_unlock( &thread_lock );
	return false;
      }

      thread_running = false;
      pthread_cancel( optimizer_thread );
      pthread_mutex_unlock( &thread_lock );
      
      return true;
    }


    // returns true if solution converged and we cannot
    // find better solution
    template <typename T>
    bool LBFGS<T>::solutionConverged() const
    {
      return solution_converged;
    }
    
    // returns true if optimization thread is running
    template <typename T>
    bool LBFGS<T>::isRunning() const
    {
      return thread_running;
    }
    

    template <typename T>
    bool LBFGS<T>::linesearch(vertex<T>& xn,
			     const vertex<T>& x,
			     const vertex<T>& d) const
    {
      // finds the correct scale first
      // (exponential search)

      vertex<T> localbestx = x + d;
      T localbest  = T(1000000000.0f);
      // T best_alpha = T(1.0f);
      unsigned int found = 0;

      // best_alpha = 0.0f;
      localbestx = x;
      localbest = U(localbestx);
      

      unsigned int k = 0;

      while(found <= 0 && k <= 20){
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
    bool LBFGS<T>::wolfe_conditions(const vertex<T>& x0,
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
    void LBFGS<T>::__optimizerloop()
    {
      vertex<T> d, g; // gradient
      vertex<T> x(bestx), xn;
      vertex<T> s, q;
      T y;
      
      T prev_error = T(1000.0f);
      T error      = T(1000.0f);
      T ratio      = T(1000.0f);

      
      unsigned int M = 35; // history size
      std::list< vertex<T> > yk;
      std::list< vertex<T> > sk;
      std::list< T > rk;
      
      
      while(thread_running){
	try{
	  // we keep iterating until we converge (later) or
	  // the real error starts to increase
	  prev_error = error;
	  error = getError(x);
	  ratio = (prev_error - error)/prev_error;

	  if(ratio < T(0.0f)){
	    break;
	  }


	  ////////////////////////////////////////////////////////////
	  g = Ugrad(x);

	  // d = -H*g; // linsolve(H, d, -g);	  
	  // calculates aprox hessian product (L-BFGS method)
	  if(sk.size() > 0){
	    vertex<T> q(g);

	    vertex<T> alpha;
	    vertex<T> beta;

	    alpha.resize(sk.size());
	    beta.resize(sk.size());
	    alpha.zero();
	    beta.zero();
	    
	    {
	      typename std::list< vertex<T> >::iterator si;
	      typename std::list< vertex<T> >::iterator yi;
	      typename std::list< T >::iterator ri;
	      unsigned int i = 0;
	      
	      for(si = sk.begin(), yi = yk.begin(), ri = rk.begin(), i=0;si!=sk.end();si++,yi++,ri++,i++)
	      {
		alpha[i] = (*ri) * ( ((*si)*q)[0] );
		q = q - alpha[i] * (*yi);
	      }
	    }

	    T Hk = ((*yk.begin()) * (*sk.begin()))[0] / ((*yk.begin())*(*yk.begin()))[0];

	    vertex<T> z = Hk*q;
	    
	    {
	      typename std::list< vertex<T> >::reverse_iterator si;
	      typename std::list< vertex<T> >::reverse_iterator yi;
	      typename std::list< T >::reverse_iterator ri;
	      int i = 0;

	      const unsigned int m = sk.size();
		
	      for(si = sk.rbegin(), yi = yk.rbegin(), ri = rk.rbegin(), i=m-1;si!=sk.rend();si++,yi++,ri++,i--)
	      {
		beta[i] = (*ri) * ((*yi) * z)[0];
		z = z + (*si) * (alpha[i] - beta[i]);
	      }
	      
	    }

	    d = -z;
	  }
	  else{
	    d = -g; // initially just follow the gradient
	  }
	  
	
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
	  
	  
	  s = xn - x;
	  vertex<T> yy = Ugrad(xn) - g; // Ugrad(xn) - Ugrad(x)
	  T r = T(1.0f)/(s*yy)[0];

	  sk.push_front(s);
	  yk.push_front(yy);
	  rk.push_front(r);

	  while(sk.size() > M) sk.pop_back();
	  while(yk.size() > M) yk.pop_back();
	  while(rk.size() > M) rk.pop_back();
	  
	  x = xn;
	  
	  iterations++;
	}
	catch(std::exception& e){
	  std::cout << "ERROR: Unexpected exception: "
		    << e.what() << std::endl;
	}

	////////////////////////////////////////////////////////////
	// checks if thread has been cancelled.
	pthread_testcancel();
	
	// checks if thread has been ordered to sleep
	while(sleep_mode){
	  sleep(1);
	}
      }
      
      
      // everything done. time to quit
      
      pthread_mutex_lock( &thread_lock );
      thread_running = false;
      pthread_mutex_unlock( &thread_lock );
    }
    
    
    // explicit template instantations
    
    template class LBFGS< float >;
    template class LBFGS< double >;
    template class LBFGS< blas_real<float> >;
    template class LBFGS< blas_real<double> >;    
    
  };
};




extern "C" {
  void* __lbfgs_optimizer_thread_init(void *optimizer_ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(optimizer_ptr)
      ((whiteice::math::LBFGS< whiteice::math::blas_real<float> >*)optimizer_ptr)->__optimizerloop();
    
    pthread_exit(0);

    return 0;
  }
};
