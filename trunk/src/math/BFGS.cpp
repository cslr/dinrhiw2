
#include "BFGS.h"
#include "linear_equations.h"
#include <iostream>
#include <pthread.h>
#include <time.h>

#ifdef WINNT
#include <windows.h>
#endif


extern "C" { static void* __bfgs_optimizer_thread_init(void *optimizer_ptr); };


namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    BFGS<T>::BFGS()
    {
      thread_running = false;
      pthread_mutex_init(&thread_lock, 0);
      pthread_mutex_init(&sleep_lock, 0);
      pthread_mutex_init(&solution_lock, 0);
    }
    
    
    template <typename T>
    BFGS<T>::~BFGS()
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
      
      pthread_mutex_lock( &solution_lock );

      thread_running = true;
      pthread_create(&optimizer_thread, 0,
		     __bfgs_optimizer_thread_init,
		     (void*)this);
      pthread_detach( optimizer_thread);
      
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
      pthread_mutex_lock( &solution_lock );
      
      return true;
    }
    
    
    // continues, pauses, stops computation
    template <typename T>
    bool BFGS<T>::continueComputation(){
      pthread_mutex_lock( &sleep_lock );
      if(sleep_mode){
	pthread_mutex_unlock( &sleep_lock );
	return false;
      }
      
      sleep_mode = true;
      pthread_mutex_unlock( &sleep_lock );
      
      return true;
    }
    
    
    template <typename T>
    bool BFGS<T>::pauseComputation(){
      pthread_mutex_lock( &sleep_lock );
      if(!sleep_mode){
	pthread_mutex_unlock( &sleep_lock );
	return true;
      }
      
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
      pthread_cancel( optimizer_thread );
      pthread_mutex_unlock( &thread_lock );
      
      return true;
    }


    template <typename T>
    void BFGS<T>::linesearch(vertex<T>& xn,
			     const vertex<T>& x,
			     const vertex<T>& d) const
    {
      // finds the correct scale first
      // (exponential search)

      T localbest = U(x + d);
      math::vertex<T> localbestx = x + d;
      
      for(int j=-6;j<=6;j++){
	float an = powf(2.0f, (float)j);

	math::vertex<T> t = x + T(an)*d;
	T tvalue = U(t);

	if(tvalue < localbest){
	  localbest = tvalue;
	  localbestx = t;
	}
      }

      xn = localbestx;
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
      
      
      while(thread_running){
	////////////////////////////////////////////////////////////
	g = Ugrad(x);
	d = -H*g; // linsolve(H, d, -g); 
	
	// linear search finds xn = x + alpha*d
	// so that U(xn) is minimized
	linesearch(xn, x, d); 

	y = U(xn);
	
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
	
	H += A*H*B;
	H += r*s.outerproduct();

	x = xn;

	iterations++;

#if 0	
	T scal = ((s*s)[0])/((s*(H*s))[0]);
	
	// slow, optimize with CBLAS H += a*(H^t * H)
	matrix<T> Q(H); // make copy of matrix
	Q.transpose();  // isn't needed
	
	H -= scal*Q*H; // makes copy of matrix
	
	
	// H += (q*q')/(q'*s) (optimize with CBLAS)
	scal = (q*s)[0];
	
	for(unsigned int j=0,index=0;j<H.ysize();j++)
	  for(unsigned int i=0;i<H.xsize();i++,index++)
	    H[index] += scal*q[j]*q[i];
	
	x = xn;
#endif
	////////////////////////////////////////////////////////////
	// checks if thread has been cancelled.
	pthread_testcancel();
	
	// checks if thread has been ordered to sleep
	while(sleep_mode){
#ifndef WINNT
	  struct timespec ts;
	  ts.tv_sec  = 0;
	  ts.tv_nsec = 500000000; // 500ms
	  nanosleep(&ts, 0);
#else
	  Sleep(500);
#endif
	}
      }
      
      
      // everything done. time to quit
      
      pthread_mutex_lock( &thread_lock );
      thread_running = false;
      pthread_mutex_unlock( &thread_lock );
    }
    
    
    // explicit template instantations
    
    template class BFGS< float >;
    template class BFGS< double >;
    template class BFGS< atlas_real<float> >;
    template class BFGS< atlas_real<double> >;    
    
  };
};




extern "C" {
  void* __bfgs_optimizer_thread_init(void *optimizer_ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(optimizer_ptr)
      ((whiteice::math::BFGS< whiteice::math::atlas_real<float> >*)optimizer_ptr)->__optimizerloop();
    
    pthread_exit(0);

    return 0;
  }
};
