
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
    bool BFGS<T>::minimize(whiteice::optimized_function<T>* f,
			   vertex<T>& x0)
    {
      pthread_mutex_lock( &thread_lock );
      if(thread_running){
	pthread_mutex_unlock( &thread_lock );
	return false;
      }
      
      // calculates initial solution
      pthread_mutex_lock( &solution_lock );
      this->bestx = x0;
      f->calculate(x0, this->besty);
      H.resize(x0.size(), x0.size());
      H.identity();
      
      pthread_mutex_lock( &solution_lock );
      
      pthread_create(&optimizer_thread, 0,
		     __bfgs_optimizer_thread_init,
		     (void*)this);
      pthread_detach( optimizer_thread);
      thread_running = true;
      
      pthread_mutex_unlock( &thread_lock );
      
      return true;
    }
    
    
    template <typename T>
    bool BFGS<T>::getSolution(vertex<T>& x, T& y)
    {
      // gets current solution
      pthread_mutex_lock( &solution_lock );
      x = bestx;
      y = besty;
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
      
      pthread_cancel( optimizer_thread );
      thread_running = false;
      pthread_mutex_unlock( &thread_lock );
      
      return true;
    }
    
    
    template <typename T>
    void BFGS<T>::__optimizerloop()
    {
      vertex<T> d, g; // gradient
      vertex<T> x(bestx), xn;
      vertex<T> s, q;
      T y;
      
      
      if(f->hasGradient()){
	while(1){
	  ////////////////////////////////////////////////////////////
	  
	  f->grad(x, g);
	  linsolve(H, d, g);
	  
	  // linear search
	  xn = x + d; // really simple for now
	  
	  f->calculate(xn, y);
	  
	  if(y < besty){
	    pthread_mutex_lock( &solution_lock );
	    bestx = xn;
	    besty = y;
	    pthread_mutex_unlock( &solution_lock );
	  }
	  
	  // updates hessian
	  s = xn - x;
	  q = f->grad(xn) - f->grad(x);
	  
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
	  
	  ////////////////////////////////////////////////////////////
	  // checks if thread has been cancelled.
	  pthread_testcancel();
	  
	  // checks if thread has been ordered to sleep
	  while(sleep_mode){
#ifndef WINNT
	    struct timespec ts;
	    ts.tv_sec  = 0;
	    ts.tv_nsec = 10000000; // 10ms
	    nanosleep(&ts, 0);
#else
	    Sleep(10);
#endif
	  }
	}
	
      }
      else{ // don't have gradient, must approximate it
	while(1){
	  ////////////////////////////////////////////////////////////
	  
	  std::cout << "aprox gradient method hasn't been implemented yet."
		    << std::endl;
	  
	  
	  ////////////////////////////////////////////////////////////
	  // checks if thread has been cancelled.
	  pthread_testcancel();
	  
	  // checks if thread has been ordered to sleep
	  while(sleep_mode){
#ifndef WINNT
	    struct timespec ts;
	    ts.tv_sec  = 0;
	    ts.tv_nsec = 50000000; // 50ms
	    nanosleep(&ts, 0);
#else
	    Sleep(50);
#endif
	  }
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
