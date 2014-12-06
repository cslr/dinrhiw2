
#include "BFGS.h"
#include "linear_equations.h"
#include <iostream>



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
      optimizer_thread = nullptr;
      
      this->overfit = overfit;
    }
    
    
    template <typename T>
    BFGS<T>::~BFGS()
    {
      thread_mutex.lock();
      
      if(thread_running){
	// pthread_cancel( optimizer_thread );
	thread_running = false;
	
	// waits for thread to stop running
	while(thread_is_running > 0){
	  std::unique_lock<std::mutex> lk(thread_is_running_mutex);
	  thread_is_running_cond.wait(lk);
	}
      }
      
      thread_mutex.unlock();
    }
    
    
    template <typename T>
    bool BFGS<T>::minimize(vertex<T>& x0)
    {
      thread_mutex.lock();
      
      if(thread_running){
	thread_mutex.unlock();
	return false;
      }
      
      // calculates initial solution
      solution_mutex.lock();
      {
	this->bestx = x0;
	this->besty = U(x0);
	iterations  = 0;
      }
      solution_mutex.unlock();

      thread_running = true;
      sleep_mode = false;
      solution_converged = false;
      thread_is_running = 0;
      
      try{
	optimizer_thread = new thread(std::bind(&BFGS<T>::optimizer_loop, this));
	optimizer_thread->detach();
      }
      catch(std::exception& e){
	thread_running = false;
	thread_mutex.unlock();
	return false;
      }
      
      thread_mutex.unlock();
      
      return true;
      
      return true;
    }
    
    
    template <typename T>
    bool BFGS<T>::getSolution(vertex<T>& x, T& y, unsigned int& iterations) const
    {
      // gets current solution
      std::lock_guard<std::mutex> lock(solution_mutex);

      x = bestx;
      y = besty;
      iterations = this->iterations;
      
      return true;
    }
    
    
    // continues, pauses, stops computation
    template <typename T>
    bool BFGS<T>::continueComputation()
    {
      std::lock_guard<std::mutex> lock(sleep_mutex);
      sleep_mode = false;
      
      return true;
    }
    
    
    template <typename T>
    bool BFGS<T>::pauseComputation()
    {
      std::lock_guard<std::mutex> lock(sleep_mutex);
      sleep_mode = true;
      
      return true;
    }
    
    
    template <typename T>
    bool BFGS<T>::stopComputation()
    {
      thread_mutex.lock();
      
      if(thread_running == false){
	thread_mutex.unlock();
	return false;
      }

      thread_running = false;
      
      {
	// waits for thread to stop running
	while(thread_is_running > 0){
	  std::unique_lock<std::mutex> lock(thread_is_running_mutex);
	  thread_is_running_cond.wait(lock);
	}
      }

      if(optimizer_thread)
	delete optimizer_thread;
      optimizer_thread = nullptr;
      
      thread_mutex.unlock();
      
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
    void BFGS<T>::optimizer_loop()
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
      
      thread_is_running++;
      thread_is_running_cond.notify_all();
      
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
	  
	  // cancellation point
	  {
	    thread_is_running--;
	    if(thread_running == false){
	      thread_is_running_cond.notify_all();
	      return;
	    }
	    thread_is_running++;
	  }
	
	  // linear search finds xn = x + alpha*d
	  // so that U(xn) is minimized
	  if(linesearch(xn, x, d) == false){
	    solution_converged = true;
	    break; // we stop computation as we cannot find better solution
	  }
	  
	  heuristics(xn);
	  
	  // cancellation point
	  {
	    thread_is_running--;
	    if(thread_running == false){
	      thread_is_running_cond.notify_all();
	      return;
	    }
	    thread_is_running++;
	  }
	  
	  y = U(xn);

	  // std::cout << "xn = " << xn << std::endl;
	  // std::cout << "H = " << H << std::endl;
	  // std::cout << "y = " << y << std::endl;
	  
	  if(y < besty){
	    std::lock_guard<std::mutex> lock(solution_mutex);
	    bestx = xn;
	    besty = y;
	  }
	  
	  // cancellation point
	  {
	    thread_is_running--;
	    if(thread_running == false){
	      thread_is_running_cond.notify_all();
	      return;
	    }
	    thread_is_running++;
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
	  std::chrono::milliseconds duration(1000);
	  std::this_thread::sleep_for(duration);
	}
      }
      
      
      thread_running = false;
      thread_is_running--;
      thread_is_running_cond.notify_all();
    }
    
    
    // explicit template instantations
    
    template class BFGS< float >;
    template class BFGS< double >;
    template class BFGS< blas_real<float> >;
    template class BFGS< blas_real<double> >;    
    
  };
};

