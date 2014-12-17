
#include "LBFGS.h"
#include "linear_equations.h"
#include <iostream>
#include <list>




namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    LBFGS<T>::LBFGS(bool overfit)
    {
      thread_running = false;
      sleep_mode = false;
      solution_converged = false;
      optimizer_thread = nullptr;
      
      this->overfit = overfit;
    }
    
    
    template <typename T>
    LBFGS<T>::~LBFGS()
    {
      thread_mutex.lock();
      
      if(thread_running){
	thread_running = false;

	// waits for thread to stop running
	while(thread_is_running > 0){
	  std::unique_lock<std::mutex> lock(thread_is_running_mutex);
	  thread_is_running_cond.wait(lock);
	}
	
	if(optimizer_thread)
	  delete optimizer_thread;
	optimizer_thread = nullptr;
      }
      
      thread_mutex.unlock();
    }
    
    
    template <typename T>
    bool LBFGS<T>::minimize(vertex<T> x0)
    {
      thread_mutex.lock();
      
      if(thread_running){
	thread_mutex.unlock();
	return false;
      }
      
      // calculates initial solution
      solution_mutex.lock();
      {
	heuristics(x0);
	
	this->bestx = x0;
	this->besty = getError(x0);
	
	iterations  = 0;
      }
      solution_mutex.unlock();

      thread_running = true;
      sleep_mode = false;
      solution_converged = false;
      thread_is_running = 0;
      
      try{
	optimizer_thread = new thread(std::bind(&LBFGS<T>::optimizer_loop, this));
	optimizer_thread->detach();
      }
      catch(std::exception& e){
	thread_running = false;
	thread_mutex.unlock();
	return false;
      }
      
      thread_mutex.unlock();
      
      return true;
    }
    
    
    template <typename T>
    bool LBFGS<T>::getSolution(vertex<T>& x, T& y, unsigned int& iterations) const
    {
      // gets the best found solution
      solution_mutex.lock();
      {
	x = bestx;
	y = besty;
	iterations = this->iterations;
      }
      solution_mutex.unlock();
      
      return true;
    }
    
    
    // continues, pauses, stops computation
    template <typename T>
    bool LBFGS<T>::continueComputation()
    {
      sleep_mutex.lock();
      {
	sleep_mode = false;
      }
      sleep_mutex.unlock();
      
      return true;
    }
    
    
    template <typename T>
    bool LBFGS<T>::pauseComputation()
    {
      sleep_mutex.lock();
      {
	sleep_mode = true;
      }
      sleep_mutex.unlock();
      
      return true;
    }
    
    
    template <typename T>
    bool LBFGS<T>::stopComputation()
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
	  std::unique_lock<std::mutex> lk(thread_is_running_mutex);
	  thread_is_running_cond.wait(lk);
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

      vertex<T> t;
      
      // best_alpha = 0.0f;
      localbestx = x;
      localbest = U(localbestx);
      
      int k = 0;
      T alpha = T(1.0f);

      while(found <= 0 && k <= 30){ // min 2**(-30) = 10e-9 step length
	
	alpha  = T(::pow(2.0f, k));
	T tvalue;

#if 0
	t = x + alpha*d;
	tvalue = U(t);
	
	if(tvalue < localbest){
	  // if(wolfe_conditions(x, alpha, d))
	  {
	    localbest = tvalue;
	    localbestx = t;
	    found++;
	    break;
	  }
	}
#endif

	alpha  = T(1.0f)/alpha;
	
	t = x + alpha*d;
	tvalue = U(t);

	if(tvalue < localbest){
	  // if(wolfe_conditions(x, alpha, d))
	  {
	    localbest = tvalue;
	    localbestx = t;
	    found++;
	    break;
	  }
	}
	
	k++;
      }
      
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
    void LBFGS<T>::optimizer_loop()
    {
      
      vertex<T> d, g; // gradient
      vertex<T> x(bestx), xn;
      vertex<T> s, q;
      
      T y          = besty;
      T ratio      = T(1.0f);
      
      std::list<T> ratios;
      bool reset = false;
      
      unsigned int M = 35; // history size
      std::list< vertex<T> > yk;
      std::list< vertex<T> > sk;
      std::list< T > rk;
      
      thread_is_running++;
      thread_is_running_cond.notify_all();
      
      while(thread_running){
	try{
	  
	  // we keep iterating until we converge (later) or
	  // the real error starts to increase
	  if(overfit == false){
	    ratio = y/besty;
	    
	    ratios.push_back(ratio);
	    while(ratios.size() > 10)
	      ratios.pop_front();
	    
	    T mean_ratio = 1000.0f;
	    // T inv = 1.0f/ratios.size();
	    
	    for(auto& r : ratios)
	      if(r < mean_ratio) 
		mean_ratio = r; // min
	    
	    // mean_ratio = math::pow(mean_ratio, inv);
	    
	    // std::cout << "ratio = " << mean_ratio << std::endl;
	    
	    // 10% increase from the minimum found
	    if(mean_ratio > T(1.10f) && iterations > 10){ 
	      break;
	    }
	  }

	  ////////////////////////////////////////////////////////////
	  g = Ugrad(x);
	  
	  // cancellation point
	  if(thread_running == false){
	    thread_is_running_mutex.lock();
	    thread_is_running--;
	    thread_is_running_mutex.unlock();
	    thread_is_running_cond.notify_all();
	    return;
	  }
	  
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
	  
	  
	  // cancellation point
	  if(thread_running == false){
	    thread_is_running_mutex.lock();
	    thread_is_running--;
	    thread_is_running_mutex.unlock();
	    thread_is_running_cond.notify_all();
	    return;
	  }
	  
	  // linear search finds xn = x + alpha*d
	  // so that U(xn) is minimized
	  if(linesearch(xn, x, d) == false){
	    // reset
	    sk.clear();
	    yk.clear();
	    rk.clear();
	    
	    if(reset == false){
	      reset = true;
	      // continue;
	    }
	    else{
	      // there was reset during the last iteration and 
	      // we still cannot improve the result
	      solution_converged = true;
	      break; // we stop computation as we cannot find better solution
	    }
	  }
	  else{
	    reset = false;
	  }
	  
	  if(iterations % 10 == 0)
	    heuristics(xn); // heuristically improve xn
	  
	  
	  // cancellation point
	  if(thread_running == false){
	    thread_is_running_mutex.lock();
	    thread_is_running--;
	    thread_is_running_mutex.unlock();
	    thread_is_running_cond.notify_all();
	    return;
	  }

	  
	  // y = U(xn);
	  y = getError(xn);
	    
	  if(y < besty){
	    std::lock_guard<std::mutex> lock(solution_mutex);
	    bestx = xn;
	    besty = y;	    
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
	  

	  // cancellation point
	  if(thread_running == false){
	    thread_is_running_mutex.lock();
	    thread_is_running--;
	    thread_is_running_mutex.unlock();
	    thread_is_running_cond.notify_all();
	    return;
	  }
	  
	  iterations++;
	}
	catch(std::exception& e){
	  std::cout << "ERROR: Unexpected exception: "
		    << e.what() << std::endl;
	}

	////////////////////////////////////////////////////////////
	
	// checks if thread has been ordered to sleep
	while(sleep_mode){
	  std::chrono::milliseconds duration(1000);
	  std::this_thread::sleep_for(duration);
	}
      }
      
      
      
      {
	thread_is_running_mutex.lock();
	thread_is_running--;
	thread_running = false;
	thread_is_running_mutex.unlock();
	thread_is_running_cond.notify_all();
      }
    }
    
    
    // explicit template instantations
    
    template class LBFGS< float >;
    template class LBFGS< double >;
    template class LBFGS< blas_real<float> >;
    template class LBFGS< blas_real<double> >;    
    
  };
};

