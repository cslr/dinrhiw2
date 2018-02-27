/*
 * parallel BFGS optimizer for neural networks
 *
 */


#include "pBFGS_nnetwork.h"
#include "deep_ica_network_priming.h"
#include <vector>
#include <functional>
#include <unistd.h>



namespace whiteice
{
  
  template <typename T>
  pBFGS_nnetwork<T>::pBFGS_nnetwork(const nnetwork<T>& nn,
				    const dataset<T>& d,
				    bool overfit,
				    bool negativefeedback) :
    net(nn), data(d)
  {
    thread_running = false;
    this->overfit = overfit;
    this->negativefeedback = negativefeedback;
  }

  
  template <typename T>
  pBFGS_nnetwork<T>::~pBFGS_nnetwork()
  {
    // stops thread if needed
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(thread_running){
	thread_running = false;
	
	while(thread_is_running > 0){
	  std::unique_lock<std::mutex> lock(thread_is_running_mutex);
	  thread_is_running_cond.wait(lock);
	}
      }
    }


    {
      std::lock_guard<std::mutex> lock(bfgs_mutex);
      
      for(auto& o : optimizers)
	if(o.get() != nullptr)
	  o->stopComputation();
      
      optimizers.clear();
    }
    
  }
  

  template <typename T>
  bool pBFGS_nnetwork<T>::minimize(unsigned int NUMTHREADS)
  {
    thread_mutex.lock();

    if(thread_running){
      thread_mutex.unlock();
      return false; // already running
    }
    
    bfgs_mutex.lock();
    
    if(optimizers.size() > 0){
      bfgs_mutex.unlock();
      thread_mutex.unlock();
      return false; // already running
    }


    optimizers.resize(NUMTHREADS);
    
    for(auto& o : optimizers)
      o = nullptr;


    try{
      unsigned int index = 0;
      
      for(auto& o : optimizers){
	o.reset(new BFGS_nnetwork<T>(net, data, overfit, negativefeedback));
	
	nnetwork<T> nn(this->net);

	// we keep a single instance (i=0) of
	// the original nn in a starting set
	if(index != 0){
	  nn.randomize();
	  normalize_weights_to_unity(nn);
	  T alpha = T(0.5f);
	  negative_feedback_between_neurons(nn, data, alpha);
	}
	
	math::vertex<T> w;
	nn.exportdata(w);

	if(index == 0){
	  global_best_x = w;
	  global_best_y = T(10e10);
	  global_iterations = 0;
	}
	
	o->minimize(w);
      }
    }
    catch(std::exception& e){
      optimizers.clear();
      thread_running = false;

      bfgs_mutex.unlock();
      thread_mutex.unlock();
      
      return false;
    }

    bfgs_mutex.unlock();

    
    thread_running = true;
    
    thread_is_running_mutex.lock();
    thread_is_running = 0;
    thread_is_running_mutex.unlock();
    
    try{
      updater_thread = std::thread(std::bind(&pBFGS_nnetwork<T>::updater_loop, this));
      updater_thread.detach();
    }
    catch(std::exception& e){
      {
	std::lock_guard<std::mutex> lock(bfgs_mutex);
	optimizers.clear();
      }
      thread_running = false;
      
      return false;
    }
    
    thread_mutex.unlock();
    
    return true;
  }
  

  template <typename T>
  bool pBFGS_nnetwork<T>::getSolution(math::vertex<T>& x, T& y,
				      unsigned int& iterations) const
  {
    std::lock_guard<std::mutex> lock(bfgs_mutex);
    
    if(optimizers.size() <= 0)
      return false;

    x = global_best_x;
    y = global_best_y;    
    iterations = global_iterations;
    
    for(auto& o : optimizers){
      math::vertex<T> _x;
      T _y;
      unsigned int iters = 0;
      
      if(o->getSolution(_x,_y,iters)){
	if(_y < y){
	  y = _y;
	  x = _x;
	}
	iterations += iters;
      }
    }

    return true;
  }

  
  template <typename T>
  T pBFGS_nnetwork<T>::getError(const math::vertex<T>& x) const
  {
    whiteice::nnetwork<T> nnet(this->net);
    nnet.importdata(x);
    
    math::vertex<T> err;
    T e = T(0.0f);

    // E = SUM e(i)^2
    for(unsigned int i=0;i<data.size(0);i++){
      nnet.input() = data.access(0, i);
      nnet.calculate(false);
      err = data.access(1, i) - nnet.output();
      T inv = T(1.0f/err.size());
      err = inv*(err*err);
      e += err[0];
      e += T(0.5f)*err[0];
    }
    
    e /= T( (float)data.size(0) ); // per N

    return e;
  }

  
  // continues, pauses, stops computation
  template <typename T>
  bool pBFGS_nnetwork<T>::continueComputation()
  {
    std::lock_guard<std::mutex> lock(bfgs_mutex);
    
    if(optimizers.size() <= 0)
      return false;
    
    for(auto& o : optimizers)
      o->continueComputation();
    
    return true;
  }


  template <typename T>
  bool pBFGS_nnetwork<T>::pauseComputation()
  {
    std::lock_guard<std::mutex> lock(bfgs_mutex);
    
    if(optimizers.size() <= 0)
      return false;
    
    for(auto& o : optimizers)
      o->pauseComputation();
    
    return true;
  }


  template <typename T>
  bool pBFGS_nnetwork<T>::stopComputation()
  {
    {
      std::lock_guard<std::mutex> lock(thread_mutex);
      
      if(thread_running == false)
	return false;
      
      thread_running = false;
      
      while(thread_is_running > 0){
	std::unique_lock<std::mutex> lock(thread_is_running_mutex);
	thread_is_running_cond.wait(lock);
      }
      
    }
    
    {
      std::lock_guard<std::mutex> lock(bfgs_mutex);
      
      for(auto& o : optimizers)
	o->stopComputation();
      
      optimizers.clear();
    }
    
    return true;
  }


  template <typename T>
  void pBFGS_nnetwork<T>::updater_loop()
  {
    {
      {
	std::lock_guard<std::mutex> lock(thread_is_running_mutex);
	thread_is_running++;
      }
      thread_is_running_cond.notify_all();
    }
    
    while(thread_running){
      std::chrono::seconds duration(1);
      std::this_thread::sleep_for(duration);
      
      // checks that if some BFGS thread has been converged or
      // not running anymore and recreates a new optimizer thread
      // after checking what the best solution found was updated
      
      try{
	std::lock_guard<std::mutex> lock(bfgs_mutex);
	
	for(auto& o : optimizers){
	  if(o->solutionConverged() || o->isRunning() == false){
	    
	    math::vertex<T> x;
	    T y;
	    unsigned int iters = 0;
	    
	    if(o->getSolution(x, y, iters)){
	      if(y < global_best_y){
		global_best_y = y;
		global_best_x = x;
	      }
	      
	      global_iterations += iters;
	    }
	    
	    {
	      o.reset(new BFGS_nnetwork<T>(net, data, overfit, negativefeedback));

	      nnetwork<T> nn(this->net);
	      nn.randomize();
	      normalize_weights_to_unity(nn);
	      T alpha = T(0.5f);
	      negative_feedback_between_neurons(nn, data, alpha);
	      
	      math::vertex<T> w;
	      nn.exportdata(w);
	      
	      o->minimize(w);
	    }
	  }
	}
      }
      catch(std::exception& e){
      }
      
    }
    
    
    {
      std::lock_guard<std::mutex> lock(bfgs_mutex);
      
      for(auto& o : optimizers)
	o->stopComputation();
      
      optimizers.clear();
    }
    
    
    
    {
      {
	std::lock_guard<std::mutex> lock(thread_is_running_mutex);
	thread_is_running--;
      }
      thread_is_running_cond.notify_all();
    }    
    
    
  }
  
  
  template class pBFGS_nnetwork< float >;
  template class pBFGS_nnetwork< double >;
  template class pBFGS_nnetwork< math::blas_real<float> >;
  template class pBFGS_nnetwork< math::blas_real<double> >;
  
};

