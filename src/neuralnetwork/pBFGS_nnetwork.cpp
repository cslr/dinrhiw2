/*
 * parallel BFGS optimizer for neural networks
 *
 */


#include "pBFGS_nnetwork.h"
#include "deep_ica_network_priming.h"
#include <vector>
#include <unistd.h>


extern "C" { static void* __pbfgs_thread_init(void *optimizer_ptr); };


namespace whiteice
{
  
  template <typename T>
  pBFGS_nnetwork<T>::pBFGS_nnetwork(const nnetwork<T>& nn,
				    const dataset<T>& d) :
    net(nn), data(d)
  {
    thread_running = false;
    pthread_mutex_init(&bfgs_lock, 0);
  }

  
  template <typename T>
  pBFGS_nnetwork<T>::~pBFGS_nnetwork()
  {
    //TODO: stop all computation here

    pthread_mutex_lock( &bfgs_lock );

    if(thread_running){
      thread_running = false;
      pthread_cancel( updater_thread );
    }
    
    for(unsigned int i=0;i<optimizers.size();i++){
      optimizers[i]->stopComputation();
      delete optimizers[i];
      optimizers[i] = NULL;
    }

    optimizers.resize(0);
    
    pthread_mutex_unlock( &bfgs_lock );
    
    pthread_mutex_destroy(&bfgs_lock);
  }
  

  template <typename T>
  bool pBFGS_nnetwork<T>::minimize(unsigned int NUMTHREADS)
  {
    pthread_mutex_lock( &bfgs_lock );
    
    if(optimizers.size() > 0){
      pthread_mutex_unlock( &bfgs_lock );
      return false; // already running
    }


    optimizers.resize(NUMTHREADS);
    
    for(unsigned int i=0;i<optimizers.size();i++){
      optimizers[i] = NULL;
    }

    try{
      for(unsigned int i=0;i<optimizers.size();i++){
	optimizers[i] = new BFGS_nnetwork<T>(net, data);
	
	nnetwork<T> nn(this->net);
	nn.randomize();
	normalize_weights_to_unity(nn);
	
	math::vertex<T> w;
	nn.exportdata(w);

	if(i == 0){
	  global_best_x = w;
	  global_best_y = T(10e10);
	  global_iterations = 0;
	}
	
	optimizers[i]->minimize(w);
      }
    }
    catch(std::exception& e){
      for(unsigned int i=0;i<optimizers.size();i++){
	if(optimizers[i]){
	  delete optimizers[i];
	  optimizers[i] = NULL;
	}
      }

      thread_running = false;

      optimizers.resize(0);
      
      pthread_mutex_unlock( &bfgs_lock );
      
      return false;
    }


    thread_running = true;

    pthread_create(&updater_thread, 0,
		   __pbfgs_thread_init,
		   (void*)this);
    pthread_detach(updater_thread);
    

    pthread_mutex_unlock( &bfgs_lock );

    return true;
  }
  

  template <typename T>
  bool pBFGS_nnetwork<T>::getSolution(math::vertex<T>& x, T& y,
				      unsigned int& iterations) const
  {
    pthread_mutex_lock( &bfgs_lock );
    
    if(optimizers.size() <= 0){
      pthread_mutex_unlock( &bfgs_lock );
      return false;
    }

    x = global_best_x;
    y = global_best_y;    
    iterations = global_iterations;
    
    for(unsigned int i=0;i<optimizers.size();i++){
      math::vertex<T> _x;
      T _y;
      unsigned int iters = 0;

      if(optimizers[i] != NULL){
	if(optimizers[i]->getSolution(_x,_y,iters)){
	  if(_y < y){
	    y = _y;
	    x = _x;
	  }
	  iterations += iters;
	}
      }
    }

    pthread_mutex_unlock( &bfgs_lock );
    
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
      // e += T(0.5f)*err[0];
    }
    
    e /= T( (float)data.size(0) ); // per N

    return e;
  }

  
  // continues, pauses, stops computation
  template <typename T>
  bool pBFGS_nnetwork<T>::continueComputation()
  {
    pthread_mutex_lock( &bfgs_lock );
    
    if(optimizers.size() <= 0){
      pthread_mutex_unlock( &bfgs_lock );
      return false;
    }

    for(unsigned int i=0;i<optimizers.size();i++)
      optimizers[i]->continueComputation();

    pthread_mutex_unlock( &bfgs_lock );

    return true;
  }


  template <typename T>
  bool pBFGS_nnetwork<T>::pauseComputation()
  {
    pthread_mutex_lock( &bfgs_lock );
    
    if(optimizers.size() <= 0){
      pthread_mutex_unlock( &bfgs_lock );
      return false;
    }

    for(unsigned int i=0;i<optimizers.size();i++)
      optimizers[i]->pauseComputation();

    pthread_mutex_unlock( &bfgs_lock );

    return true;
  }


  template <typename T>
  bool pBFGS_nnetwork<T>::stopComputation()
  {
    pthread_mutex_lock( &bfgs_lock );
    
    if(optimizers.size() <= 0){
      pthread_mutex_unlock( &bfgs_lock );
      return false;
    }
    
    for(unsigned int i=0;i<optimizers.size();i++){
      optimizers[i]->stopComputation();
      delete optimizers[i];
      optimizers[i] = NULL;
    }

    thread_running = false;
    pthread_cancel( updater_thread );

    optimizers.resize(0);

    pthread_mutex_unlock( &bfgs_lock );

    return true;
  }


  template <typename T>
  void pBFGS_nnetwork<T>::__updater_loop()
  {
    
    while(1){
      sleep(1);
      
      pthread_testcancel();
	
      pthread_mutex_lock( &bfgs_lock );

      if(thread_running == false){
	pthread_mutex_unlock( &bfgs_lock );
	break;
      }

      
      // checks that if some BFGS thread has been converged or
      // not running anymore and recreates a new optimizer thread
      // after checking what the best solution found was
      
      try{
	
	for(unsigned int i=0;i<optimizers.size();i++){
	  if(optimizers[i] != NULL){
	    if(optimizers[i]->solutionConverged() ||
	       optimizers[i]->isRunning() == false){

	      math::vertex<T> x;
	      T y;
	      unsigned int iters = 0;

	      if(optimizers[i]->getSolution(x, y, iters)){
		if(y < global_best_y){
		  global_best_y = y;
		  global_best_x = x;
		}

		global_iterations += iters;
	      }
	      

	      delete optimizers[i];
	      optimizers[i] = NULL;
	    }
	  }

	  if(optimizers[i] == NULL){
	    optimizers[i] = new BFGS_nnetwork<T>(net, data);

	    nnetwork<T> nn(this->net);
	    nn.randomize();
	    normalize_weights_to_unity(nn);
	    
	    math::vertex<T> w;
	    nn.exportdata(w);
	    
	    optimizers[i]->minimize(w);
	  }
	}
      }
      catch(std::exception& e){
      }
      
      pthread_mutex_unlock( &bfgs_lock );
    }
  }
  
  
  //template class pBFGS_nnetwork< float >;
  //template class pBFGS_nnetwork< double >;
  template class pBFGS_nnetwork< math::blas_real<float> >;
  //template class pBFGS_nnetwork< math::blas_real<double> >;
  
};


extern "C" {
  void* __pbfgs_thread_init(void *optimizer_ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(optimizer_ptr)
      ((whiteice::pBFGS_nnetwork< whiteice::math::blas_real<float> >*)optimizer_ptr)->__updater_loop();
    
    pthread_exit(0);

    return 0;
  }
};



