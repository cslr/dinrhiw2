
#include "NNGradDescent.h"
#include <time.h>

#ifdef WINOS
#include <windows.h>
#endif


extern "C" {
  static void* __nngrad_optimizer_thread_init(void* param);
};


namespace whiteice
{
  namespace math
  {

    template <typename T>
    NNGradDescent<T>::NNGradDescent(bool negativefeedback)
    {
      best_error = T(1000.0f);
      converged_solutions = 0;
      data = NULL;
      NTHREADS = 0;
      this->negativefeedback = negativefeedback;

      running = false;
      nn = NULL;

      pthread_mutex_init(&solution_lock, 0);
      pthread_mutex_init(&start_lock, 0);
    }

    
    template <typename T>
    NNGradDescent<T>::~NNGradDescent()
    {
      pthread_mutex_lock( &start_lock );

      if(running)
	for(unsigned int i=0;i<optimizer_thread.size();i++){
	  pthread_cancel( optimizer_thread[i] );
	}

      running = false;
      
      if(nn) delete nn;
      nn = NULL;

      pthread_mutex_unlock( &start_lock );

      pthread_mutex_destroy( &solution_lock );
      pthread_mutex_destroy( &start_lock );
    }
    
    /*
     * starts the optimization process using data as 
     * the dataset as a training and testing data 
     * (implements early stopping)
     *
     * Executes NTHREADS in parallel when looking for
     * the optimal solution.
     */
    template <typename T>
    bool NNGradDescent<T>::startOptimize(const whiteice::dataset<T>& data,
					 const whiteice::nnetwork<T>& nn,
					 unsigned int NTHREADS,
					 unsigned int MAXITERS)
    {
      if(data.getNumberOfClusters() != 2) return false;
      if(data.size(0) != data.size(1)) return false;

      // need at least 50 datapoints
      if(data.size(0) <= 50) return false;

      if(data.dimension(0) != nn.input_size() ||
	 data.dimension(1) != nn.output_size())
	return false;
      
      pthread_mutex_lock( &start_lock );
      
      if(running == true){
	pthread_mutex_unlock( &start_lock );
	return false;
      }
      
      
      this->data = &data;
      this->NTHREADS = NTHREADS;
      this->MAXITERS = MAXITERS;
      best_error = T(1000.0f);
      converged_solutions = 0;
      running = true;
      thread_is_running = 0;
      
      this->nn = new nnetwork<T>(nn); // copies network (settings)
      nn.exportdata(bestx);

      optimizer_thread.resize(NTHREADS);

      for(unsigned int i=0;i<optimizer_thread.size();i++){
	pthread_create(& (optimizer_thread[i]), 0,
		       __nngrad_optimizer_thread_init, (void*)this);
	pthread_detach( optimizer_thread[i] );
      }
      
      
      pthread_mutex_unlock( &start_lock );

      return true;
    }
    

    
    /*
     * returns the best NN solution found so far and
     * its average error in testing dataset and the number
     * of converged solutions so far.
     */
    template <typename T>
    bool NNGradDescent<T>::getSolution(whiteice::nnetwork<T>& nn,
				       T& error, unsigned int& Nconverged)
    {
      // checks if the neural network architecture is the correct one
      if(this->nn == NULL) return false;

      pthread_mutex_lock( &solution_lock );

      nn = *(this->nn);
      nn.importdata(bestx);
	    
      error = best_error;
      Nconverged = converged_solutions;

      pthread_mutex_unlock( &solution_lock );

      return true;
    }

    
    /* used to pause, continue or stop the optimization process */
    template <typename T>
    bool NNGradDescent<T>::stopComputation()
    {
      pthread_mutex_lock( &solution_lock );
      pthread_mutex_lock( &start_lock );

      running = false;
      
      
      for(unsigned int i=0;i<optimizer_thread.size();i++){
	pthread_cancel( optimizer_thread[i] );
      }
      
      while(thread_is_running > 0){
	pthread_mutex_unlock( &solution_lock );
	sleep(1); // waits for threads to stop running
	pthread_mutex_lock( &solution_lock );
      }
      
      pthread_mutex_unlock( &start_lock );
      pthread_mutex_unlock( &solution_lock );

      return true;
    }


    template <typename T>
    void NNGradDescent<T>::__optimizerloop()
    {
      if(data == NULL)
	return; // silent failure if there is bad data
      
      thread_is_running++;
      
      // 1. divides data to to training and testing sets
      ///////////////////////////////////////////////////
      
      whiteice::dataset<T> dtrain, dtest;
      
      dtrain = *data;
      dtest  = *data;
      
      dtrain.clearData(0);
      dtrain.clearData(1);
      dtest.clearData(0);
      dtest.clearData(1);
      
      
      for(unsigned int i=0;i<data->size(0);i++){
	const unsigned int r = (rand() & 1);
	
	if(r == 0){
	  math::vertex<T> in  = data->access(0,i);
	  math::vertex<T> out = data->access(1,i);
	  
	  dtrain.add(0, in,  true);
	  dtrain.add(1, out, true);
	}
	else{
	  math::vertex<T> in  = data->access(0,i);
	  math::vertex<T> out = data->access(1,i);
	  
	  dtest.add(0, in,  true);
	  dtest.add(1, out, true);	    
	}
      }
      
      
      while(running){
	// keep looking for solution forever

	// starting location for neural network
	nnetwork<T> nn(*(this->nn));
	
	// use heuristic to normalize weights to unity (keep input weights)
	if(0){
	  nn.randomize();
	  normalize_weights_to_unity(nn); 
	  T alpha = T(0.5f);
	  negative_feedback_between_neurons(nn, dtrain, alpha);
	}

	T regularizer = T(1.0); // adds regularizer term to gradient (to work against overfitting)
	
	  
	// 2. normal gradient descent
	///////////////////////////////////////
	{
	  math::vertex<T> grad, err, weights;
	  math::vertex<T> prev_sumgrad;
	  
	  unsigned int counter = 0;
	  T prev_error, error, ratio;
	  T lrate = T(0.05f);
	  T delta_error = 0.0f;
	  
	  error = T(1000.0f);
	  prev_error = T(1000.0f);
	  ratio = T(1000.0f);
	  
	  while(error > T(0.001f) && 
		ratio > T(0.000001f) && 
		counter < MAXITERS)
	  {
	    prev_error = error;
	    error = T(0.0f);
	    
	    // goes through data, calculates gradient
	    // exports weights, weights -= lrate*gradient
	    // imports weights back

	    T ninv = T(1.0f/dtrain.size(0));
	    math::vertex<T> sumgrad;
	    

	    for(unsigned int i=0;i<dtrain.size(0);i++){
	      nn.input() = dtrain.access(0, i);
	      nn.calculate(true);
	      err = dtrain.access(1,i) - nn.output();
	      
	      if(nn.gradient(err, grad) == false)
		std::cout << "gradient failed." << std::endl;

	      if(i == 0)
		sumgrad = ninv*grad;
	      else
		sumgrad += ninv*grad;
	    }
	    
	    // cancellation point
	    {
	      thread_is_running--;
	      pthread_testcancel();
	      thread_is_running++;
	    }
	    	      
	    if(nn.exportdata(weights) == false)
	      std::cout << "export failed." << std::endl;

	    // ADDS STRONG REGULARIZER TERM TO GRADIENT!
	    sumgrad += regularizer*weights;

	    if(prev_sumgrad.size() <= 1){
	      weights -= lrate * sumgrad;
	    }
	    else{
	      T momentum = T(0.8f); // MOMENTUM TERM!
	      
	      weights -= lrate * sumgrad + momentum*prev_sumgrad;
	      prev_sumgrad = lrate * sumgrad;
	    }
	      
	    if(nn.importdata(weights) == false)
	      std::cout << "import failed." << std::endl;
	    

	    if(negativefeedback){
	      // using negative feedback heuristic 
	      T alpha = T(0.5f); // lrate;
	      negative_feedback_between_neurons(nn, dtrain, alpha);
	    }
	    
	    // calculates error from the testing dataset
	    for(unsigned int i=0;i<dtest.size(0);i++){
	      nn.input() = dtest.access(0, i);
	      nn.calculate(false);
	      err = dtest.access(1,i) - nn.output();
	      
	      for(unsigned int i=0;i<err.size();i++)
		error += (err[i]*err[i]) / T((float)err.size());
	    }
	    
	    error /= T((float)dtest.size(0));
	    
	    delta_error = (prev_error - error);
	    ratio = delta_error / error;

	    
	    // cancellation point
	    {
	      thread_is_running--;
	      pthread_testcancel();
	      thread_is_running++;
	    }

	    // printf("\r%d : %f (%f)                  ", counter, error.c[0], ratio.c[0]);
	    // fflush(stdout);
	    
	    counter++;
	  }
	
	  // printf("\r%d : %f (%f)                  \n", counter, error.c[0], ratio.c[0]);
	  // fflush(stdout);


	  // 3. after convergence checks if the result is better
	  //    than the earlier one
	  pthread_mutex_lock( &solution_lock );

	  if(error < best_error){
	    // improvement (smaller error with early stopping)
	    best_error = error;
	    nn.exportdata(bestx);
	  }

	  converged_solutions++;

	  pthread_mutex_unlock( &solution_lock);

	}
	
	
      }
      
      thread_is_running--;
    }


    
    template class NNGradDescent< float >;
    template class NNGradDescent< double >;
    template class NNGradDescent< blas_real<float> >;
    template class NNGradDescent< blas_real<double> >;    
    
  };
};


extern "C" {
  void* __nngrad_optimizer_thread_init(void *optimizer_ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(optimizer_ptr)
      ((whiteice::math::NNGradDescent< whiteice::math::blas_real<float> >*)optimizer_ptr)->__optimizerloop();
    
    pthread_exit(0);

    return 0;
  }
};
