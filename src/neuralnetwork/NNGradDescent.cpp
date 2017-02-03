
#include "NNGradDescent.h"
#include <time.h>

#ifdef WINOS
#include <windows.h>
#endif



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
    }

    
    template <typename T>
    NNGradDescent<T>::~NNGradDescent()
    {
      start_lock.lock();

      if(running){
	running = false;
	for(unsigned int i=0;i<optimizer_thread.size();i++){
	  optimizer_thread[i]->join();
	  delete optimizer_thread[i];
	  optimizer_thread[i] = nullptr;
	}
      }

      if(nn) delete nn;
      nn = nullptr;

      start_lock.unlock();
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

      start_lock.lock();
      
      if(running == true){
	start_lock.unlock();
	return false;
      }
      
      
      this->data = &data;
      this->NTHREADS = NTHREADS;
      this->MAXITERS = MAXITERS;
      best_error = T(INFINITY);
      converged_solutions = 0;
      running = true;
      thread_is_running = 0;

      {
	std::lock_guard<std::mutex> lock(first_time_lock);
	first_time = true; // first thread uses weights from user supplied NN
      }
      
      this->nn = new nnetwork<T>(nn); // copies network (settings)
      nn.exportdata(bestx);
      best_error = getError(nn, data);

      optimizer_thread.resize(NTHREADS);

      for(unsigned int i=0;i<optimizer_thread.size();i++){
	optimizer_thread[i] =
	  new thread(std::bind(&NNGradDescent<T>::optimizer_loop,
			       this));
      }
      
      start_lock.unlock();

      return true;
    }
    
    template <typename T>
    bool NNGradDescent<T>::isRunning()
    {
      return running;
    }
    
    /*
     * returns the best NN solution found so far and
     * its average error in testing dataset and the number
     * of converged solutions so far.
     */
    template <typename T>
    bool NNGradDescent<T>::getSolution(whiteice::nnetwork<T>& nn,
				       T& error,
				       unsigned int& Nconverged)
    {
      // checks if the neural network architecture is the correct one
      if(this->nn == NULL) return false;

      solution_lock.lock();
      
      nn = *(this->nn);
      nn.importdata(bestx);
	    
      error = best_error;
      Nconverged = converged_solutions;

      solution_lock.unlock();

      return true;
    }

    
    /* used to pause, continue or stop the optimization process */
    template <typename T>
    bool NNGradDescent<T>::stopComputation()
    {
      start_lock.lock();
      solution_lock.lock();

      if(running == false){
	solution_lock.unlock();
	start_lock.unlock();
	return false; // not running
      }

      running = false;
      
      while(thread_is_running > 0){
	running = true;
	solution_lock.unlock();
	start_lock.unlock();
	sleep(1); // waits for threads to stop running
	start_lock.lock();
	solution_lock.lock();
	running = false;
	sleep(1);
      }

      solution_lock.unlock();
      start_lock.unlock();

      return true;
    }


    template <typename T>
    T NNGradDescent<T>::getError(const whiteice::nnetwork<T>& net,
				 const whiteice::dataset<T>& dtest)
    {
      T error = T(0.0);
      
      // calculates initial error
#pragma omp parallel
      {
	T esum = T(0.0f);
	
	const whiteice::nnetwork<T>& nnet = net;
	math::vertex<T> err;

	// calculates error from the testing dataset
#pragma omp for nowait schedule(dynamic)	    	    
	for(unsigned int i=0;i<dtest.size(0);i++){
	  math::vertex<T> out;
	  
	  nnet.calculate(dtest.access(0, i), out);
	  err = dtest.access(1,i) - out;

	  for(unsigned int i=0;i<err.size();i++)
	    esum += T(0.5)*(err[i]*err[i]);
	}
	
	esum /= T((float)dtest.size(0));
	
#pragma omp critical
	{
	  error += esum;
	}
      }

      return error;
    }


    template <typename T>
    void NNGradDescent<T>::optimizer_loop()
    {
      if(data == NULL)
	return; // silent failure if there is bad data
      
      if(data->size(0) <= 1 || running == false)
	return;

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

      while(dtrain.size(0) == 0 || dtrain.size(1) == 0 ||
	    dtest.size(0)  == 0 || dtest.size(1)  == 0){

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
      }

      while(running && converged_solutions < MAXITERS){
	// keep looking for solution forever
	
	// starting location for neural network
	nnetwork<T> nn(*(this->nn));

	{
	  std::lock_guard<std::mutex> lock(first_time_lock);
	  
	  // use heuristic to normalize weights to unity (keep input weights) [the first try is always given imported weights]
	  if(first_time == false){
	    nn.randomize();
	    normalize_weights_to_unity(nn); 
	    T alpha = T(0.5f);
	    negative_feedback_between_neurons(nn, dtrain, alpha);
	  }
	  else{
	    first_time = false;
	  }
	}

	unsigned int counter = 0;
	      
	// T regularizer = T(1.0); // adds regularizer term to gradient (to work against overfitting)
	  
	// 2. normal gradient descent
	///////////////////////////////////////
	{
	  math::vertex<T> weights, w0;
	  math::vertex<T> prev_sumgrad;
	  	  
	  T prev_error, error;	  
	  T delta_error = 0.0f;
	  
	  error = getError(nn, dtest);

	  {
	    solution_lock.lock();
	    
	    if(error < best_error){
	      // improvement (smaller error with early stopping)
	      best_error = error;
	      nn.exportdata(bestx);
	    }
	    
	    solution_lock.unlock();
	  }

	  prev_error = error;

	  T lrate = T(0.01f);
	  
	  while(error > T(0.001f) && 
		// ratio > T(0.00001f) && 
		counter < MAXITERS)
	  {
	    prev_error = error;

	    // std::cout << "*********************** RATIO " << ratio << std::endl;
	    // fflush(stdout);
	    
	    // goes through data, calculates gradient
	    // exports weights, weights -= lrate*gradient
	    // imports weights back

	    math::vertex<T> sumgrad;
	    sumgrad.resize(nn.exportdatasize());
	    sumgrad.zero();

#pragma omp parallel shared(sumgrad)
	    {
	      T ninv = T(1.0f/dtrain.size(0));
	      math::vertex<T> sgrad, grad;
	      sgrad.resize(nn.exportdatasize());
	      sgrad.zero();

	      whiteice::nnetwork<T> nnet(nn);
	      math::vertex<T> err;

#pragma omp for nowait schedule(dynamic)
	      for(unsigned int i=0;i<dtrain.size(0);i++){
		nnet.input() = dtrain.access(0, i);
		nnet.calculate(true);
		err = dtrain.access(1,i) - nnet.output();
		
		if(nnet.gradient(err, grad) == false)
		  std::cout << "gradient failed." << std::endl;
		
		sgrad += ninv*grad;
	      }

#pragma omp critical
	      {
		sumgrad += sgrad;
	      }
	      
	    }
	    
	    // cancellation point
	    {
	      if(running == false){
		thread_is_running--;
		return; // cancels execution
	      }
	    }
	    	      
	    if(nn.exportdata(weights) == false)
	      std::cout << "export failed." << std::endl;

	    w0 = weights;

#if 0
	    // ADDS STRONG REGULARIZER TERM TO GRADIENT!
	    sumgrad += regularizer*weights;
#endif

	    lrate *= 4;
	    
	    do{
	      nn.importdata(w0);
	      weights = w0;

	      weights -= lrate * sumgrad;

#if 0
	      if(prev_sumgrad.size() <= 1){
		weights -= lrate * sumgrad;
	      }
	      else{
		T momentum = T(0.8f); // MOMENTUM TERM!
		
		weights -= lrate * sumgrad + momentum*prev_sumgrad;		
	      }
#endif
	      
	      if(nn.importdata(weights) == false)
		std::cout << "import failed." << std::endl;
	      
	      
	      if(negativefeedback){
		// using negative feedback heuristic 
		T alpha = T(0.5f); // lrate;
		negative_feedback_between_neurons(nn, dtrain, alpha);
	      }

	      error = getError(nn, dtest);

	      delta_error = (prev_error - error);
	      // ratio = delta_error / error;

#if 0
	      std::cout << "ERROR = " << error << std::endl;
	      std::cout << "DELTA = " << delta_error << std::endl;
	      std::cout << "RATIO = " << ratio << std::endl;
#endif
	      
#if 1
	      if(delta_error < T(0.0)){ // if error grows we reduce learning rate
		lrate *= T(0.50);
		// std::cout << "NEW LRATE= " << lrate << std::endl;
	      }
	      else if(delta_error > T(0.0)){ // error becomes smaller we increase learning rate
		lrate *= T(1.0/0.50);
		// std::cout << "NEW LRATE= " << lrate << std::endl;
	      }
#endif

	    }
	    while(delta_error < T(0.0) && lrate != T(0.0));

	    // std::cout << "*******************************************************************" << error << std::endl;

	    prev_sumgrad = lrate * sumgrad;
	    
	    w0 = weights;
	    nn.importdata(w0);

	    {
	      solution_lock.lock();
	      
	      if(error < best_error){
		// improvement (smaller error with early stopping)
		best_error = error;
		nn.exportdata(bestx);
		
		// std::cout << "BEST ERROR = " << best_error << std::endl;
	      }
	    
	      solution_lock.unlock();
	    }

	    
	    // cancellation point
	    {
	      if(running == false){
		thread_is_running--;
		// printf("3: THEAD IS RUNNING: %d\n", thread_is_running);
		return; // stops execution
	      }
	    }

	    // printf("\r%d : %f (%f)                  ", counter, error.c[0], ratio.c[0]);
	    // fflush(stdout);
	    
	    counter++;
	  }
	
	  // printf("\r%d : %f (%f)                  \n", counter, error.c[0], ratio.c[0]);
	  // fflush(stdout);


	  // 3. after convergence checks if the result is better
	  //    than the earlier one
	  
	  solution_lock.lock();

	  if(error < best_error){
	    // improvement (smaller error with early stopping)
	    best_error = error;
	    nn.exportdata(bestx);

	    // std::cout << "BEST ERROR = " << best_error << std::endl;
	  }
	  
	  // converged_solutions++;
	  converged_solutions += counter;

	  solution_lock.unlock();
	}
	
	
      }

      start_lock.lock();
      {
	thread_is_running--;
	
	if(thread_is_running <= 0)
	  running = false;
      }
      start_lock.unlock();
      
      // printf("4: THEAD IS RUNNING: %d\n", thread_is_running);
      
      return;
    }


    
    template class NNGradDescent< float >;
    template class NNGradDescent< double >;
    template class NNGradDescent< blas_real<float> >;
    template class NNGradDescent< blas_real<double> >;    
    
  };
};
