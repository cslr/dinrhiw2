
#include "NNGradDescent.h"
#include <time.h>
#include <pthread.h>
#include <sched.h>

#ifdef WINOS
#include <windows.h>
#endif

#include <memory>


namespace whiteice
{
  namespace math
  {

    template <typename T>
    NNGradDescent<T>::NNGradDescent(bool heuristics, bool errorTerms, bool deep_pretraining)
    {
      best_error = T(INFINITY);
      best_pure_error = T(INFINITY);
      iterations = 0;
      data = NULL;
      NTHREADS = 0;
      thread_is_running = 0;
      
      this->heuristics = heuristics;
      this->errorTerms = errorTerms;
      this->deep_pretraining = deep_pretraining;

      dropout = false;

      running = false;
      nn = NULL;

      first_time = true;

      // regularizer = T(0.0001); // 1/10.000 (keep weights from becoming large)
      // regularizer = T(1.0); // this works for "standard" cases
      
      regularizer = T(0.0);    // regularizer is DISABLED
    }


    template <typename T>
    NNGradDescent<T>::NNGradDescent(const NNGradDescent<T>& grad)
    {
      best_error = grad.best_error;
      best_pure_error = grad.best_pure_error;
      iterations = grad.iterations;
      data = grad.data;
      NTHREADS = grad.NTHREADS;
      MAXITERS = grad.MAXITERS;      
      
      this->heuristics = grad.heuristics;
      this->errorTerms = grad.errorTerms;
      this->deep_pretraining = grad.deep_pretraining;

      dropout = grad.dropout;
      regularizer = grad.regularizer;

      running = grad.running;

      bestx = grad.bestx;

      if(grad.nn)
	nn = new whiteice::nnetwork<T>(*grad.nn);
      else
	nn = grad.nn;

      data = grad.data;

      first_time = grad.first_time;

      thread_is_running = 0;
      running = false;
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
     * 
     * initiallyUseNN = true => initially use given parameter nn weights
     */
    template <typename T>
    bool NNGradDescent<T>::startOptimize(const whiteice::dataset<T>& data,
					 const whiteice::nnetwork<T>& nn,
					 unsigned int NTHREADS,
					 unsigned int MAXITERS,
					 bool dropout,
					 bool initiallyUseNN)
    {
      if(data.getNumberOfClusters() != 2) return false;

      if(data.size(0) != data.size(1)) return false;

      // need at least 10 datapoints
      if(data.size(0) <= 10) return false;

      if(data.dimension(0) != nn.input_size() ||
	 data.dimension(1) != nn.output_size())
	return false;

      start_lock.lock();

      {
	std::lock_guard<std::mutex> lock(thread_is_running_mutex);
	if(thread_is_running > 0){
	  start_lock.unlock();
	  return false;
	}
      }
      
      
      this->data = &data;
      this->NTHREADS = NTHREADS;
      this->MAXITERS = MAXITERS;
      best_error = T(INFINITY);
      best_pure_error = T(INFINITY);
      iterations = 0;
      running = true;
      thread_is_running = 0;

      {
	std::lock_guard<std::mutex> lock(first_time_lock);
	// first thread uses weights from user supplied NN
	first_time = initiallyUseNN;
      }

      this->nn = new nnetwork<T>(nn); // copies network (settings)
      nn.exportdata(bestx);
      best_error = getError(nn, data);
      best_pure_error = getError(nn, data, false);
      
      this->dropout = dropout;

      optimizer_thread.resize(NTHREADS);
      
      for(unsigned int i=0;i<optimizer_thread.size();i++){
	optimizer_thread[i] =
	  new thread(std::bind(&NNGradDescent<T>::optimizer_loop,
			       this));
      }

      {
	std::unique_lock<std::mutex> lock(thread_is_running_mutex);

	// there is a bug if thread manages to notify and then continue and
	// reduce variable back to zero before this get chance to execute again
	while(thread_is_running == 0)
	  thread_is_running_cond.wait(lock);
      }

      
      start_lock.unlock();

      return true;
    }
    
    template <typename T>
    bool NNGradDescent<T>::isRunning()
    {
      std::lock_guard<std::mutex>  lock1(start_lock);
      std::unique_lock<std::mutex> lock2(thread_is_running_mutex);
      return running && (thread_is_running > 0);
    }
    
    /*
     * returns the best NN solution found so far and
     * its average error in testing dataset and the number
     * iterations optimizer has executed so far
     */
    template <typename T>
    bool NNGradDescent<T>::getSolution(whiteice::nnetwork<T>& nn,
				       T& error,
				       unsigned int& iterations) const
    {
      // checks if the neural network architecture is the correct one
      if(this->nn == NULL) return false;

      solution_lock.lock();
      
      nn = *(this->nn);
      nn.importdata(bestx);
	    
      error = best_pure_error;
      iterations = this->iterations;

      solution_lock.unlock();

      return true;
    }


        template <typename T>
    bool NNGradDescent<T>::getSolutionStatistics(T& error,
						 unsigned int& iterations) const
    {
      // checks if the neural network architecture is the correct one
      if(this->nn == NULL) return false;

      solution_lock.lock();
      
      error = best_pure_error;
      iterations = this->iterations;

      solution_lock.unlock();

      return true;
    }


    template <typename T>
    bool NNGradDescent<T>::getSolution(whiteice::nnetwork<T>& nn) const
    {
      // checks if the neural network architecture is the correct one
      if(this->nn == NULL) return false;

      solution_lock.lock();
      
      nn = *(this->nn);
      
      solution_lock.unlock();

      return true;
    }
    

    
    /* used to pause, continue or stop the optimization process */
    template <typename T>
    bool NNGradDescent<T>::stopComputation()
    {
      start_lock.lock();

      if(thread_is_running == 0 || running == false){
	start_lock.unlock();
	return false; // not running (anymore)
      }

      running = false;

      {
	std::unique_lock<std::mutex> lock(thread_is_running_mutex);

	while(thread_is_running > 0)
	  thread_is_running_cond.wait(lock);
      }

      start_lock.unlock();

      return true;
    }


    template <typename T>
    T NNGradDescent<T>::getError(const whiteice::nnetwork<T>& net,
				 const whiteice::dataset<T>& dtest,
				 bool regularize)
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
	  const auto& doi = dtest.access(1, i);
	  
	  nnet.calculate(dtest.access(0, i), out);

	  if(errorTerms == false){
	    err = doi - out;
	  }
	  else{
	    err.resize(out.size());
	    err.zero();

	    for(unsigned int k=0;k<doi.size();k++){
	      // NaNs are used to signal as not used fields/dimensions
	      if(whiteice::math::isnan(doi[k]) == false)
		err[k] = doi[k] - out[k];
	    }
	  }

	  for(unsigned int i=0;i<err.size();i++)
	    esum += T(0.5)*(err[i]*err[i]);
	}
	
	esum /= T((float)dtest.size(0));
	
#pragma omp critical
	{
	  error += esum;
	}
      }

      if(regularize){
	whiteice::math::vertex<T> w;

	net.exportdata(w);

	error += regularizer * T(0.5) * (w*w)[0];
      }

      return error;
    }


    template <typename T>
    void NNGradDescent<T>::optimizer_loop()
    {
      // set thread priority (non-standard)
      {
	sched_param sch_params;
	int policy = SCHED_FIFO;
	
	pthread_getschedparam(pthread_self(),
			      &policy, &sch_params);
	
#ifdef linux
	policy = SCHED_IDLE; // in linux we can set idle priority
#endif	
	sch_params.sched_priority = sched_get_priority_min(policy);
	
	if(pthread_setschedparam(pthread_self(),
				 policy, &sch_params) != 0){
	  // printf("! SETTING LOW PRIORITY THREAD FAILED\n");
	}

#ifdef WINOS
	SetThreadPriority(GetCurrentThread(),
			  THREAD_PRIORITY_IDLE);
#endif	
      }

      
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

      
      {
	std::lock_guard<std::mutex> lock(thread_is_running_mutex);
	thread_is_running++;
	thread_is_running_cond.notify_all();
      }

      
      // acquires lock temporally to wait for startOptimizer() to finish
      {
	start_lock.lock();
	start_lock.unlock();
      }

      
      while(running && iterations < MAXITERS){
	// keep looking for solution forever
	
	// starting location for neural network
	std::unique_ptr< nnetwork<T> > nn(new nnetwork<T>(*(this->nn)));

	{
	  char buffer[128];
	  snprintf(buffer, 128, "NNGradDescent: %d/%d reset/fresh neural network", iterations, MAXITERS);
	  whiteice::logging.info(buffer);
	}

	{
	  std::lock_guard<std::mutex> lock(first_time_lock);
	  
	  // use heuristic to normalize weights to unity
	  // (keep input weights) [the first try is always given imported weights]
	  if(first_time == false){
	    nn->randomize();

	    if(deep_pretraining){
	      auto ptr = nn.release();
	      // verbose = 2: logs training to log file..
	      if(deep_pretrain_nnetwork(ptr, dtrain, false, 2, &running) == false)
		whiteice::logging.error("NNGradDescent: deep pretraining FAILED");
	      else
		whiteice::logging.info("NNGradDescent: deep pretraining completed");
		
	      nn.reset(ptr);
	    }

	    
	    if(heuristics){
	      normalize_weights_to_unity(*nn);
	      
#if 0
	      if(whiten1d_nnetwork(*nn, dtrain) == false)
		printf("ERROR: whiten1d_nnetwork failed\n");
#endif
#if 0
	      normalize_weights_to_unity(*nn);
	      T alpha = T(0.5f);
	      negative_feedback_between_neurons(*nn, dtrain, alpha);
#endif
	    }
	    
	  }
	  else{
	    first_time = false;
	  }
	}

	// cancellation point
	{
	  if(running == false){
	    std::lock_guard<std::mutex> lock(thread_is_running_mutex);
	    thread_is_running--;
	    thread_is_running_cond.notify_all();
	    return; // cancels execution
	  }
	}
	    	      


	// 2. normal gradient descent
	///////////////////////////////////////
	{
	  math::vertex<T> weights, w0;
	  
	  T prev_error, error;	  
	  T delta_error = 0.0f;

	  error = getError(*nn, dtest);

	  {
	    solution_lock.lock();
	    
	    if(error < best_error){
	      // improvement (smaller error with early stopping)
	      best_error = error;
	      best_pure_error = getError(*nn, dtest, false);
	      nn->exportdata(bestx);
	    }
	    
	    solution_lock.unlock();
	  }

	  prev_error = error;

	  T lrate = T(0.01f);
	  T ratio = T(1.0f);

	  
	  do
	  {
	    prev_error = error;

	    // goes through data, calculates gradient
	    // exports weights, weights -= lrate*gradient
	    // imports weights back

	    math::vertex<T> sumgrad;
	    sumgrad.resize(nn->exportdatasize());
	    sumgrad.zero();

#pragma omp parallel shared(sumgrad)
	    {
	      T ninv = T(1.0f/dtrain.size(0));
	      math::vertex<T> sgrad, grad;
	      sgrad.resize(nn->exportdatasize());
	      sgrad.zero();

	      whiteice::nnetwork<T> nnet(*nn);
	      math::vertex<T> err;
	      
#pragma omp for nowait schedule(dynamic)
	      for(unsigned int i=0;i<dtrain.size(0);i++){

		if(dropout) nnet.setDropOut();
		
		nnet.input() = dtrain.access(0, i);
		nnet.calculate(true);

		if(errorTerms == false){
		  err = dtrain.access(1,i) - nnet.output();
		}
		else{
		  const auto& doi = dtrain.access(1,i);
		  err.resize(doi.size());
		  err.zero();
		  
		  for(unsigned int k=0;k<doi.size();k++)
		    if(whiteice::math::isnan(doi[k]) == false)
		      err[k] = doi[k] - nnet.output()[k];
		}
		
		if(nnet.gradient(err, grad) == false)
		  std::cout << "gradient failed." << std::endl;
		
		sgrad += ninv*grad;
	      }

#pragma omp critical
	      {
		sumgrad += sgrad;
	      }
	      
	    }

	    {
	      char buffer[80];
	      double tmp = 0.0;
	      whiteice::math::convert(tmp, sumgrad.norm());
	      
	      snprintf(buffer, 80, "NNGradDescent: %d/%d gradient norm: %f", iterations, MAXITERS, tmp);
	      whiteice::logging.info(buffer);
	    }


	    // cancellation point
	    {
	      if(running == false){
		std::lock_guard<std::mutex> lock(thread_is_running_mutex);
		thread_is_running--;
		thread_is_running_cond.notify_all();
		return; // cancels execution
	      }
	    }
	    	      
	    if(nn->exportdata(weights) == false)
	      std::cout << "export failed." << std::endl;

	    w0 = weights;

	    
	    // adds regularizer to gradient (1/2*||w||^2)
	    {
	      sumgrad += regularizer*w0;
	    }
	    
	    
	    lrate *= 4;

	    // line search: (we should maybe increase lrate to both directions lrate_next = 2.0*lrate and lrate_next2 = 0.5*lrate...
	    do{
	      weights = w0;
	      weights -= lrate * sumgrad;

	      if(nn->importdata(weights) == false)
		std::cout << "import failed." << std::endl;

	      if(dropout) nn->removeDropOut();
	      
	      if(heuristics){
		normalize_weights_to_unity(*nn);
#if 0
		if(whiten1d_nnetwork(*nn, dtrain) == false)
		  printf("ERROR: whiten1d_nnetwork failed\n");
#endif
		
#if 0
		// using negative feedback heuristic 
		T alpha = T(0.5f); // lrate;
		negative_feedback_between_neurons(*nn, dtrain, alpha);
#endif
	      }

	      error = getError(*nn, dtrain);

	      delta_error = (prev_error - error);
	      ratio = abs(delta_error) / abs(error);

	      if(delta_error < T(0.0)){ // if error grows we reduce learning rate
		lrate *= T(0.50);
	      }
	      else if(delta_error > T(0.0)){ // error becomes smaller we increase learning rate
		lrate *= T(1.0/0.50);
	      }

	      {
		char buffer[128];
		double tmp1, tmp2, tmp3, tmp4;
		whiteice::math::convert(tmp1, error);
		whiteice::math::convert(tmp2, delta_error);
		whiteice::math::convert(tmp4, ratio);
		whiteice::math::convert(tmp3, lrate);
		
		snprintf(buffer, 128, "NNGradDescent: %d/%d linesearch error: %f delta-error: %f ratio: %f lrate: %e", iterations, MAXITERS, tmp1, tmp2, tmp4, tmp3);
		whiteice::logging.info(buffer);
	      }
	    }
	    while(delta_error < T(0.0) && lrate >= T(10e-30) && running);
	    
	    {
	      char buffer[128];
	      double tmp1, tmp2, tmp3, tmp4;
	      whiteice::math::convert(tmp1, error);
	      whiteice::math::convert(tmp2, delta_error);
	      whiteice::math::convert(tmp4, ratio);
	      whiteice::math::convert(tmp3, lrate);
	      
	      snprintf(buffer, 128, "NNGradDescent: %d/%d linesearch STOP error: %f delta-error: %f ratio: %f lrate: %e", iterations, MAXITERS, tmp1, tmp2, tmp4, tmp3);
	      whiteice::logging.info(buffer);
	    }
	    
	    
	    nn->exportdata(weights);
	    w0 = weights;

	    {
	      solution_lock.lock();
	      
	      if(error < best_error){
		// improvement (smaller error with early stopping)
		best_error = error;
		best_pure_error = getError(*nn, dtrain, false);
		nn->exportdata(bestx);
	      }
	    
	      solution_lock.unlock();
	    }

	    iterations++;
	    
	    // cancellation point
	    {
	      if(running == false){
		std::lock_guard<std::mutex> lock(thread_is_running_mutex);
		thread_is_running--;
		thread_is_running_cond.notify_all();
		return; // stops execution
	      }
	    }

	    
	  }
	  while(error > T(0.00001f) && 
		lrate >= T(10e-30) && 
		iterations < MAXITERS && 
		running);

	
	  
	  // 3. after convergence checks if the result is better
	  //    than the earlier one
	  {
	    solution_lock.lock();
	    
	    if(error < best_error){
	      // improvement (smaller error with early stopping)
	      best_error = error;
	      best_pure_error = getError(*nn, dtrain, false);
	      nn->exportdata(bestx);
	    }
	    
	    solution_lock.unlock();
	  }
	}
	
	
      }

      
      std::lock_guard<std::mutex> lock(thread_is_running_mutex);
      thread_is_running--;
      thread_is_running_cond.notify_all();
      
      return;
    }


    
    template class NNGradDescent< float >;
    template class NNGradDescent< double >;
    template class NNGradDescent< blas_real<float> >;
    template class NNGradDescent< blas_real<double> >;    
    
  };
};
