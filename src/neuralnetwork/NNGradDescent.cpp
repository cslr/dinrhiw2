
#include "NNGradDescent.h"
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <functional>

#ifdef WINOS
#include <windows.h>
#endif

#include <sstream>
#include <memory>


namespace whiteice
{
  namespace math
  {

    template <typename T>
    NNGradDescent<T>::NNGradDescent(bool heuristics, bool deep_pretraining)
    {
      best_error = T(INFINITY);
      best_pure_error = T(INFINITY);
      iterations = 0;
      data = NULL;
      NTHREADS = 0;
      thread_is_running = 0;
      
      this->heuristics = heuristics;
      this->deep_pretraining = deep_pretraining;
      this->use_minibatch = false;

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
      this->deep_pretraining = grad.deep_pretraining;
      this->use_minibatch = grad.use_minibatch;

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


    template <typename T>
    void NNGradDescent<T>::setUseMinibatch(bool minibatch)
    {
      use_minibatch = minibatch;
    }

    
    template <typename T>
    bool NNGradDescent<T>::getUseMinibatch()
    {
      return use_minibatch;
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
      best_error = getError(nn, data, (regularizer>0.0f), dropout);
      if(dropout){
	auto nn_without_dropout = nn;
	nn_without_dropout.removeDropOut();
	best_pure_error = getError(nn_without_dropout, data, false, false);
      }
      else{
	best_pure_error = getError(nn, data, false, false);
      }

      std::cout << "INITIAL BEST PURE ERROR: " << best_pure_error << std::endl;
      std::cout << std::flush;
      
      {
	std::lock_guard<std::mutex> lock(noimprove_lock);
	noimprovements.clear();
      }

      {
	std::lock_guard<std::mutex> lock(convergence_lock);
	convergence.clear();
      }
      
      this->dropout = dropout;

      optimizer_thread.resize(NTHREADS);

      {
	std::lock_guard<std::mutex> lock(errors_lock);
	errors.clear();
      }
      
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

    
    template <typename T>
    bool NNGradDescent<T>::hasConverged(T percentage)
    {
      if(percentage <= T(0.0f)) return true;
      if(percentage >= T(1.0f)) return false;

      {
	std::lock_guard<std::mutex> lock(convergence_lock);
	
	if(convergence.size() != NTHREADS) return false;
	unsigned int threadnum = 0;

	for(const auto& c : convergence){
	  if(c.second == true) threadnum++;
	}

	if(threadnum == convergence.size())
	  return true;
      }

#if 0
      // this does not work anymore as we allow continuation
      // for sometime after better values are not found.
      {
	std::vector<bool> convergence;
	
	std::lock_guard<std::mutex> lock(errors_lock);

	if(errors.size() != NTHREADS) return false;
	unsigned int threadnum = 0;
	
	for(const auto& err : errors){
	  if(err.second.size() >= EHISTORY){
	    T m = T(0.0), v = T(0.0);
	    
	    for(const auto& e : err.second){
	      m += e;
	    }
	    
	    m /= err.second.size();

	    for(const auto& e : err.second){
	      v += (e - m)*(e - m);
	    }
	    
	    v /= (err.second.size() - 1);
	    
	    v = sqrt(abs(v));

	    std::cout << "THREAD " << threadnum << "/" << (int)errors.size()
		      << " (" << err.second.size() << " samples)"
		      << ": ERROR CONVERGENCE: " << T(100.0)*v/m << "% (convergence: "
		      << T(100.0)*percentage << "%)"
		      << std::endl;

	    if(v/m <= percentage) // 1% is a good value
	      convergence.push_back(true);
	    else
	      convergence.push_back(false);
	  }
	  else{
	    convergence.push_back(false);
	  }

	  threadnum++;
	}
	
	
	unsigned int cnum = 0;
	
	for(const auto& c : convergence){
	  if(c == true) cnum++;
	}
	
	if(cnum == NTHREADS)
	  return true;
      }
#endif

      return false;
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

      {
	solution_lock.lock();
	
	nn = *(this->nn);
	nn.importdata(bestx);
	
	error = best_pure_error;
	iterations = this->iterations;
	
	solution_lock.unlock();
      }

      return true;
    }


        template <typename T>
    bool NNGradDescent<T>::getSolutionStatistics(T& error,
						 unsigned int& iterations) const
    {
      // checks if the neural network architecture is the correct one
      if(this->nn == NULL) return false;

      {
	solution_lock.lock();
	
	error = best_pure_error;
	iterations = this->iterations;
	
	solution_lock.unlock();
      }
	
      return true;
    }


    template <typename T>
    bool NNGradDescent<T>::getSolution(whiteice::nnetwork<T>& nn) const
    {
      // checks if the neural network architecture is the correct one
      if(this->nn == NULL) return false;

      {
	solution_lock.lock();
	
	nn = *(this->nn);
	
	solution_lock.unlock();
      }
	
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
				 const bool regularize,
				 const bool dropout)
    {
      // error term is E[ 0.5*||y-f(x)||^2 ]
      T error = T(0.0);

      //const unsigned int MINIBATCHSIZE = 200; // number of samples used to estimate gradient
      
      // calculates initial error
#pragma omp parallel
      {
	T esum = T(0.0f);
	
	whiteice::nnetwork<T> nnet = net;
	math::vertex<T> err;

	// calculates error from the testing dataset
#pragma omp for nowait schedule(dynamic)	    	    
	for(unsigned int i=0;i<dtest.size(0);i++){
	  const unsigned int index = i; // rng.rand() % dtest.size(0);
	  math::vertex<T> out;
	  const auto& yvalue = dtest.access(1, index);

	  if(dropout) nnet.setDropOut();
	  
	  nnet.calculate(dtest.access(0, index), out);
	  
	  err = yvalue - out;

	  for(unsigned int i=0;i<err.size();i++)
	    esum += T(0.5)*(err[i]*err[i]);
	}
	
	esum /= T((float)dtest.size(0));
	// esum /= T((float)MINIBATCHSIZE);
	
#pragma omp critical
	{
	  error += esum;
	}
      }

      error /= T((float)dtest.access(1,0).size()); // divides per output dimension

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

      {
	std::lock_guard<std::mutex> lock(convergence_lock);
	
	convergence[std::this_thread::get_id()] = false;
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

      int counter = 0;

      while((dtrain.size(0) == 0 || dtrain.size(1) == 0 ||
	     dtest.size(0)  == 0 || dtest.size(1)  == 0) && counter < 10){

	dtrain.clearData(0);
	dtrain.clearData(1);
	dtest.clearData(0);
	dtest.clearData(1);
      
	for(unsigned int i=0;i<data->size(0);i++){
	  const unsigned int r = (rand() & 3);
	
	  if(r != 0){ // 75% will go to training data
	    math::vertex<T> in  = data->access(0,i);
	    math::vertex<T> out = data->access(1,i);
	  
	    dtrain.add(0, in,  true);
	    dtrain.add(1, out, true);
	  }
	  else{ // 25% will go to testing data
	    math::vertex<T> in  = data->access(0,i);
	    math::vertex<T> out = data->access(1,i);
	  
	    dtest.add(0, in,  true);
	    dtest.add(1, out, true);
	  }

	}

	counter++;
      }

      if(counter >= 10){ // too little data to divive datasets
	dtrain = *data;
	dtest  = *data;
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
	std::unique_ptr< nnetwork<T> > nn(new nnetwork<T>(*this->nn));

	{
	  char buffer[256];

	  std::ostringstream ss;
	  ss << std::this_thread::get_id();
	  std::string str_id = ss.str();
	  
	  snprintf(buffer, 256, "NNGradDescent: %d/%d (%s) reset/fresh neural network", iterations, MAXITERS, str_id.c_str());
	  whiteice::logging.info(buffer);
	}

	
	{
	  std::lock_guard<std::mutex> lock(first_time_lock);

	  // use heuristic to normalize weights to unity
	  // (keep input weights) [the first try is always given imported weights]

	  if(first_time == false){
#if 0
	    {
	      std::ostringstream ss;
	      ss << std::this_thread::get_id();
	      std::string str_id = ss.str();
		
	      printf("RANDOMIZE NEURAL NETWORK WEIGHTS (%s)\n",
		     str_id.c_str());
	      fflush(stdout);
	    }
#endif
	    
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
	    {
	      std::ostringstream ss;
	      ss << std::this_thread::get_id();
	      std::string str_id = ss.str();
	      
	      printf("USE PRESET NEURAL NETWORK WEIGHTS (%s)\n",
		     str_id.c_str());
	      fflush(stdout);
	    }
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

	  error = getError(*nn, dtest, (regularizer>0.0f), dropout);

	  {
	    solution_lock.lock();
	    
	    if(error < best_error){
	      // improvement (smaller error with early stopping)
	      
	      if(dropout){
		auto nn_without_dropout = *nn;
		nn_without_dropout.removeDropOut();
		
		const T gerror = getError(nn_without_dropout, *data,
					  false, false);
		
		if(gerror < best_pure_error){
		  nn_without_dropout.exportdata(bestx);
		  best_error = error;
		  best_pure_error = gerror;
		}
	      }
	      else{
		const T gerror = getError(*nn, *data, false, false);

		if(gerror < best_pure_error){
		  nn->exportdata(bestx);
		  best_error = error;
		  best_pure_error = gerror;
		}
	      }
	      
	    }
	    
	    solution_lock.unlock();
	  }

	  prev_error = error;

	  
	  // resets no improvement counter to check for convergence
	  // and sets best error for this loop iteration
	  T local_thread_best_error = T(INFINITY);
	  
	  {
	    std::lock_guard<std::mutex> lock(noimprove_lock);
	    
	    noimprovements[std::this_thread::get_id()] = 0;
	    local_thread_best_error = error;
	  }


	  T lrate = T(0.01f);
	  T ratio = T(1.0f);
	  
	  error = getError(*nn, dtrain, (regularizer>0.0f), dropout);
	  
	  do
	  {
	    prev_error = error;

	    // goes through data, calculates gradient
	    // exports weights, weights -= lrate*gradient
	    // imports weights back

	    math::vertex<T> sumgrad;
	    sumgrad.resize(nn->exportdatasize());
	    sumgrad.zero();

	    // number of samples used to estimate gradient in minibatch mode
	    const unsigned int MINIBATCHSIZE = 100; 

	    if(use_minibatch){
#pragma omp parallel shared(sumgrad)
	      {
		T ninv = T(1.0f/MINIBATCHSIZE);
		//T ninv = T(1.0f/dtrain.size(0));
		math::vertex<T> sgrad, grad;
		sgrad.resize(nn->exportdatasize());
		sgrad.zero();
		
		whiteice::nnetwork<T> nnet(*nn);
		math::vertex<T> err;
		
#pragma omp for nowait schedule(dynamic)
		for(unsigned int i=0;i<MINIBATCHSIZE;i++){
		  const unsigned int index = rng.rand() % dtrain.size(0);
		  // const unsigned int index = i;
		  
		  if(dropout) nnet.setDropOut();
		  
		  nnet.input() = dtrain.access(0, index);
		  nnet.calculate(true);
		  
		  err = dtrain.access(1,index) - nnet.output();
		  
		  if(nnet.gradient(err, grad) == false)
		    std::cout << "gradient failed." << std::endl;
		  
		  sgrad += ninv*grad;
		}
		
#pragma omp critical
		{
		  sumgrad += sgrad;
		}
	      
	      }
	    }
	    else{ // do not use minibatch but ALL data is used to compute gradient
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
		  const unsigned int index = i;
		  
		  if(dropout) nnet.setDropOut();
		  
		  nnet.input() = dtrain.access(0, index);
		  nnet.calculate(true);
		  
		  err = dtrain.access(1,index) - nnet.output();
		  
		  if(nnet.gradient(err, grad) == false)
		    std::cout << "gradient failed." << std::endl;
		  
		  sgrad += ninv*grad;
		}
		
#pragma omp critical
		{
		  sumgrad += sgrad;
		}
	      
	      }
	      
	    }
	    

	    {
	      char buffer[256];
	      double tmp = 0.0;
	      whiteice::math::convert(tmp, sumgrad.norm());
	      
	      snprintf(buffer, 256, "NNGradDescent: %d/%d gradient norm: %f", iterations, MAXITERS, tmp);
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
	    

	    // restarts gradient descent from lrate = 0.50
	    // lrate *= 4;
	    lrate = 0.50f;
	    
	    // line search: (we should maybe increase lrate to both directions lrate_next = 2.0*lrate and lrate_next2 = 0.5*lrate...
	    do{
	      weights = w0;
	      weights -= lrate * sumgrad;

	      if(nn->importdata(weights) == false)
		std::cout << "import failed." << std::endl;

	      if(heuristics){
		normalize_weights_to_unity(*nn);
#if 0
		if(whiten1d_nnetwork(*nn, dtrain) == false)
		  printf("ERROR: whiten1d_nnetwork failed\n");
		
		// using negative feedback heuristic 
		T alpha = T(0.5f); // lrate;
		negative_feedback_between_neurons(*nn, dtrain, alpha);
#endif
	      }

	      error = getError(*nn, dtrain, (regularizer>0.0f), dropout);

	      delta_error = (prev_error - error);
	      ratio = abs(delta_error) / abs(error);

	      if(delta_error < T(0.0)){ // if error grows we reduce learning rate
		lrate *= T(0.50);
	      }
	      else if(delta_error > T(0.0)){ // error becomes smaller we increase learning rate
		lrate *= T(1.0/0.50);
	      }

	      {
		char buffer[256];
		double tmp1, tmp2, tmp3, tmp4;
		whiteice::math::convert(tmp1, error);
		whiteice::math::convert(tmp2, delta_error);
		whiteice::math::convert(tmp4, ratio);
		whiteice::math::convert(tmp3, lrate);
		
		std::ostringstream ss;
		ss << std::this_thread::get_id();
		std::string str_id = ss.str();
		
		snprintf(buffer, 256,
			 "NNGradDescent: %d/%d (%s) linesearch error: %f delta-error: %f ratio: %f lrate: %e",
			 iterations, MAXITERS,
			 str_id.c_str(),
			 tmp1, tmp2, tmp4, tmp3);
		whiteice::logging.info(buffer);
	      }

	      // leaky error reduction, we sometimes allow jump to worse
	      // position in gradient direction
	      if((rng.rand() % 5) == 0 && error < 0.50)
		break;
	    }
	    while(delta_error < T(0.0) && lrate >= T(10e-25) && running);

	    
	    // replaces error with TESTing set error
	    error = getError(*nn, dtest, (regularizer>0.0f), dropout);

	    {
	      if(error > local_thread_best_error){
		// no improvement for this iteration
		std::lock_guard<std::mutex> lock(noimprove_lock);
		
		noimprovements[std::this_thread::get_id()]++;
	      }
	      else{ // improvement in this iteration
		// resets no improvement counter for this thread
		local_thread_best_error = error;
		
		std::lock_guard<std::mutex> lock(noimprove_lock);
		
		noimprovements[std::this_thread::get_id()] = 0;
	      }
	    }
	    
	    
	    {
	      char buffer[256];
	      double tmp1, tmp2, tmp3, tmp4;
	      whiteice::math::convert(tmp1, error);
	      whiteice::math::convert(tmp2, delta_error);
	      whiteice::math::convert(tmp4, ratio);
	      whiteice::math::convert(tmp3, lrate);
	      
	      std::ostringstream ss;
	      ss << std::this_thread::get_id();
	      std::string str_id = ss.str();
	      
	      snprintf(buffer, 256,
		       "NNGradDescent: %d/%d (%s) linesearch STOP noimprove counter: %d error: %f delta-error: %f ratio: %f lrate: %e",
		       iterations, MAXITERS,
		       str_id.c_str(),
		       noimprovements[std::this_thread::get_id()],
		       tmp1, tmp2, tmp4, tmp3);
	      whiteice::logging.info(buffer);
	    }

	    
	    nn->exportdata(weights);
	    w0 = weights;

	    {
	      if(error < best_error){
		std::lock_guard<std::mutex> lock(solution_lock);

		if(dropout){
		  auto nn_without_dropout = *nn;
		  nn_without_dropout.removeDropOut();
		  
		  const T gerror = getError(nn_without_dropout, *data,
					    false, false);
		  
		  if(gerror < best_pure_error){
		    nn_without_dropout.exportdata(bestx);
		    best_error = error;
		    best_pure_error = gerror;
		  }
		}
		else{
		  const T gerror = getError(*nn, *data, false, false);
		  
		  if(gerror < best_pure_error){
		    nn->exportdata(bestx);
		    best_error = error;
		    best_pure_error = gerror;
		  }
		}
	      }
	      
	      
	      // TODO: PUSH ALWAYS BEST ERROR TO SEE CONVERGENCE OF THREAD
	      // ONLY TRUE IMPROVEMENTS ALLOW KEEPING ITERATING
	      {
		std::lock_guard<std::mutex> lock(errors_lock);
		
		auto& e = errors[std::this_thread::get_id()];
		e.push_back(error);
		while(e.size() > EHISTORY) e.pop_front();
	      }

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
		noimprovements[std::this_thread::get_id()] < MAX_NOIMPROVEMENT_ITERS &&
		iterations < MAXITERS &&
		running);

	  {
	    // marks first convergence by this thread
	    // (restart from random point)
	    std::lock_guard<std::mutex> lock(convergence_lock);
	    
	    convergence[std::this_thread::get_id()] = true;
	  }

#if 0
	  {
	    // REMOVE THIS LATER
	    float errorf = 0.0f;
	    whiteice::math::convert(errorf, error);

	    std::ostringstream ss;
	    ss << std::this_thread::get_id();
	    std::string str_id = ss.str();
	    
	    printf("Iter: %d. Thread converged (%s). RESTART THREAD: %d %d %d %d: %f.\n",
		   iterations,
		   str_id.c_str(),
		   (int)(error > T(0.00001f)),
		   (int)noimprovements[std::this_thread::get_id()],
		   (int)(iterations < MAXITERS),
		   (int)(running),
		   (float)errorf);
	    fflush(stdout);
	  }
#endif
	
	  
	  // 3. after convergence checks if the result is better
	  //    than the earlier one
	  {
	    solution_lock.lock();
	    
	    if(error < best_error){
	      // improvement (smaller error with early stopping)
	      
	      if(dropout){
		auto nn_without_dropout = *nn;
		nn_without_dropout.removeDropOut();

		const T gerror = getError(nn_without_dropout, *data,
					  false, false);

		if(gerror < best_pure_error){
		  nn_without_dropout.exportdata(bestx);
		  best_error = error;
		  best_pure_error = gerror;
		}
	      }
	      else{
		const T gerror = getError(*nn, *data, false, false);

		if(gerror < best_pure_error){
		  nn->exportdata(bestx);
		  best_error = error;
		  best_pure_error = gerror;
		}
	      }

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
