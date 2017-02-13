
#include "PolicyGradAscent.h"


namespace whiteice
{

  template <typename T>
  PolicyGradAscent<T>::PolicyGradAscent()
  {
    best_value = T(-INFINITY);
    best_q_value = T(-INFINITY);
    iterations = 0;

    Q = NULL;
    data = NULL;
    policy = NULL;

    heuristics = false;
    dropout = false;

    first_time = true;
    
    NTHREADS = 0;
    MAXITERS = 0;

    running = false;
    thread_is_running = 0;

    regularizer = T(0.0001); // 1/10.000
    // regularizer = T(0.0); // regularization DISABLED
  }

  template <typename T>
  PolicyGradAscent<T>::PolicyGradAscent(const PolicyGradAscent<T>& grad)
  {
    best_value = grad.best_value;
    best_q_value = grad.best_q_value;
    iterations = grad.iterations;

    if(grad.Q)
      this->Q = new whiteice::nnetwork<T>(*grad.Q);
    else
      this->Q = NULL;

    this->data = grad.data;

    if(grad.policy)
      this->policy = new whiteice::nnetwork<T>(*grad.policy);
    else
      this->policy = NULL;

    heuristics = grad.heuristics;
    dropout = grad.dropout;
    regularizer = grad.regularizer;

    first_time = true;

    NTHREADS = 0;
    MAXITERS = 0;

    running = false;
    thread_is_running = 0;
  }

  template <typename T>
  PolicyGradAscent<T>::~PolicyGradAscent()
  {
    start_lock.lock();

    if(running){
      running = false;

      for(unsigned int i=0;i<optimizer_thread.size();i++){
	if(optimizer_thread[i]){
	  optimizer_thread[i]->join();
	  delete optimizer_thread[i];
	  optimizer_thread[i] = NULL;
	}
      }
    }

    if(Q) delete Q;
    if(policy) delete policy;

    Q = NULL;
    policy = NULL;

    start_lock.unlock();
  }


  
  /*
   * starts the optimization process using data as 
   * the dataset as a training and testing data 
   * (implements early stopping)
   *
   * Executes NTHREADS in parallel when looking for
   * the optimal solution and continues up to MAXITERS iterations.
   */
  template <typename T>
  bool PolicyGradAscent<T>::startOptimize(const whiteice::dataset<T>* data,
					  const whiteice::nnetwork<T>& Q,
					  // optimized policy
					  const whiteice::nnetwork<T>& policy, 
					  unsigned int NTHREADS,
					  unsigned int MAXITERS,
					  bool dropout)
  {
    if(data == NULL) return false;
    
    if(data->getNumberOfClusters() != 1) // dataset only contains state variables
      return false; 
    
    // need at least 10 datapoints
    if(data->size(0) <= 10) return false;
    
    if(data->dimension(0) != policy.input_size() ||
       data->dimension(0) + policy.output_size() != Q.input_size())
      return false;
    
    start_lock.lock();
    
    {
      std::lock_guard<std::mutex> lock(thread_is_running_mutex);
      if(thread_is_running > 0){
	start_lock.unlock();
	return false;
      }
    }
    

    // CANNOT become non-accessable during optimization
    this->data = data;
    
    this->NTHREADS = NTHREADS;
    this->MAXITERS = MAXITERS;
    best_value = T(-INFINITY);
    best_q_value = T(-INFINITY);
    iterations = 0;
    running = true;
    thread_is_running = 0;
    
    {
      std::lock_guard<std::mutex> lock(first_time_lock);
      first_time = true; // first thread uses weights from user supplied NN
    }

    // FIXME can run out of memory and throw exception!
    {
      auto newQ = new nnetwork<T>(Q); // copies network (settings)
      auto newpolicy = new nnetwork<T>(policy);
      
      if(this->Q) delete this->Q;
      if(this->policy) delete this->policy;
      
      this->Q = newQ;
      this->policy = newpolicy;
    }

    policy.exportdata(bestx);
    best_value = getValue(policy, Q, *data);
    best_q_value = getValue(policy, Q, *data, false);
    
    this->dropout = dropout;
    
    optimizer_thread.resize(NTHREADS);
    
    for(unsigned int i=0;i<optimizer_thread.size();i++){
      optimizer_thread[i] =
	new thread(std::bind(&PolicyGradAscent<T>::optimizer_loop,
			     this));
      
      // NON-STANDARD WAY TO SET THREAD PRIORITY (POSIX)
      {
	sched_param sch_params;
	int policy = SCHED_RR;
	
	pthread_getschedparam(optimizer_thread[i]->native_handle(),
			      &policy, &sch_params);
	
	sch_params.sched_priority = 20;
	if(pthread_setschedparam(optimizer_thread[i]->native_handle(),
				 SCHED_RR, &sch_params) != 0){
	}
      }
    }

    // waits for threads to start
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
  bool PolicyGradAscent<T>::isRunning()
  {
    std::lock_guard<std::mutex>  lock1(start_lock);
    std::unique_lock<std::mutex> lock2(thread_is_running_mutex);
    return running && (thread_is_running > 0);
  }

  
  /*
   * returns the best policy (nnetwork) solution found so far and
   * its average error in testing dataset and the number
   * iterations optimizer has executed so far
   */
  template <typename T>
  bool PolicyGradAscent<T>::getSolution(whiteice::nnetwork<T>& policy,
					T& value,
					unsigned int& iterations)
  {
    // checks if the neural network architecture is the correct one
    if(this->policy == NULL) return false;
    
    solution_lock.lock();
    
    policy = *(this->policy);
    policy.importdata(bestx);
    
    value = best_q_value;
    iterations = this->iterations;
    
    solution_lock.unlock();
    
    return true;
  }
  
  
  /* used to pause, continue or stop the optimization process */
  template <typename T>
  bool PolicyGradAscent<T>::stopComputation()
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

  //////////////////////////////////////////////////////////////////////
  
  // calculates mean Q-value of the policy in dtest dataset (states are inputs)
  template <typename T>
  T PolicyGradAscent<T>::getValue(const whiteice::nnetwork<T>& policy,
				  const whiteice::nnetwork<T>& Q, 
				  const whiteice::dataset<T>& dtest,
				  bool regularize) const
  {
    T value = T(0.0);
      
    // calculates mean q-value of policy
#pragma omp parallel
    {
      T vsum = T(0.0f);
      
      // calculates mean q-value from the testing dataset
#pragma omp for nowait schedule(dynamic)	    	    
      for(unsigned int i=0;i<dtest.size(0);i++){
	const auto& state = dtest.access(0, i);
	
	math::vertex<T> in(policy.input_size() + policy.output_size());
	in.zero();
	
	in.write_subvertex(state, 0);
	
	whiteice::math::vertex<T> action;
	
	policy.calculate(state, action);
	
	in.write_subvertex(action, state.size());

	whiteice::math::vertex<T> q;
	
	Q.calculate(in, q);
	
	vsum += q[0];
      }
	
      vsum /= T((double)dtest.size(0));
      
#pragma omp critical
      {
	value += vsum;
      }
    }

    
    if(regularize){
      // adds regularizer term (-0.5*||w||^2)
      whiteice::math::vertex<T> w;

      policy.exportdata(w);

      value -= regularizer * T(0.5) * (w*w)[0];
    }
    
    return value;
  }

  //////////////////////////////////////////////////////////////////////

  template <typename T>
  void PolicyGradAscent<T>::optimizer_loop()
  {
    
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

    {
      sched_param sch_params;
      int policy = SCHED_RR;
      
      pthread_getschedparam(pthread_self(), &policy, &sch_params);
      
      sch_params.sched_priority = 20;
      if(pthread_setschedparam(pthread_self(),
			       SCHED_RR, &sch_params) != 0){
      }
    }

    if(data == NULL){
      assert(0);
      return; // silent failure if there is bad data
    }
    
    if(data->size(0) <= 1 || running == false){
      assert(0);
      return;
    }

    
    
    // 1. divides data to to training and testing sets
    ///////////////////////////////////////////////////
    
    whiteice::dataset<T> dtrain, dtest;
    
    dtrain = *data;
    dtest  = *data;
    
    dtrain.clearData(0);
    dtest.clearData(0);
    
    while(dtrain.size(0) == 0 || dtest.size(0)  == 0){
      dtrain.clearData(0);
      dtest.clearData(0);
      
      for(unsigned int i=0;i<data->size(0);i++){
	const unsigned int r = (rand() & 1);
	
	if(r == 0){
	  dtrain.add(0, data->access(0, i),  true);
	}
	else{
	  dtest.add(0, data->access(0, i),  true);
	}
	
      }
    }


    
    while(running && iterations < MAXITERS){
      // keep looking for solution until MAXITERS
	
      // starting position for neural network
      whiteice::nnetwork<T> policy(*(this->policy));

      
      std::cout << "************ RESET POLICY NETWORK" << std::endl;


      {
	std::lock_guard<std::mutex> lock(first_time_lock);
	
	// use heuristic to normalize weights to unity (keep input weights) [the first try is always given imported weights]
	if(first_time == false){
	  policy.randomize();
	  
	  if(heuristics){
	    normalize_weights_to_unity(policy);	      
	  }
	}
	else{
	  first_time = false;
	}
      }

      
      // 2. normal gradient ascent
      ///////////////////////////////////////
      {
	math::vertex<T> weights, w0;
	  
	T prev_value, value;
	T delta_value = 0.0f;

	value = getValue(policy, *Q, dtest);

	{
	  solution_lock.lock();
	  
	  if(value > best_value){
	    // improvement (larger mean q-value of the policy)
	      best_value = value;
	      best_q_value = getValue(policy, *Q, dtest, false);
	      policy.exportdata(bestx);
	  }
	  
	  solution_lock.unlock();
	}

	T lrate = T(0.01);
	
	
	do{
	  prev_value = value;
	  
	  // goes through data, calculates gradient
	  // exports weights, weights -= lrate*gradient
	  // imports weights back

	  math::vertex<T> sumgrad;
	  sumgrad.resize(policy.exportdatasize());
	  sumgrad.zero();

#pragma omp parallel shared(sumgrad)
	  {
	    T ninv = T(1.0f/dtrain.size(0));
	    math::vertex<T> sgrad, grad;
	    grad.resize(policy.exportdatasize());
	    sgrad.resize(policy.exportdatasize());
	    sgrad.zero();

	    whiteice::nnetwork<T> pnet(policy);
	    math::vertex<T> err;
	      
#pragma omp for nowait schedule(dynamic)
	      for(unsigned int i=0;i<dtrain.size(0);i++){

		if(dropout) pnet.setDropOut();

		// calculates gradients for Q(state, action(state)) and policy(state)

		const auto& state = dtrain.access(0, i);

		whiteice::math::vertex<T> action;

		pnet.calculate(state, action);

		whiteice::math::vertex<T> in(state.size() + action.size());

		in.write_subvertex(state, 0);
		in.write_subvertex(action, state.size());

		whiteice::math::matrix<T> gradQ;
		{
		  whiteice::math::matrix<T> full_gradQ;
		  Q->gradient_value(in, full_gradQ);

		  full_gradQ.submatrix(gradQ,
				       state.size(), 0,
				       action.size(), 1);
		}

		whiteice::math::matrix<T> gradP;

		pnet.gradient(state, gradP);

		{
		  whiteice::math::matrix<T> g;

		  g = gradQ * gradP;

		  for(unsigned int i=0;i<policy.exportdatasize();i++)
		    grad[i] = g(0, i);
		}
		
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
		std::lock_guard<std::mutex> lock(thread_is_running_mutex);
		thread_is_running--;
		thread_is_running_cond.notify_all();
		return; // cancels execution
	      }
	    }
	    	      
	    if(policy.exportdata(weights) == false)
	      std::cout << "export failed." << std::endl;

	    w0 = weights;


	    // adds regularizer to gradient -(1/2*||w||^2)
	    {
	      sumgrad -= regularizer*w0;
	    }
	    
	    
	    lrate *= 4;
	    
	    do{
	      weights = w0;
	      weights += lrate * sumgrad;

	      if(policy.importdata(weights) == false)
		std::cout << "import failed." << std::endl;

	      if(dropout) policy.removeDropOut();
	      
	      if(heuristics){
		normalize_weights_to_unity(policy);
	      }

	      value = getValue(policy, *Q, dtrain);

	      delta_value = (prev_value - value);
	      
	      std::cout << "POLICY VALUE: " << value
			<< " PREV VALUE: " << prev_value
			<< " DELTA: " << delta_value
			<< " LRATE: " << lrate << std::endl;


	      // if value becomes smaller we reduce learning rate
	      if(delta_value >= T(0.0)){ 
		lrate *= T(0.50);
	      }
	      // value becomes larger we increase learning rate
	      else if(delta_value < T(0.0)){ 
		lrate *= T(1.0/0.50);
	      }
	      
	    }
	    while(delta_value >= T(0.0) && lrate >= T(10e-30) &&
		  abs(delta_value) > T(10e-12f) && running);

	    std::cout << "POLICY UPDATED."
		      << "VALUE: " << value 
		      << " DELTA: " << delta_value
		      << " LRATE: " << lrate
		      << std::endl;
	    
	    
	    policy.exportdata(weights);
	    w0 = weights;

	    iterations++;
	    
	    {
	      solution_lock.lock();
	      
	      if(value > best_value){
		// improvement (larger mean q-value of the policy)
		best_value = value;
		best_q_value = getValue(policy, *Q, dtest, false);
		policy.exportdata(bestx);
		
		std::cout << "************ BETTER POLICY FOUND: "
			  << best_q_value
			  << " ITER " << iterations << std::endl;
	      }
	    
	      solution_lock.unlock();
	    }

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
	while(abs(delta_value) > T(10e-12f) &&
	      lrate >= T(10e-30) && 
	      iterations < MAXITERS &&
	      running);

	
	  
	  // 3. after convergence checks if the result is better
	  //    than the earlier one
	  {
	    solution_lock.lock();
	    
	    if(value > best_value){
	      // improvement (larger mean q-value of the policy)
	      best_value = value;
	      best_q_value = getValue(policy, *Q, dtest, false);
	      policy.exportdata(bestx);

	      std::cout << "************ BETTER POLICY FOUND: "
			<< best_q_value
			<< " ITER " << iterations << std::endl;
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

  //////////////////////////////////////////////////////////////////////
  

  template class PolicyGradAscent< float >;
  template class PolicyGradAscent< double >;
  template class PolicyGradAscent< whiteice::math::blas_real<float> >;
  template class PolicyGradAscent< whiteice::math::blas_real<double> >;
};
