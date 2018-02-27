
#include "PolicyGradAscent.h"

#include "Log.h"
#include "blade_math.h"

#ifdef WINOS
#include <windows.h>
#endif

#include <functional>
#include <sched.h>

namespace whiteice
{

  template <typename T>
  PolicyGradAscent<T>::PolicyGradAscent(bool deep_pretraining)
  {
    best_value = T(-INFINITY);
    best_q_value = T(-INFINITY);
    iterations = 0;

    Q = NULL;
    Q_preprocess = NULL;
    data = NULL;
    policy = NULL;

    heuristics = false;
    dropout = false;
    regularize = false;
    this->deep_pretraining = deep_pretraining;
    
    debug = false;
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

    if(grad.Q_preprocess)
      this->Q_preprocess = new whiteice::dataset<T>(*grad.Q_preprocess);
    else
      this->Q_preprocess = NULL;

    if(grad.policy)
      this->policy = new whiteice::nnetwork<T>(*grad.policy);
    else
      this->policy = NULL;

    heuristics = grad.heuristics;
    dropout = grad.dropout;
    regularizer = grad.regularizer;
    deep_pretraining = grad.deep_pretraining;

    debug = grad.debug;
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
    if(Q_preprocess) delete Q_preprocess;
    if(policy) delete policy;

    Q = NULL;
    Q_preprocess = NULL;
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
					  const whiteice::dataset<T>& Q_preprocess,
					  // optimized policy
					  const whiteice::nnetwork<T>& policy, 
					  unsigned int NTHREADS,
					  unsigned int MAXITERS,
					  bool dropout,
					  bool initiallyUseNN)
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
      // first thread uses weights from user supplied NN
      first_time = initiallyUseNN;
    }

    // FIXME can run out of memory and throw exception!
    {
      auto newQ = new nnetwork<T>(Q); // copies network (settings)
      auto newpolicy = new nnetwork<T>(policy);
      auto newpreprocess = new dataset<T>(Q_preprocess);
      
      if(this->Q) delete this->Q;
      if(this->policy) delete this->policy;
      if(this->Q_preprocess) delete this->Q_preprocess;
      
      this->Q = newQ;
      this->Q_preprocess = newpreprocess;
      this->policy = newpolicy;

      whiteice::logging.info("PolicyGradAscent: input Q weights diagnostics");
      this->Q->diagnosticsInfo();

      whiteice::logging.info("PolicyGradAscent: input policy weights diagnostics");
      this->policy->diagnosticsInfo();
    }

    policy.exportdata(bestx);
    best_value = getValue(policy, Q, Q_preprocess, *data);
    best_q_value = getValue(policy, Q, Q_preprocess, *data);
    
    this->dropout = dropout;
    
    optimizer_thread.resize(NTHREADS);
    
    for(unsigned int i=0;i<optimizer_thread.size();i++){
      optimizer_thread[i] =
	new thread(std::bind(&PolicyGradAscent<T>::optimizer_loop,
			     this));
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
					unsigned int& iterations) const
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
  
  
  template <typename T>
  bool PolicyGradAscent<T>::getSolutionStatistics(T& value,
						  unsigned int& iterations) const
  {
    if(this->policy == NULL) return false;

    solution_lock.lock();

    value = best_q_value;
    iterations = this->iterations;

    solution_lock.unlock();

    return true;
  }

  
  template <typename T>
  bool PolicyGradAscent<T>::getSolution(whiteice::nnetwork<T>& policy) const
  {
    // checks if the neural network architecture is the correct one
    if(this->policy == NULL) return false;
    
    solution_lock.lock();
    
    policy = *(this->policy);
    policy.importdata(bestx);
    
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
				  const whiteice::dataset<T>& Q_preprocess,
				  const whiteice::dataset<T>& dtest) const
  {
    T value = T(0.0);
      
    // calculates mean q-value of policy
#pragma omp parallel
    {
      T vsum = T(0.0f);
      
      // calculates mean q-value from the testing dataset
#pragma omp for nowait schedule(dynamic)	    	    
      for(unsigned int i=0;i<dtest.size(0);i++){
	auto state = dtest.access(0, i);
	
	math::vertex<T> in(policy.input_size() + policy.output_size());
	in.zero();
	
	whiteice::math::vertex<T> action;

	policy.calculate(state, action);

	dtest.invpreprocess(0, state);

	in.write_subvertex(state, 0);
	in.write_subvertex(action, state.size());

	whiteice::math::vertex<T> q;

	Q_preprocess.preprocess(0, in);
	
	Q.calculate(in, q);

	Q_preprocess.invpreprocess(1, q);
	
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
    const bool regularize = false;
    
    {
      sched_param sch_params;
      int policy = SCHED_FIFO; // SCHED_RR
      
      pthread_getschedparam(pthread_self(), &policy, &sch_params);

#ifdef linux
      policy = SCHED_IDLE; // in linux we can set idle priority
#endif
      sch_params.sched_priority = sched_get_priority_min(policy);
      
      if(pthread_setschedparam(pthread_self(),
			       policy, &sch_params) != 0){
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

    dtrain.diagnostics();

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
      // keep looking for solution until MAXITERS
	
      // starting position for neural network
      // whiteice::nnetwork<T> policy(*(this->policy));
      std::unique_ptr< whiteice::nnetwork<T> > policy(new nnetwork<T>(*(this->policy)));

      whiteice::logging.info("PolicyGradAscent: reset policy network");
      
      
      {
	std::lock_guard<std::mutex> lock(first_time_lock);
	
	// use heuristic to normalize weights to unity (keep input weights) [the first try is always given imported weights]
	if(first_time == false){
	  policy->randomize();

	  if(deep_pretraining){
	    auto ptr = policy.release();
	    if(deep_pretrain_nnetwork_full_sigmoid(ptr, dtrain, false, 2, &running) == false)
	      whiteice::logging.error("PolicyGradAscent: deep pretraining FAILED");
	    else
	      whiteice::logging.info("PolicyGradAscent: deep pretraining completed");
	    
	    policy.reset(ptr);
	  }
	  
	  if(heuristics){
	    normalize_weights_to_unity(*policy);
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

	value = getValue(*policy, *Q, *Q_preprocess, dtest);

	{
	  solution_lock.lock();
	  
	  if(value > best_value){
	    // improvement (larger mean q-value of the policy)
	      best_value = value;
	      best_q_value = getValue(*policy, *Q, *Q_preprocess, dtest);
	      policy->exportdata(bestx);
	  }
	  
	  solution_lock.unlock();
	}

	
	T lrate = T(1.0);


	// calculates proper scaling for gradient terms..
	T q_grad_scaling = T(1.0);
	T p_grad_scaling = T(1.0);

#if 0
	{
	  std::vector<T> q_norms; // q gradient norms
	  std::vector<T> p_norms; // policy gradient norms
	  	    
#pragma omp parallel for schedule(dynamic)
	  for(unsigned int i=0;i<100;i++){
	    
	    // calculates gradients for Q(state, action(state)) and policy(state)
	    const unsigned int index = rng.rand() % dtrain.size(0);

	    auto state = dtrain.access(0, index); // preprocessed state vector
	    
	    whiteice::math::vertex<T> action;

	    policy->calculate(state, action);
	    
	    whiteice::math::matrix<T> gradP;
	    
	    policy->gradient(state, gradP);

	    const T p = frobenius_norm(gradP);

	    dtrain.invpreprocess(0, state); // original state for Q network
	      
	    whiteice::math::vertex<T> in(state.size() + action.size());
	      
	    in.write_subvertex(state, 0);
	    in.write_subvertex(action, state.size());
	      
	    Q_preprocess->preprocess(0, in);

	    whiteice::math::matrix<T> full_gradQ;

	    Q->gradient_value(in, full_gradQ);

	    whiteice::math::matrix<T> Qpostprocess_grad;
	    whiteice::math::matrix<T> Qpreprocess_grad_full;
		
	    Q_preprocess->preprocess_grad(0, Qpreprocess_grad_full);
	    Q_preprocess->invpreprocess_grad(1, Qpostprocess_grad);
		
	    whiteice::math::matrix<T> Qpreprocess_grad;
		
	    Qpreprocess_grad_full.submatrix(Qpreprocess_grad,
					    state.size(), 0,
					    action.size(),
					    Qpreprocess_grad_full.ysize());

	    
	    auto gradQ = Qpostprocess_grad * full_gradQ * Qpreprocess_grad;

	    const T q = frobenius_norm(gradQ);

#pragma omp critical
	    {
	      p_norms.push_back(p);
	      q_norms.push_back(q);
	    }
	    
	  }


	  T meanp = T(10e-10);

	  for(const auto& p : p_norms)
	    meanp += p / T(p_norms.size());

	  T meanq = T(10e-10);

	  for(const auto& q : q_norms)
	    meanq += q / T(q_norms.size());

	  
	  p_grad_scaling = T(1.0)/meanp;
	  q_grad_scaling = T(1.0)/meanq;
	}
#endif
	
	
	do{
	  prev_value = value;

	  whiteice::logging.info("PolicyGradAscent: calculates gradient");
	  
	  // goes through data, calculates gradient
	  // exports weights, weights -= lrate*gradient
	  // imports weights back

	  math::vertex<T> sumgrad;
	  sumgrad.resize(policy->exportdatasize());
	  sumgrad.zero();

#pragma omp parallel shared(sumgrad)
	  {
	    T ninv = T(1.0f/dtrain.size(0));
	    math::vertex<T> sgrad, grad;
	    grad.resize(policy->exportdatasize());
	    sgrad.resize(policy->exportdatasize());
	    sgrad.zero();

	    whiteice::nnetwork<T> pnet(*policy);
	    math::vertex<T> err;

	    char buffer[80];
	    double temp = 0.0;
	    
#pragma omp for nowait schedule(dynamic)
	    for(unsigned int i=0;i<dtrain.size(0);i++){
	      
	      if(dropout) pnet.setDropOut();
	      
	      // calculates gradients for Q(state, action(state)) and policy(state)

	      auto state = dtrain.access(0, i); // preprocessed state vector

	      if(debug)
	      {
		whiteice::math::convert(temp, state.norm());
		snprintf(buffer, 80, "PolicyGradientAscent: norm(state) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
	      
	      
	      whiteice::math::vertex<T> action;
	      
	      pnet.calculate(state, action);

	      if(debug)
	      {
		whiteice::math::convert(temp, action.norm());
		snprintf(buffer, 80, "PolicyGradientAscent: norm(action) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
	      
	      whiteice::math::matrix<T> gradP;
	      
	      pnet.gradient(state, gradP);

	      gradP *= p_grad_scaling;

	      if(debug)
	      {
		whiteice::math::convert(temp, frobenius_norm(gradP));
		snprintf(buffer, 80, "PolicyGradientAscent: norm(gradP) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
	      
	      dtrain.invpreprocess(0, state); // original state for Q network
	      if(debug)
	      {
		whiteice::math::convert(temp, state.norm());
		snprintf(buffer, 80, "PolicyGradientAscent: norm(invpreprocess(state)) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
	      
	      whiteice::math::vertex<T> in(state.size() + action.size());
	      
	      in.write_subvertex(state, 0);
	      in.write_subvertex(action, state.size());

	      if(debug)
	      {
		whiteice::math::convert(temp, in.norm());
		snprintf(buffer, 80, "PolicyGradientAscent: norm(in) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }

	      
	      Q_preprocess->preprocess(0, in);

	      if(debug)
	      {
		whiteice::math::convert(temp, in.norm());
		snprintf(buffer, 80, "PolicyGradientAscent: norm(preprocess(in)) = %e\n", 
			 temp);
		whiteice::logging.info(buffer);
	      }
	      
	      whiteice::math::matrix<T> gradQ;
	      {
		whiteice::math::matrix<T> full_gradQ;
		Q->gradient_value(in, full_gradQ);

		full_gradQ *= q_grad_scaling;

		if(debug)
		{
		  whiteice::logging.info("PolicyGradientAscent: Q-network diagnostics");
		  Q->diagnosticsInfo(); // login
		  
		  whiteice::math::convert(temp, frobenius_norm(full_gradQ));
		  snprintf(buffer, 80,
			   "PolicyGradientAscent: norm(full_gradQ) = %e\n", 
			   temp);
		  whiteice::logging.info(buffer);
		}
		
		whiteice::math::matrix<T> Qpostprocess_grad;
		whiteice::math::matrix<T> Qpreprocess_grad_full;
		
		Q_preprocess->preprocess_grad(0, Qpreprocess_grad_full);
		Q_preprocess->invpreprocess_grad(1, Qpostprocess_grad);
		
		whiteice::math::matrix<T> Qpreprocess_grad;
		
		Qpreprocess_grad_full.submatrix(Qpreprocess_grad,
						state.size(), 0,
						action.size(),
						Qpreprocess_grad_full.ysize());

		if(debug)
		{
		  whiteice::math::convert(temp, frobenius_norm(Qpreprocess_grad_full));
		  snprintf(buffer, 80, "PolicyGradientAscent: norm(Qpreprocess_grad_full) = %e\n", 
			   temp);
		  whiteice::logging.info(buffer);
		}

		if(debug)
		{
		  whiteice::math::convert(temp, frobenius_norm(Qpreprocess_grad));
		  snprintf(buffer, 80, "PolicyGradientAscent: norm(Qpreprocess_grad) = %e\n", 
			   temp);
		  whiteice::logging.info(buffer);
		}

		if(debug)
		{
		  whiteice::math::convert(temp, frobenius_norm(Qpostprocess_grad));
		  snprintf(buffer, 80, "PolicyGradientAscent: norm(Qpostprocess_grad) = %e\n", 
			   temp);
		  whiteice::logging.info(buffer);
		}
		
		gradQ = Qpostprocess_grad * full_gradQ * Qpreprocess_grad;

		if(debug)
		{
		  whiteice::math::convert(temp, frobenius_norm(gradQ));
		  snprintf(buffer, 80, "PolicyGradientAscent: norm(gradQ) = %e\n", 
			   temp);
		  whiteice::logging.info(buffer);
		}
	      }
	      
	      {
		whiteice::math::matrix<T> g;
		
		g = gradQ * gradP;
		
		for(unsigned int i=0;i<policy->exportdatasize();i++)
		  grad[i] = g(0, i);
	      }

	    
	      sgrad += ninv*grad;
	      // sgrad += grad;
	    }
	    
#pragma omp critical
	    {
	      sumgrad += sgrad;
	    }
	  }

	  {
	    char buffer[80];
	    double gradlen = 0.0;
	    
	    whiteice::math::convert(gradlen, sumgrad.norm());

	    snprintf(buffer, 80, "PolicyGradAscent: raw gradient norm: %e",
		     gradlen);
	    
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
	  
	  if(policy->exportdata(weights) == false)
	    whiteice::logging.error("PolicyGradAscent: weight export failed");
	  
	  w0 = weights;
	  
	  
	  // adds regularizer to gradient -(1/2*||w||^2)
	  if(regularize)
	  {
	    sumgrad -= regularizer*w0;
	  }

	  if(debug)
	    whiteice::logging.info("PolicyGradAscent: calculates linesearch..");

	  // sumgrad.normalize(); // normalizes gradient length to unit..
	  
	  lrate *= T(4.0);
	  
	  do{
	    weights = w0;
	    weights += lrate * sumgrad;
	    
	    if(policy->importdata(weights) == false)
	      whiteice::logging.error("PolicyGradAscent: weight import failed");
	    
	    if(dropout) policy->removeDropOut();
	    
	    if(heuristics){
	      normalize_weights_to_unity(*policy);
	    }
	    
	    value = getValue(*policy, *Q, *Q_preprocess, dtrain);
	    
	    delta_value = (prev_value - value);
	    
	    {
	      char buffer[128];
	      
	      double v, p, d, l;
	      whiteice::math::convert(v, value);
	      whiteice::math::convert(p, prev_value);
	      whiteice::math::convert(d, delta_value);
	      whiteice::math::convert(l, lrate);
	      
	      snprintf(buffer, 128,
		       "PolicyGradAscent: gradstep curvalue %f prevvalue %f delta %e lrate %e\n", v, p, d, l);
	      whiteice::logging.info(buffer);
	    }
	    
	    // if value becomes smaller we reduce learning rate
	    if(delta_value >= T(0.0)){ 
	      lrate *= T(0.50);
	    }
	    // value becomes larger we increase learning rate
	    else if(delta_value < T(0.0)){ 
	      lrate *= T(1.0/0.50);
	    }
	    
	  }
	  while(delta_value > T(0.0) && lrate >= T(10e-30) &&
		abs(delta_value) > T(10e-30f) && running);
	  
	  {
	    char buffer[128];
	    
	    double v, d, l;
	    whiteice::math::convert(v, value);
	    whiteice::math::convert(d, delta_value);
	    whiteice::math::convert(l, lrate);
	    
	    snprintf(buffer, 128,
		     "PolicyGradAscent: policy updated value %f delta %e lrate %e\n",
		     v, d, l);
	    whiteice::logging.info(buffer);

	    whiteice::logging.info("PolicyGradientAscent: policy-network diagnostics");
	    policy->diagnosticsInfo();
	  }
	  
	  policy->exportdata(weights);
	  w0 = weights;
	  
	  iterations++;
	  
	  {
	    solution_lock.lock();
	      
	    if(value > best_value){
	      // improvement (larger mean q-value of the policy)
	      best_value = value;
	      best_q_value = getValue(*policy, *Q, *Q_preprocess, dtest);
	      policy->exportdata(bestx);
	      
	      {
		char buffer[128];
		
		double b;
		whiteice::math::convert(b, best_q_value);
		
		snprintf(buffer, 128,
			 "PolicyGradAscent: better policy found: %e iter %d",
			 b, iterations);
		whiteice::logging.info(buffer);
	      }
	      
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
	while(abs(delta_value) > T(10e-30f) &&
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
	    best_q_value = getValue(*policy, *Q, *Q_preprocess, dtest);
	    policy->exportdata(bestx);
	    
	    {
	      char buffer[128];
	      
	      double b;
	      whiteice::math::convert(b, best_q_value);
	      
	      snprintf(buffer, 128,
		       "PolicyGradAscent: better policy found: %f iter %d",
		       b, iterations);
	      whiteice::logging.info(buffer);
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

  //////////////////////////////////////////////////////////////////////
  

  template class PolicyGradAscent< float >;
  template class PolicyGradAscent< double >;
  template class PolicyGradAscent< whiteice::math::blas_real<float> >;
  template class PolicyGradAscent< whiteice::math::blas_real<double> >;
};
