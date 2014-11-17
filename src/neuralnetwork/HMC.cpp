

#include "HMC.h"
#include "NNGradDescent.h"
#include <random>
#include <list>


extern "C" {
  static void* __hmcmc_thread_init(void* param);
};


namespace whiteice
{
  template <typename T>
  HMC<T>::HMC(const whiteice::nnetwork<T>& net,
	      const whiteice::dataset<T>& ds,
	      bool adaptive) : nnet(net), data(ds)
  {
    this->adaptive = adaptive;
    
    sum_N = 0;
    sum_mean.zero();
    sum_covariance.zero();
    
    running = false;
    paused = false;
    
    // nnet = net;
    
    pthread_mutex_init(&solution_lock, 0);
    pthread_mutex_init(&start_lock, 0);      
  }
  
  
  template <typename T>
  HMC<T>::~HMC()
  {
    pthread_mutex_lock( &start_lock );
    
    if(running){
      for(unsigned int i=0;i<sampling_thread.size();i++)
	pthread_cancel(sampling_thread[i]);
      
      running = false;

      while(threadIsRunning > 0)
	sleep(1); // waits for thread to stop
    }
    
    running = false;
    
    pthread_mutex_unlock( &start_lock );
    
    pthread_mutex_destroy( &solution_lock );
    pthread_mutex_destroy( &start_lock );      
  }
  
  
  template <typename T>
  T HMC<T>::U(const math::vertex<T>& q, bool useRegulizer) const
  {
    whiteice::nnetwork<T> nnet(this->nnet);
    
    nnet.importdata(q);
    
    math::vertex<T> err;
    T e = T(0.0f);

    // E = SUM 0.5*e(i)^2
    for(unsigned int i=0;i<data.size(0);i++){
      nnet.input() = data.access(0, i);
      nnet.calculate(false);
      err = data.access(1, i) - nnet.output();
      // T inv = T(1.0f/err.size());
      err = (err*err);
      e += T(0.5f)*err[0];
    }
    
    // e /= T( (float)data.size(0) ); // per N

    if(useRegulizer){
      T alpha = T(0.01);   // regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
      err = alpha*(q*q);
      e += q[0];
    }
    

    // TODO: is this really correct squared error term to use?
    
    return (e);
  }
  
  
  template <typename T>
  math::vertex<T> HMC<T>::Ugrad(const math::vertex<T>& q)
  {
    whiteice::nnetwork<T> nnet(this->nnet);
    
    T ninv = T(1.0f); // T(1.0f/data.size(0));
    math::vertex<T> sumgrad, grad, err;

    nnet.importdata(q);

    for(unsigned int i=0;i<data.size(0);i++){
      nnet.input() = data.access(0, i);
      nnet.calculate(true);
      err = data.access(1,i) - nnet.output();
      
      if(nnet.gradient(err, grad) == false){
	std::cout << "gradient failed." << std::endl;
	assert(0); // FIXME
      }
      
      if(i == 0)
	sumgrad = ninv*grad;
      else
	sumgrad += ninv*grad;
    }

    T alpha = T(0.01f);

    sumgrad += alpha*q;

    // TODO: is this really correct gradient to use
    // (we want to use: 0,5*SUM e(i)^2 + alpha*w^2
    
    
    return (sumgrad);
  }
  
  
  template <typename T>
  bool HMC<T>::startSampler(unsigned int NUM_THREADS)
  {
    if(NUM_THREADS <= 0) return false;
    
    pthread_mutex_lock( &start_lock );
    
    if(running){
      pthread_mutex_unlock( &start_lock );
      return false; // already running
    }
    
    if(data.size(0) != data.size(1)){
      pthread_mutex_unlock( &start_lock );
      return false;
    }
      
    if(data.size(0) <= 0){
      pthread_mutex_unlock( &start_lock );
      return false;
    }

    // initial starting position: 
    // we use gradient descent to go
    // to a local minima and use it as
    // a starting point
    // (local mode of the distribution)
    // this means our HMC sampler has 
    // higher quality samples immediately
    // it has started
    {
      nnet.randomize(); // initally random

      std::vector<unsigned int> arch;
      nnet.getArchitecture(arch);

      whiteice::math::NNGradDescent<T> grad;

      // optimize for the optimal position
      // until we get the first converged solution
      // (or until timeout which is hard-coded
      //  to be 2 minutes if problem do not solve)
      if(grad.startOptimize(data, arch, 1)){
	time_t start_time = time(0);
	unsigned int counter = 0;
	const unsigned int TIMEOUT = 120;
	
	T error = T(0.0f);
	unsigned int converged = 0;
	
	while(grad.getSolution(nnet, error, converged) && counter < TIMEOUT){
	  if(converged > 0)
	    break; // we got what we wanted

	  sleep(1); // give optimization thread time to run

	  counter = time(0) - start_time;
	}
	
	grad.stopComputation();
      }
      
    }
    

    threadIsRunning = 0;
    running = true;
    paused = false;

    sampling_thread.resize(NUM_THREADS);

    for(unsigned int i=0;i<NUM_THREADS;i++){
      unsigned int ok = pthread_create(&(sampling_thread[i]), 0,
				       __hmcmc_thread_init, (void*)this);
      if(ok != 0){ // failure
	running = false; // just stop all threads
	paused = false;
	
	pthread_mutex_unlock( &start_lock );

	while(threadIsRunning > 0)
	  sleep(1); // wait for threads to stop

	return false;
      }
      
      pthread_detach( sampling_thread[i] );
    }

    
    samples.clear();
    sum_mean.zero();
    sum_covariance.zero();
    sum_N = 0;
    
    pthread_mutex_unlock( &start_lock );
    
    return true;
  }
  
  
  template <typename T>
  bool HMC<T>::pauseSampler()
  {
    if(!running) return false;
    
    paused = true;
    return true;
  }
  
  
  template <typename T>
  bool HMC<T>::continueSampler()
  {
    paused = false;
    return true;
  }
  
  
  template <typename T>
  bool HMC<T>::stopSampler()
  {
    pthread_mutex_lock( &start_lock );
    
    if(running){
      for(unsigned int i=0;i<sampling_thread.size();i++){
	pthread_cancel( sampling_thread[i] );
      }
    }
    else{
      return false;
      pthread_mutex_unlock( &start_lock );
    }
    
    running = false;
    paused = false;
    
    while(threadIsRunning > 0)
      sleep(1); // waits until the threads have stopped

    pthread_mutex_unlock( &start_lock );
        
    return true;
  }
  
    
  template <typename T>
  unsigned int HMC<T>::getSamples(std::vector< math::vertex<T> >& samples) const
  {
    pthread_mutex_lock( &solution_lock );
    
    samples = this->samples;
    unsigned int N = this->samples.size();
    
    pthread_mutex_unlock( &solution_lock );
    
    return N;
  }
  

  template <typename T>
  unsigned int HMC<T>::getNumberOfSamples() const
  {
    pthread_mutex_lock( &solution_lock );
    unsigned int N = samples.size();
    pthread_mutex_unlock( &solution_lock );
    
    return N;
  }
  
  
  template <typename T>
  bool HMC<T>::getNetwork(bayesian_nnetwork<T>& bnn)
  {
    pthread_mutex_lock( &solution_lock );
    
    if(samples.size() <= 0){
      pthread_mutex_unlock( &solution_lock );
      return false;
    }

    std::vector<unsigned int> arch;
    nnet.getArchitecture(arch);
    
    if(bnn.importSamples(arch, samples) == false){
      pthread_mutex_unlock( &solution_lock );
      return false;
    }

    
    pthread_mutex_unlock( &solution_lock );
    
    return true;
  }
  
  
  template <typename T>
  math::vertex<T> HMC<T>::getMean() const
  {
    pthread_mutex_lock( &solution_lock );
    
    if(sum_N > 0){
      T inv = T(1.0f)/T(sum_N);
      math::vertex<T> m = inv*sum_mean;
      
      pthread_mutex_unlock( &solution_lock );
      
      return m;
    }
    else{
      math::vertex<T> m;
      m.zero();
      
      pthread_mutex_unlock( &solution_lock );
      
      return m;
    }
  }
  
  
  template <typename T>
  math::matrix<T> HMC<T>::getCovariance() const
  {
    pthread_mutex_lock( &solution_lock );
    
    if(sum_N > 0){
      T inv = T(1.0f)/T(sum_N);
      math::vertex<T> m = inv*sum_mean;
      math::matrix<T> C = inv*sum_covariance;
      
      C -= m.outerproduct();
      
      pthread_mutex_unlock( &solution_lock );
      
      return C;
    }
    else{
      math::matrix<T> C;
      C.zero();
      
      pthread_mutex_unlock( &solution_lock );
      
      return C;
    }
  }


  template <typename T>
  T HMC<T>::getMeanError(unsigned int latestN) const
  {
    pthread_mutex_lock( &solution_lock );

    if(!latestN) latestN = samples.size();
    if(latestN > samples.size()) latestN = samples.size();

    T sumErr = T(0.0f);

    for(unsigned int i=samples.size()-latestN;i<samples.size();i++)
      sumErr += U(samples[i], false);

    sumErr /= T((float)latestN);
    sumErr /= T((float)data.size());

    pthread_mutex_unlock( &solution_lock );

    return sumErr;
  }
  
  
  template <typename T>
  void HMC<T>::__sampler_loop()
  {
    // getting the start lock here temporarily makes sure that
    // global variables are properly initialized before threads
    // touch to them
    pthread_mutex_lock( &start_lock );
    
    threadIsRunning++;
    
    pthread_mutex_unlock( &start_lock );

    // q = location, p = momentum, H(q,p) = hamiltonian
    math::vertex<T> q, p;
    
    nnet.exportdata(q); // random position q
    
    p.resize(q.size()); // momentum is initially zero
    p.zero();
    
    T epsilon = T(0.01f);
    unsigned int L = 20;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> rng(0, 1); // N(0,1) variables


    // used to adaptively finetune step length epsilon based on accept rate
    // the aim of the adaptation is to keep accept rate near optimal 70%
    // L is fixed to rather large value 20
    T accept_rate = T(0.0f);
    unsigned int accept_rate_samples = 0;
    
    
    while(running) // keep sampling forever
    {
      // [we don't have normal distribution
      //  random number generator RNG -
      //  use and test ziggurat method]
      
      // FIXME: p = N(0,I), now we have p = Uniform(-0.5,0.5)
      for(unsigned int i=0;i<p.size();i++)
	p[i] = T(rng(gen)); // Normal distribution 
      // p[i] = T( 1.0f*(((float)rand())/((float)RAND_MAX)) - 0.5f );

      math::vertex<T> old_q = q;
      math::vertex<T> current_p = p;
      
      p -= T(0.5f) * epsilon * Ugrad(q);

      for(unsigned int i=0;i<L;i++){
	q += epsilon * p;
	if(i != L-1)
	  p -= epsilon*Ugrad(q);
      }

      p -= T(0.5f) * epsilon * Ugrad(q);

      p = -p;
      
      T current_U  = U(old_q);
      T proposed_U = U(q);
      
      T current_K  = T(0.0f);
      T proposed_K = T(0.0f);
      
      for(unsigned int i=0;i<p.size();i++){
	current_K  += T(0.5f)*current_p[i]*current_p[i];
	proposed_K += T(0.5f)*p[i]*p[i];
      }

      T r = T( (float)rand()/((float)RAND_MAX) );

      if(r < exp(current_U-proposed_U+current_K-proposed_K))
      {
	// accept (q)
	// printf("ACCEPT\n");
	
	pthread_mutex_lock( &solution_lock );
	
	if(sum_N > 0){
	  sum_mean += q;
	  sum_covariance += q.outerproduct();
	  sum_N++;
	}
	else{
	  sum_mean = q;
	  sum_covariance = q.outerproduct();
	  sum_N++;
	}
	
	samples.push_back(q);
        
	pthread_mutex_unlock( &solution_lock );

	if(adaptive){
	  accept_rate++;
	  accept_rate_samples++;
	}
		
      }
      else{
	// reject (keep old_q)
	// printf("REJECT\n");
	
	q = old_q;
	
	pthread_mutex_lock( &solution_lock );
	
	if(sum_N > 0){
	  sum_mean += q;
	  sum_covariance += q.outerproduct();
	  sum_N++;
	}
	else{
	  sum_mean = q;
	  sum_covariance = q.outerproduct();
	  sum_N++;
	}
	
	samples.push_back(q);
	
	pthread_mutex_unlock( &solution_lock );

	if(adaptive){
	  // accept_rate;
	  accept_rate_samples++;
	}
      }


      if(adaptive){
	// use accept rate to adapt epsilon
	// adapt sampling rate every N iteration (sample)
	if(accept_rate_samples >= 50)
	{
	  accept_rate /= accept_rate_samples;

	  // std::cout << "ACCEPT RATE: " << accept_rate << std::endl;
	  
	  if(accept_rate <= T(0.65f)){
	    epsilon = T(0.8)*epsilon;
	    // std::cout << "NEW SMALLER EPSILON: " << epsilon << std::endl;
	    
	  }
	  else if(accept_rate >= T(0.85f)){
	    epsilon = T(1.1)*epsilon;
	    // std::cout << "NEW LARGER  EPSILON: " << epsilon << std::endl;
	  }

	  accept_rate = T(0.0f);
	  accept_rate_samples = 0;
	}
      }
	    
      
      // printf("SAMPLES: %d\n", samples.size());
      
      while(paused && running) // pause
	sleep(1);
      
    }

    threadIsRunning--;
  }
  
  
};


namespace whiteice
{  
  template class HMC< float >;
  template class HMC< double >;
  template class HMC< math::atlas_real<float> >;
  template class HMC< math::atlas_real<double> >;    
};


extern "C" {
  void* __hmcmc_thread_init(void *ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(ptr)
      ((whiteice::HMC< whiteice::math::atlas_real<float> >*)ptr)->__sampler_loop();
    
    pthread_exit(0);

    return 0;
  }
};
