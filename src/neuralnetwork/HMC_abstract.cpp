

#include "HMC_abstract.h"
#include <random>
#include <list>


extern "C" {
  static void* __hmc_abstract_thread_init(void* param);
};


namespace whiteice
{
  template <typename T>
  HMC_abstract<T>::HMC_abstract(bool adaptive)
  {
    this->adaptive = adaptive;
    
    sum_N = 0;
    sum_mean.zero();
    // sum_covariance.zero();
    
    running = false;
    paused = false;
    
    // nnet = net;
    
    pthread_mutex_init(&solution_lock, 0);
    pthread_mutex_init(&start_lock, 0);      
  }
  
  
  template <typename T>
  HMC_abstract<T>::~HMC_abstract()
  {
    pthread_mutex_lock( &start_lock );
    
    if(running){
      pthread_cancel(sampling_thread);

      running = false;

      while(threadIsRunning)
	sleep(1); // waits for thread to stop
    }
    
    running = false;
    
    pthread_mutex_unlock( &start_lock );
    
    pthread_mutex_destroy( &solution_lock );
    pthread_mutex_destroy( &start_lock );      
  }
  
  
  
  template <typename T>
  bool HMC_abstract<T>::startSampler()
  {
    pthread_mutex_lock( &start_lock );
    
    if(running){
      pthread_mutex_unlock( &start_lock );
      return false; // already running
    }
    
    running = true;
    paused = false;
    
    if(pthread_create(&sampling_thread, 0,
		      __hmc_abstract_thread_init, (void*)this) == 0){
      pthread_mutex_unlock( &start_lock );
      return true;
    }
    
    running = false;
    paused = false;
    
    pthread_mutex_unlock( &start_lock );
    
    return false;
  }
  
  
  template <typename T>
  bool HMC_abstract<T>::pauseSampler()
  {
    if(!running) return false;
    
    paused = true;
    return true;
  }
  
  
  template <typename T>
  bool HMC_abstract<T>::continueSampler()
  {
    if(!running) return false;
    
    paused = false;
    return true;
  }
  
  
  template <typename T>
  bool HMC_abstract<T>::stopSampler()
  {
    pthread_mutex_lock( &start_lock );

    if(running){
      pthread_cancel(sampling_thread);
    }
    else{
      return false;
      pthread_mutex_unlock( &start_lock );
    }
    
    running = false;
    paused = false;
    
    while(threadIsRunning == true)
      sleep(1); // waits until the thread is stopped

    pthread_mutex_unlock( &start_lock );
    

    
    return true;
  }
  
    
  template <typename T>
  unsigned int HMC_abstract<T>::getSamples(std::vector< math::vertex<T> >& samples) const
  {
    pthread_mutex_lock( &solution_lock );
    
    samples = this->samples;
    unsigned int N = this->samples.size();
    
    pthread_mutex_unlock( &solution_lock );
    
    return N;
  }
  

  template <typename T>
  unsigned int HMC_abstract<T>::getNumberOfSamples() const
  {
    pthread_mutex_lock( &solution_lock );
    unsigned int N = samples.size();
    pthread_mutex_unlock( &solution_lock );
    
    return N;
  }
  
  
  template <typename T>
  math::vertex<T> HMC_abstract<T>::getMean() const
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
  
  /*
  template <typename T>
  math::matrix<T> HMC_abstract<T>::getCovariance() const
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
  */


  template <typename T>
  T HMC_abstract<T>::getMeanError(unsigned int latestN) const
  {
    pthread_mutex_lock( &solution_lock );

    if(!latestN) latestN = samples.size();
    if(latestN > samples.size()) latestN = samples.size();

    T sumErr = T(0.0f);

    for(unsigned int i=samples.size()-latestN;i<samples.size();i++)
      sumErr += U(samples[i]);

    sumErr /= T((float)latestN);

    pthread_mutex_unlock( &solution_lock );

    return sumErr;
  }
  
  
  template <typename T>
  void HMC_abstract<T>::__sampler_loop()
  {
    threadIsRunning = true;

    samples.clear();
    sum_mean.zero();
    // sum_covariance.zero();
    sum_N = 0;
    
    // q = location, p = momentum, H(q,p) = hamiltonian
    math::vertex<T> q, p;
    
    starting_position(q); // random starting position q
    
    p.resize(q.size()); // momentum is initially zero
    p.zero();
    
    // epsilon = epsilon0/sqrt(D) in order to keep distance ||x(n+1) - x(n)|| = epsilon0 for all dimensions dim(x) = D
    T epsilon = T(0.01f)/math::sqrt(q.size());
    unsigned int L = 20;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> rng(0, 1); // N(0,1) variables
    auto normalrnd = std::bind(rng, std::ref(gen));

    // used to adaptively finetune step length epsilon based on accept rate
    // the aim of the adaptation is to keep accept rate near optimal 70%
    // L is fixed to rather large value 20
    T accept_rate = T(0.0f);
    unsigned int accept_rate_samples = 0;

    
    while(running) // keep sampling forever
    {
      for(unsigned int i=0;i<p.size();i++)
	p[i] = T(normalrnd()); // Normal distribution 

      math::vertex<T> old_q = q;
      math::vertex<T> current_p = p;
#if 1
      p -= T(0.5f) * epsilon * Ugrad(q);

      for(unsigned int i=0;i<L;i++){
	q += epsilon * p;
	if(i != L-1) p -= epsilon*Ugrad(q);
      }

      p -= T(0.5f) * epsilon * Ugrad(q);

      p = -p;
#else
      // just gradient descent...
      auto g = Ugrad(q);
      q = epsilon * g;
#endif
      T current_U  = U(old_q);
      T proposed_U = U(q);
      
      //std::cout << "current_U  = " << U(old_q) << std::endl;
      //std::cout << "proposed_U = " << U(q) << std::endl;
      //std::cout << "epslon     = " << epsilon << std::endl;

      T current_K  = T(0.0f);
      T proposed_K = T(0.0f);
      
      for(unsigned int i=0;i<p.size();i++){
	current_K  += T(0.5f)*current_p[i]*current_p[i];
	proposed_K += T(0.5f)*p[i]*p[i];
      }
      
      T r = T( (float)rand()/((float)RAND_MAX) );

      if(r <= exp(current_U-proposed_U+current_K-proposed_K))
      {
	// accept (q)
	// printf("************************************************************ ACCEPT\n");
	
	pthread_mutex_lock( &solution_lock );
	
	if(sum_N > 0){
	  sum_mean += q;
	  // sum_covariance += q.outerproduct();
	  sum_N++;
	}
	else{
	  sum_mean = q;
	  // sum_covariance = q.outerproduct();
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
	  // sum_covariance += q.outerproduct();
	  sum_N++;
	}
	else{
	  sum_mean = q;
	  // sum_covariance = q.outerproduct();
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

    threadIsRunning = false;
  }
  
  
};


namespace whiteice
{  
  template class HMC_abstract< float >;
  template class HMC_abstract< double >;
  template class HMC_abstract< math::blas_real<float> >;
  template class HMC_abstract< math::blas_real<double> >;    
};


extern "C" {
  void* __hmc_abstract_thread_init(void *ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(ptr)
      ((whiteice::HMC_abstract< whiteice::math::blas_real<float> >*)ptr)->__sampler_loop();
    
    pthread_exit(0);

    return 0;
  }
};
