/*
 * hamiltonian MCMC sampling for neural networks
 */

#include "UHMC.h"
#include "NNGradDescent.h"

#include <random>
#include <list>
#include <chrono>


namespace whiteice
{
	template <typename T>
	UHMC<T>::UHMC(const whiteice::nnetwork<T>& net,
			const whiteice::dataset<T>& ds,
			bool adaptive, T alpha, bool store) : nnet(net), data(ds)
	{
		this->adaptive = adaptive;
		this->alpha = alpha;
		this->temperature = T(1.0);
		this->store = store;

		sum_N = 0;
		sum_mean.zero();
		// sum_covariance.zero();

		running = false;
		paused = false;
	}


	template <typename T>
	UHMC<T>::~UHMC()
	{
		std::lock_guard<std::mutex> lock(start_lock);

		if(running){
			running = false;

			for(auto t : sampling_thread){
				t->join();
				delete t;
			}

			sampling_thread.clear();
		}
	}


	// set "temperature" for probability distribution [default T = 1 => no temperature]
	template <typename T>
	bool UHMC<T>::setTemperature(const T t){
		if(t <= T(0.0)) return false;
		else temperature = t;
		return true;
	}

	// get "temperature" of probability distribution
	template <typename T>
	T UHMC<T>::getTemperature(){
		return temperature;
	}
  

	template <typename T>
	T UHMC<T>::U(const math::vertex<T>& q, bool useRegulizer) const
	{
		T E = T(0.0f);

		// E = SUM 0.5*e(i)^2
#pragma omp parallel shared(E)
		{
			whiteice::nnetwork<T> nnet(this->nnet);
			nnet.importdata(q);

			math::vertex<T> err;
			T e = T(0.0f);

#pragma omp for nowait schedule(auto)
			for(unsigned int i=0;i<data.size(0);i++){
				nnet.input() = data.access(0, i);
				nnet.calculate(false);
				err = data.access(1, i) - nnet.output();
				// T inv = T(1.0f/err.size());
				err = (err*err);
				e = e  + T(0.5f)*err[0];
			}

#pragma omp critical
			{
				E = E + e;
			}
		}

		E /= sigma2;

		E /= temperature;
		
		// TODO: is this really correct squared error term to use?

		return (E);
	}
  
  
	template <typename T>
	math::vertex<T> UHMC<T>::Ugrad(const math::vertex<T>& q) const
	{
		math::vertex<T> sum;
		sum.resize(q.size());
		sum.zero();

		// positive gradient
#pragma omp parallel shared(sum)
		{
			// const T ninv = T(1.0f); // T(1.0f/data.size(0));
			math::vertex<T> sumgrad, grad, err;
			sumgrad.resize(q.size());
			sumgrad.zero();

			whiteice::nnetwork<T> nnet(this->nnet);
			nnet.importdata(q);

#pragma omp for nowait schedule(auto)
			for(unsigned int i=0;i<data.size(0);i++){
				nnet.input() = data.access(0, i);
				nnet.calculate(true);
				err = nnet.output() - data.access(1,i);

				if(nnet.mse_gradient(err, grad) == false){
					std::cout << "gradient failed." << std::endl;
					assert(0); // FIXME
				}

				sumgrad += grad;
			}

#pragma omp critical
			{
				sum += sumgrad;
			}
		}
		
		sum /= sigma2;

		sum /= temperature; // scales gradient with temperature

		
		sum += T(0.5)*alpha*q;
		
		sum.normalize();

		return (sum);
	}
  
  
  
        // calculates z-ratio between data likelihood distributions
        template <typename T>
	T UHMC<T>::zratio(const math::vertex<T>& q1, const math::vertex<T>& q2) const
	{
	  whiteice::nnetwork<T> nnet1(this->nnet);
	  whiteice::nnetwork<T> nnet2(this->nnet);
	  nnet1.importdata(q1);
	  nnet2.importdata(q2);
	  
	  std::vector<T> zratio;
	  const unsigned int BLOCKSIZE = 100;
	  const unsigned int MAXITERS  = 100;
	  unsigned int iters = 0;
	  
	  while(iters < MAXITERS) // continues until convergence
	  { 
	    unsigned int index0 = zratio.size();
	    
	    zratio.resize(index0 + BLOCKSIZE); // increases zratio size
	    
#pragma omp parallel for shared(zratio) schedule(auto)
	    for(unsigned int index=0;index<BLOCKSIZE;index++){
	      // generates negative particle (x side)
	      
	      const unsigned int data_index = rng.rand() % data.size(0);
	      auto x = data.access(0, data_index);
	      
	      // generates negative particle (y side) from input data [x => y]
	      math::vertex<T> y(nnet2.output_size());
	      nnet2.calculate(x, y);
	      
	      // samples p(y|f(x)) = N(f(x),C)
	      math::vertex<T> n(this->nnet.output_size());
	      rng.normal(n);
	      y = y + n*math::sqrt(sigma2);
	      
	      
	      // now we have negative sample (x,y) from q2 distribution and we calculate
	      // z-ratio of unscaled data probability distribution (x,y)
	      
	      math::vertex<T> y1(nnet1.output_size());
	      math::vertex<T> y2(nnet2.output_size());
	      
	      nnet1.calculate(x, y1);
	      auto error1 = T(0.5)*((y - y1)*(y - y1)/sigma2)[0];
	      
	      nnet2.calculate(x, y2);
	      auto error2 = T(0.5)*((y - y2)*(y - y2)/sigma2)[0];

	      // auto ratio = math::exp(error2 - error1);

	      T ratio = T(0.0f);
	      T delta = error2 - error1;
	      
	      if(delta > T(+30.0f)){ // to work around SIGFPE floating point exceptions
		ratio = math::exp(+30.0f);
	      }
	      else if(delta < T(-30.0f)){ // to work around SIGFPE floating point exceptions
		ratio = math::exp(-30.0f);
	      }
	      else{
		ratio = math::exp(delta);
	      }
	      
	      
	      
	      zratio[index0+index] = ratio;
	    }
	    
	    // estimates for convergence: calculates st.dev./mean 
	    T mr = T(0.0);
	    T vr = T(0.0);
	    
	    for(auto& s : zratio){
	      mr += s;
	      vr += s*s;
	    }
	    
	    mr /= T(zratio.size());
	    vr /= T(zratio.size());
	    vr -= mr*mr;
	    // changes division to 1/N-1 (sample variance)
	    vr *= T((double)zratio.size()/((double)zratio.size() - 1.0));

	    vr /= T(zratio.size()); // calculates mean estimator's variance..
	    
	    vr = math::sqrt(vr);
	    
	    T tst = vr/mr;
	    
	    // std::cout << "test stdev/mean = " << vr << " : " << mr << " : " << tst << std::endl;
	    // fflush(stdout);
	    
	    if(!(mr - T(2.0)*vr < T(1.0) && mr + T(2.0)*vr > T(1.0)) && tst <= T(0.8)){
	      // convergence [sample st.dev. is less than 1% of the mean value (1% error)]
	      
	      // printf("zratio number of iterations: %d\n", (int)zratio.size());
	      
	      mr = math::pow(mr, T(data.size(0)));
	      
	      return mr;
	    }
	    
	    iters++;
	  }
	  
	  return T(1.0);
	}
  
  
        template <typename T>
	bool UHMC<T>::sample_covariance_matrix(const math::vertex<T>& q)
	{
	  const unsigned int DIM = nnet.output_size();
	  
	  math::matrix<T> S(DIM, DIM);
	  math::vertex<T> m(DIM);
	  
	  S.zero();
	  m.zero();
	  
	  for(unsigned int i=0;i<data.size(0);i++){
	    const auto& x = data.access(0, i);
	    math::vertex<T> fx;
	    
	    nnet.calculate(x, fx);
	    auto z =  data.access(1, i) - fx;
	      
	    S += z.outerproduct();
	    m += z;
	  }
	  
	  S /= T(data.size(0));
	  m /= T(data.size(0));
	  
	  S -= m.outerproduct();
	  
	  sigma2 = S(0,0);
	  
	  for(unsigned int i=0;i<DIM;i++){
	    if(sigma2 > S(i,i))
	      sigma2 = S(i,i);
	  }
	  
	  if(sigma2 < T(0.0001))
	    sigma2 = T(0.0001);
	  
	  
	  return true;
	  
	  
#if 0	  


	  // we have PROD(i)[N(y_i-f(x_i|w)|m, S, w)] * N(m|0,S/k)InvWishart(S|L) , k = inf
	  // and the posterior is InvWishart(S|L,w)
	  
	  whiteice::nnetwork<T> nnet(this->nnet);
	  nnet.importdata(q);
	  
	  const unsigned int DIM = nnet.output_size();
	  
	  math::matrix<T> PRIOR(DIM,DIM);
	  PRIOR.zero();
	  
	  while(1){
	    math::matrix<T> Ln(DIM, DIM);
	    Ln = PRIOR; 
	    
	    for(unsigned int i=0;i<data.size(0);i++){
	      const auto& x = data.access(0, i);
	      math::vertex<T> fx;
	      
	      nnet.calculate(x, fx);
	      auto z =  data.access(1, i) - fx;
	      
	      Ln += z.outerproduct();
	    }
	    
	    unsigned int vn = 0;
	    
	    if(data.size(0) <= nnet.output_size()){
	      vn = nnet.output_size();
	      math::matrix<T> L0(nnet.output_size(), nnet.output_size());
	      L0.identity();
	      
	      Ln += L0;
	    }
	    else{
	      vn = data.size(0) - 1;
	    }
	    
	    if(Ln.inv() == false){
	      // just add I to prior to regularize more..
	      for(unsigned int i=0;i<DIM;i++)
		PRIOR(i,i) += T(1.0);
	      continue;
	    }
	    
	    // we need to sample from N(0, Ln), z = X * D^0.5 * x, x ~ N(0, I). X*D*X^t = Ln
	    
	    math::matrix<T> X(Ln);
	    
	    if(symmetric_eig(Ln, X) == false){
	      // just add I to prior to regularize more..
	      for(unsigned int i=0;i<DIM;i++)
		PRIOR(i,i) += T(1.0);
	      continue;
	    }
	    
	    
	    auto& D = Ln;
	    
	    for(unsigned int i=0;i<D.ysize();i++){
	      D(i,i) = math::sqrt(D(i,i));
	    }
	    
	    auto XD = X*D; // XD adds covariance structure to uncorrelated data
	    
	    math::matrix<T> A(DIM, DIM);
	    
	    for(unsigned int v=0;v<vn;v++){
	      math::vertex<T> x(DIM);
	      rng.normal(x);
	      
	      auto z = XD*x;
	      
	      A += z.outerproduct();
	    }
	    
	    // A matrix has is sampled from precision matrix C^-1 distribution
	    
	    A.inv();
	
	    
	    sigma2 = A(0,0);
	    
	    for(unsigned int i=0;i<DIM;i++){
	      if(sigma2 < A(i,i))
		sigma2 = A(i,i);
	    }
	    
	    if(sigma2 < T(0.0001))
	      sigma2 = T(0.0001);
	    
      
	    return true;
	  }
#endif
	}
  
  
	template <typename T>
	bool UHMC<T>::startSampler()
	{
		const unsigned int NUM_THREADS = 1; // only one thread is supported

		std::lock_guard<std::mutex> lock(start_lock);

		if(running)
			return false; // already running
    
		if(data.size(0) != data.size(1))
			return false;

		if(data.size(0) <= 0)
			return false;

		// nnet.randomize(); // initally random
		nnet.exportdata(q); // initial position q
    
		running = true;
		paused = false;

		sum_N = 0;
		sum_mean.zero();
		
		sigma2 = T(1.0);

		sampling_thread.clear();

		for(unsigned int i=0;i<NUM_THREADS;i++){

			try{
			        std::thread* t = new std::thread(&UHMC<T>::sampler_loop, this);
				// t->detach();
				sampling_thread.push_back(t);
			}
			catch(std::system_error e){
				running = false;
				paused = false;

				for(auto t : sampling_thread){
					t->join();
					delete t;
				}

				sampling_thread.clear();

				return false;
			}
		}


		samples.clear();
		sum_mean.zero();
		// sum_covariance.zero();
		sum_N = 0;

		return true;
	}
  
  
	template <typename T>
	bool UHMC<T>::pauseSampler()
	{
		if(!running) return false;

		paused = true;
		return true;
	}


	template <typename T>
	bool UHMC<T>::continueSampler()
	{
		paused = false;
		return true;
	}
  
  
	template <typename T>
	bool UHMC<T>::stopSampler()
	{
		std::lock_guard<std::mutex> lock(start_lock);

		if(!running)
			return false;

		running = false;
		paused = false;

		for(auto t : sampling_thread){
			t->join();
			delete t;
		}

		sampling_thread.clear();
		return true;
	}


	template <typename T>
	bool UHMC<T>::getCurrentSample(math::vertex<T>& s) const
	{
		std::lock_guard<std::mutex> lock(updating_sample);
		s = q;
		return true;
	}


	template <typename T>
	bool UHMC<T>::setCurrentSample(const math::vertex<T>& s){
		std::lock_guard<std::mutex> lock(updating_sample);
		q = s;
		return true;
	}
  
    
	template <typename T>
	unsigned int UHMC<T>::getSamples(std::vector< math::vertex<T> >& samples) const
	{
		std::lock_guard<std::mutex> lock(solution_lock);

		samples = this->samples;
		unsigned int N = this->samples.size();

		return N;
	}


	template <typename T>
	unsigned int UHMC<T>::getNumberOfSamples() const
	{
		std::lock_guard<std::mutex> lock(solution_lock);
		unsigned int N = samples.size();

		return N;
	}
  
  
	template <typename T>
	bool UHMC<T>::getNetwork(bayesian_nnetwork<T>& bnn)
	{
		std::lock_guard<std::mutex> lock(solution_lock);

		if(samples.size() <= 0)
			return false;

		if(bnn.importSamples(nnet, samples) == false)
			return false;

		return true;
	}


	template <typename T>
	math::vertex<T> UHMC<T>::getMean() const
	{
		std::lock_guard<std::mutex> lock(solution_lock);

		if(sum_N > 0){
			T inv = T(1.0f)/T(sum_N);
			math::vertex<T> m = inv*sum_mean;

			return m;
		}
		else{
			math::vertex<T> m;
			m.zero();

			return m;
		}
	}
  
#if 0
  template <typename T>
  math::matrix<T> UHMC<T>::getCovariance() const
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
#endif


    template <typename T>
    T UHMC<T>::getMeanError(unsigned int latestN) const
	{
    	std::lock_guard<std::mutex> lock(solution_lock);
    
    	if(!latestN) latestN = samples.size();
    	if(latestN > samples.size()) latestN = samples.size();

    	T sumErr = T(0.0f);

    	for(unsigned int i=samples.size()-latestN;i<samples.size();i++)
	{
	  T E = T(0.0f);
	  
	  // E = SUM 0.5*e(i)^2
#pragma omp parallel shared(E)
	  {
	    whiteice::nnetwork<T> nnet(this->nnet);
	    nnet.importdata(samples[i]);
	    
	    math::vertex<T> err;
	    T e = T(0.0f);
	    
#pragma omp for nowait schedule(auto)
	    for(unsigned int i=0;i<data.size(0);i++){
	      nnet.input() = data.access(0, i);
	      nnet.calculate(false);
	      err = data.access(1, i) - nnet.output();
	      // T inv = T(1.0f/err.size());
	      err = (err*err);
	      e = e  + T(0.5f)*err[0];
	    }
	    
#pragma omp critical
	    {
	      E = E + e;
	    }
	  }
	  
	  sumErr += E;
	}

	if(latestN > 0){
	  sumErr /= T((float)latestN);
	  sumErr /= T((float)data.size(0));
	}

    	return sumErr;
	}


    template <typename T>
    void UHMC<T>::sampler_loop()
	{
    	// q = location, p = momentum, H(q,p) = hamiltonian
    	math::vertex<T> p; // q is global and defined in UHMC class

    	{
    		std::lock_guard<std::mutex> lock(updating_sample);
    		nnet.exportdata(q); // initial position q
    		                    // (from the input nnetwork weights)
    	}

    	p.resize(q.size()); // momentum is initially zero
    	p.zero();

    	T epsilon = T(0.01f);
    	unsigned int L = 10;
	
	// scales epsilon heuristically according to number of datapoints in sum
	{
	  epsilon /= T(data.size(0)); // gradient sum is now "divided by number of datapoints"
	}

	// std::random_device rd;
	// std::mt19937 gen(rd());
    	// std::normal_distribution<> rngn(0, 1); // N(0,1) variables
    	// auto normalrnd = std::bind(rngn, std::ref(gen));


    	// used to adaptively finetune step length epsilon based on accept rate
    	// the aim of the adaptation is to keep accept rate near optimal 70%
    	// L is fixed to rather large value 20
    	T accept_rate = T(0.0f);
    	unsigned int accept_rate_samples = 0;
	
	// heuristics: we don't store any samples until number of accepts
	// has been 5, this is used to wait for epsilon parameter to adjust
	// correctly so that the probability of accept per iteration is reasonable
	// (we don't store rejects during initial epsiln parameter learning)
	unsigned int number_of_accepts = 0;
	const unsigned int EPSILON_LEARNING_ACCEPT_LIMIT = 5;

	const T MAX_EPSILON = T(1.0f);

	
    	while(running) // keep sampling forever or until stopped
    	{
    		updating_sample.lock();

    		// p = N(0,I)
		// for(unsigned int i=0;i<p.size();i++)
		// p[i] = T(normalrnd()); // Normal distribution
		
		sample_covariance_matrix(q);
		
		rng.normal(p);

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

		
		T r = rng.uniform();
		// T p_accept = exp(current_U-proposed_U+current_K-proposed_K);

		T p_accept = T(0.0f);
		T expvalue = current_U-proposed_U+current_K-proposed_K;
		if(expvalue < T(-10.0f)){ // to work around SIGFPE floating point exceptions
		  p_accept = exp(T(-10.0f));
		}
		else if(expvalue > T(+10.0f)){ // to work around SIGFPE floating point exceptions
		  p_accept = exp(T(+10.0f));
		}
		else{
		  p_accept = exp(expvalue);
		}
		

    		if(r < p_accept && !whiteice::math::isnan(p_accept))
    		{
    			// accept (q)
		        // printf("ACCEPT\n");

		        number_of_accepts++;
			
			if(number_of_accepts > EPSILON_LEARNING_ACCEPT_LIMIT){
			  solution_lock.lock();
			  
			  if(sum_N > 0){
			    sum_mean += q;
			    //sum_covariance += q.outerproduct();
			    sum_N++;
			  }
			  else{
			    sum_mean = q;
			    // sum_covariance = q.outerproduct();
			    sum_N++;
			  }
	
			  if(store)
			    samples.push_back(q);
			  
			  solution_lock.unlock();
			}

    			if(adaptive){
    				accept_rate++;
    				accept_rate_samples++;
    			}

    		}
    		else{
    			// reject (keep old_q)
		        // printf("REJECT\n");
		        q = old_q;
			
			if(number_of_accepts > EPSILON_LEARNING_ACCEPT_LIMIT){
			  solution_lock.lock();

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
			  
			  if(store)
			    samples.push_back(q);
			  
			  solution_lock.unlock();
			}
			

    			if(adaptive){
			  // accept_rate;
			  accept_rate_samples++;
    			}
    		}

    		updating_sample.unlock();


    		if(adaptive){
    			// use accept rate to adapt epsilon
    			// adapt sampling rate every N iteration (sample)
    			if(accept_rate_samples >= 20)
    			{
    				accept_rate /= accept_rate_samples;

				// std::cout << "ACCEPT RATE: " << accept_rate << std::endl;
				// changed from 65-85% to 50%

    				if(accept_rate < T(0.50f)){
    					epsilon = T(0.8)*epsilon;
					// std::cout << "NEW SMALLER EPSILON: " << epsilon << std::endl;
    				}
    				else if(accept_rate > T(0.50f)){
				        // important, sampler can diverge so we FORCE epsilon to be small (<MAX_EPSILON)
				        auto new_epsilon = T(1.0/0.8)*epsilon;
					if(new_epsilon < MAX_EPSILON)
					  epsilon = new_epsilon;
					
					// std::cout << "NEW LARGER  EPSILON: " << epsilon << std::endl;
    				}

    				accept_rate = T(0.0f);
    				accept_rate_samples = 0;
    			}
    		}


    		// printf("SAMPLES: %d\n", samples.size());

    		while(paused && running){ // pause
    			std::this_thread::sleep_for(std::chrono::milliseconds(500)); // sleep for 500ms
    		}
    	}

	}
  
  
};


namespace whiteice
{  
  template class UHMC< math::blas_real<float> >;
  template class UHMC< math::blas_real<double> >;    
};

