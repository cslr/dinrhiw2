/*
 * hamiltonian MCMC sampling for neural networks
 */

#include "HMC.h"
#include "NNGradDescent.h"

#include <random>
#include <list>
#include <chrono>


namespace whiteice
{
	template <typename T>
	HMC<T>::HMC(const whiteice::nnetwork<T>& net,
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
	HMC<T>::~HMC()
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
	bool HMC<T>::setTemperature(const T t){
		if(t <= T(0.0)) return false;
		else temperature = t;
		return true;
	}

	// get "temperature" of probability distribution
	template <typename T>
	T HMC<T>::getTemperature(){
		return temperature;
	}
  

	template <typename T>
	T HMC<T>::U(const math::vertex<T>& q, bool useRegulizer) const
	{
		T E = T(0.0f);

		// E = SUM 0.5*e(i)^2
#pragma omp parallel shared(E)
		{
			whiteice::nnetwork<T> nnet(this->nnet);
			nnet.importdata(q);

			math::vertex<T> err;
			T e = T(0.0f);

#pragma omp for nowait
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

		// e /= T( (float)data.size(0) ); // per N

		E /= temperature;

		if(useRegulizer){
			math::vertex<T> err;
			// regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
			err = alpha*T(0.5f)*(q*q);
			E += err[0];
			// e += q[0];
		}

		// TODO: is this really correct squared error term to use?

		return (E);
	}
  
  
	template <typename T>
	math::vertex<T> HMC<T>::Ugrad(const math::vertex<T>& q) const
	{
		math::vertex<T> sum;
		sum.resize(q.size());
		sum.zero();

#pragma omp parallel shared(sum)
		{
			// const T ninv = T(1.0f); // T(1.0f/data.size(0));
			math::vertex<T> sumgrad, grad, err;
			sumgrad.resize(q.size());
			sumgrad.zero();

			whiteice::nnetwork<T> nnet(this->nnet);
			nnet.importdata(q);

#pragma omp for nowait
			for(unsigned int i=0;i<data.size(0);i++){
				nnet.input() = data.access(0, i);
				nnet.calculate(true);
				err = data.access(1,i) - nnet.output();

				if(nnet.gradient(err, grad) == false){
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

		sum /= temperature; // scales gradient with temperature


		sum += alpha*q;

		// TODO: is this really correct gradient to use
		// (we want to use: 0,5*SUM e(i)^2 + alpha*w^2

		return (sum);
	}
  
  
	template <typename T>
	bool HMC<T>::startSampler()
	{
		const unsigned int NUM_THREADS = 1; // only one thread is supported

		std::lock_guard<std::mutex> lock(start_lock);

		if(running)
			return false; // already running
    
		if(data.size(0) != data.size(1))
			return false;

		if(data.size(0) <= 0)
			return false;

		nnet.randomize(); // initally random
		nnet.exportdata(q); // initial position q
    
		running = true;
		paused = false;

		sum_N = 0;
		sum_mean.zero();

		sampling_thread.clear();

		for(unsigned int i=0;i<NUM_THREADS;i++){

			try{
				std::thread* t = new std::thread(sampler_loop, this);
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
	bool HMC<T>::getCurrentSample(math::vertex<T>& s) const
	{
		std::lock_guard<std::mutex> lock(updating_sample);
		s = q;
		return true;
	}


	template <typename T>
	bool HMC<T>::setCurrentSample(const math::vertex<T>& s){
		std::lock_guard<std::mutex> lock(updating_sample);
		q = s;
		return true;
	}
  
    
	template <typename T>
	unsigned int HMC<T>::getSamples(std::vector< math::vertex<T> >& samples) const
	{
		std::lock_guard<std::mutex> lock(solution_lock);

		samples = this->samples;
		unsigned int N = this->samples.size();

		return N;
	}


	template <typename T>
	unsigned int HMC<T>::getNumberOfSamples() const
	{
		std::lock_guard<std::mutex> lock(solution_lock);
		unsigned int N = samples.size();

		return N;
	}
  
  
	template <typename T>
	bool HMC<T>::getNetwork(bayesian_nnetwork<T>& bnn)
	{
		std::lock_guard<std::mutex> lock(solution_lock);

		if(samples.size() <= 0)
			return false;

		std::vector<unsigned int> arch;
		nnet.getArchitecture(arch);

		if(bnn.importSamples(arch, samples) == false)
			return false;

		return true;
	}


	template <typename T>
	math::vertex<T> HMC<T>::getMean() const
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
#endif


    template <typename T>
    T HMC<T>::getMeanError(unsigned int latestN) const
	{
    	std::lock_guard<std::mutex> lock(solution_lock);
    
    	if(!latestN) latestN = samples.size();
    	if(latestN > samples.size()) latestN = samples.size();

    	T sumErr = T(0.0f);

    	for(unsigned int i=samples.size()-latestN;i<samples.size();i++)
    		sumErr += U(samples[i], false);

    	sumErr /= T((float)latestN);
    	sumErr /= T((float)data.size());

    	return sumErr;
	}


    template <typename T>
    void HMC<T>::sampler_loop()
	{
    	// q = location, p = momentum, H(q,p) = hamiltonian
    	math::vertex<T> p; // q is global and defined in HMC class

    	{
    		std::lock_guard<std::mutex> lock(updating_sample);
    		nnet.exportdata(q); // initial position q
    		                    // (from the input nnetwork weights)
    	}

    	p.resize(q.size()); // momentum is initially zero
    	p.zero();

    	T epsilon = T(0.01f);
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


    	while(running) // keep sampling forever or until stopped
    	{
    		updating_sample.lock();

    		// p = N(0,I)
    		for(unsigned int i=0;i<p.size();i++)
    			p[i] = T(normalrnd()); // Normal distribution

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

    			if(adaptive){
    				accept_rate++;
    				accept_rate_samples++;
    			}

    		}
    		else{
    			// reject (keep old_q)
    			// printf("REJECT\n");

    			q = old_q;

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

    		while(paused && running){ // pause
    			std::this_thread::sleep_for(std::chrono::milliseconds(500)); // sleep for 500ms
    		}
    	}

	}
  
  
};


namespace whiteice
{  
  template class HMC< float >;
  template class HMC< double >;
  template class HMC< math::blas_real<float> >;
  template class HMC< math::blas_real<double> >;    
};

