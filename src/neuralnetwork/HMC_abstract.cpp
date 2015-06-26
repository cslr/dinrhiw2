

#include "HMC_abstract.h"
#include <random>
#include <list>



namespace whiteice
{
	template <typename T>
	HMC_abstract<T>::HMC_abstract(bool storeSamples, bool adaptive)
	{
		this->adaptive = adaptive;
		this->storeSamples = storeSamples;

		sum_N = 0;
		sum_mean.zero();
		// sum_covariance.zero();

		running = false;
		paused = false;


	}
  
  
	template <typename T>
	HMC_abstract<T>::~HMC_abstract()
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

		running = false;
	}
  
  
  
	template <typename T>
	bool HMC_abstract<T>::startSampler()
	{
		std::lock_guard<std::mutex> lock(start_lock);

		if(running)
			return false; // already running

		running = true;
		paused = false;

		sampling_thread.clear();

		try{
			starting_position(q); // initializes starting position before starting the thread here just to be sure..
			std::thread* t = new std::thread(sampler_loop, this);
			sampling_thread.push_back(t);

			return true;
		}
		catch(std::exception& e){
			std::cout << "ERROR: unexpected exception: " << e.what() << std::endl;
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
	unsigned int HMC_abstract<T>::getSamples(std::vector< math::vertex<T> >& samples) const
	{
		std::lock_guard<std::mutex> lock(solution_lock);

		samples = this->samples;
		unsigned int N = this->samples.size();

		return N;
	}
  

	template <typename T>
	unsigned int HMC_abstract<T>::getNumberOfSamples() const
	{
		return samples.size(); // calling size() should be thread-safe... ???
	}
  
  
	template <typename T>
	bool HMC_abstract<T>::getCurrentSample(math::vertex<T>& s)
	{
		std::lock_guard<std::mutex> lock(updating_sample);
		s = q;
		return true;

	}

	template <typename T>
	bool HMC_abstract<T>::setCurrentSample(const math::vertex<T>& s)
	{
		std::lock_guard<std::mutex> lock(updating_sample);
		q_overwritten = true; // signals the sampler_loop that it should use updated q and not overwrite global q
		q = s;
		return true;

	}


	template <typename T>
	math::vertex<T> HMC_abstract<T>::getMean() const
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
		std::lock_guard<std::mutex> lock(solution_lock);;

		if(!latestN) latestN = samples.size();
		if(latestN > samples.size()) latestN = samples.size();

		T sumErr = T(0.0f);

		for(unsigned int i=samples.size()-latestN;i<samples.size();i++)
			sumErr += U(samples[i]);

		sumErr /= T((float)latestN);

		return sumErr;
	}
  
  
	template <typename T>
	void HMC_abstract<T>::sampler_loop()
	{
		samples.clear();
		sum_mean.zero();
		// sum_covariance.zero();
		sum_N = 0;

		// q = location, p = momentum, H(q,p) = hamiltonian
		math::vertex<T> p; // q is global variable

		{
			std::lock_guard<std::mutex> lock(updating_sample);
			starting_position(q); // random starting position q

			p.resize(q.size()); // momentum is initially zero
		}

		p.zero();

		// epsilon = epsilon0/sqrt(D) in order to keep distance ||x(n+1) - x(n)|| = epsilon0 for all dimensions dim(x) = D
		T epsilon = T(0.01f); ///math::sqrt(q.size()); .. NOT!!
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
			math::vertex<T> q; // local copy of q during this iteration
			{
				std::lock_guard<std::mutex> lock(updating_sample);
				q = this->q; // reads the global q
				q_overwritten = false; // detect if somebody have changed q during computation
			}

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
#else
			// just follows the gradient..

			auto g = Ugrad(old_q);

			// std::cout << "norm(Ugrad) = " << g.norm() << std::endl;
			// std::cout << "norm(q)     = " << q.norm() << std::endl;

			q += epsilon * g;
#endif
			p = -p;

			T current_U  = U(old_q);
			T proposed_U = U(q);

			// std::cout << "current_U  = " << current_U << std::endl;
			// std::cout << "proposed_U = " << proposed_U << std::endl;
			// std::cout << "log(ratio) = " << (current_U - proposed_U) << std::endl;

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
				{
					std::lock_guard<std::mutex> lock(updating_sample);

					if(q_overwritten == false)
						this->q = q; // writes the global q
				}
				// std::cout << "ACCEPT" << std::endl;

				std::lock_guard<std::mutex> lock(solution_lock);

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
	
				if(storeSamples)
					samples.push_back(q);

				if(adaptive){
					accept_rate++;
					accept_rate_samples++;
				}
			}
			else{
				// reject (keep old_q)
				// printf("REJECT\n");
				{
					std::lock_guard<std::mutex> lock(updating_sample);
					if(q_overwritten == false)
						this->q = old_q; // writes the global q
				}

				std::lock_guard<std::mutex> lock(solution_lock);

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

				if(storeSamples)
					samples.push_back(q);

				if(adaptive){
					// accept_rate;
					accept_rate_samples++;
				}
			}


			updating_sample.unlock();


			if(adaptive){
				// use accept rate to adapt epsilon
				// adapt sampling rate every N iteration (sample)
				if(accept_rate_samples >= 50)
				{
					accept_rate /= accept_rate_samples;

					// std::cout << "ACCEPT RATE: " << accept_rate << std::endl;
					// std::cout << "EPSILON:     " << epsilon << std::endl;

					// we target to 50% accept rate because it will give
					// maximum amount of information max H(accept) which hopefully
					// leads to faster convergence of estimates (to correct value,
					// the variance of estimates should be higher though).
					// Also: changing epsilon breaks MCMC sampling jump symmetry
					// but if we have converged we hopefully are close to 50%
					// all the time meaning that epsilon don't change that much +
					// I'm currently using MCMC samplers a bit more like random search
					// optimizer methods meaning that not reaching the true distribution
					// is not that serious..
					if(accept_rate < T(0.50f)){
						epsilon = T(0.8)*epsilon;
						// std::cout << "NEW SMALLER EPSILON: " << epsilon << std::endl;
	    
					}
					else if(accept_rate > T(0.50f)){
						epsilon = T(1.0/0.8)*epsilon;
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

	}
  
  
};


namespace whiteice
{  
  template class HMC_abstract< float >;
  template class HMC_abstract< double >;
  template class HMC_abstract< math::blas_real<float> >;
  template class HMC_abstract< math::blas_real<double> >;    
};

