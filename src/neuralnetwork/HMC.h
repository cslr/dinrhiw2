/*
 * Hamiltonian Monte Carlo Markov Chain sampler 
 * for neurnal network training data.
 */

// TODO: adaptive step lengths DO NOT work very well and
//       are disabled as the default

#ifndef HMC_h
#define HMC_h

#include <vector>
#include <unistd.h>

#include <thread>
#include <mutex>

#include "vertex.h"
#include "matrix.h"
#include "dataset.h"
#include "dinrhiw_blas.h"
#include "nnetwork.h"
#include "bayesian_nnetwork.h"
#include "RNG.h"


namespace whiteice
{
	template <typename T = math::blas_real<float> >
	class HMC
	{
    	public:

		HMC(const whiteice::nnetwork<T>& net, const whiteice::dataset<T>& ds, bool adaptive=false, T alpha = T(0.5), bool store = true);
		~HMC();

		bool setTemperature(const T t); // set "temperature" for probability distribution [default T = 1 => no temperature]
		T getTemperature();             // get "temperature" of probability distribution

		// probability functions for hamiltonian MC sampling
		T U(const math::vertex<T>& q, bool useRegulizer = true) const;
		math::vertex<T> Ugrad(const math::vertex<T>& q) const;
	
	       /*
	        * error terms e = y-f(x|w) covariance matrix used during
		* sampling/optimization. you can get this term by first
		* optimizing using L-BFGS and then calculating its
		* error_covariance term
	        */
	        bool startSampler();
	
		bool pauseSampler();
		bool continueSampler();
		bool stopSampler();

		bool getCurrentSample(math::vertex<T>& s) const;
		bool setCurrentSample(const math::vertex<T>& s);

		unsigned int getSamples(std::vector< math::vertex<T> >& samples) const;
		unsigned int getNumberOfSamples() const;

	        // get latestN samples from HMC sampler or all (0)
	        bool getNetwork(bayesian_nnetwork<T>& bnn, unsigned int latestN = 0);

		math::vertex<T> getMean() const;
		// math::matrix<T> getCovariance() const; // DO NOT SCALE TO HIGH DIMENSIONS

		// calculates mean error for the latest N samples, 0 = all samples
		T getMeanError(unsigned int latestN = 0) const;

		bool getAdaptive() const throw(){ return adaptive; }
	
    	private:
	        // performs leapfrog operation (keeps probability function constant)
	        void leapfrog(math::vertex<T>& p, math::vertex<T>& q, const T epsilon, const unsigned int L) const;
	
	        // calculates negative phase gradient
	        bool negative_phase(const math::vertex<T>& xx, const math::vertex<T>& yy,
				    math::vertex<T>& grad, 
				    whiteice::nnetwork<T>& nnet) const;
			    
	
	        bool solve_inverse(const math::vertex<T>& y, 
				   std::vector< math::vertex<T> >& inverses,
				   const whiteice::nnetwork<T>& nnet,
				   const whiteice::dataset<T>& data) const;
	
	        bool sample_covariance_matrix(const math::vertex<T>& q);
	
	        // calculates z-ratio between data likelihood distributions
	        T zratio(const math::vertex<T>& q1, const math::vertex<T>& q2) const;
	
		whiteice::nnetwork<T> nnet;
	        const whiteice::dataset<T>& data;
		math::vertex<T> q;
		mutable std::mutex updating_sample;
	
	        T sigma2;
	
	        std::vector< math::vertex<T> > samples;

		T alpha; // prior distribution parameter for neural networks (gaussian prior)
		T temperature; // temperature parameter for the probability function
		bool adaptive;
		bool store;

		// used to calculate statistics when needed
		math::vertex<T> sum_mean;
		// math::matrix<T> sum_covariance;
		unsigned int sum_N;

		volatile bool running, paused;

		mutable std::vector<std::thread*> sampling_thread; // threads
		mutable std::mutex solution_lock, start_lock;
	
	        whiteice::RNG<T> rng;

		void sampler_loop();
	};

};


namespace whiteice
{
	extern template class HMC< float >;
	extern template class HMC< double >;
	extern template class HMC< math::blas_real<float> >;
	extern template class HMC< math::blas_real<double> >;
};


#endif

