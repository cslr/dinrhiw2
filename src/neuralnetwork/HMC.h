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

		bool startSampler();
		bool pauseSampler();
		bool continueSampler();
		bool stopSampler();

		bool getCurrentSample(math::vertex<T>& s) const;
		bool setCurrentSample(const math::vertex<T>& s);

		unsigned int getSamples(std::vector< math::vertex<T> >& samples) const;
		unsigned int getNumberOfSamples() const;

		bool getNetwork(bayesian_nnetwork<T>& bnn);

		math::vertex<T> getMean() const;
		// math::matrix<T> getCovariance() const; // DO NOT SCALE TO HIGH DIMENSIONS

		// calculates mean error for the latest N samples, 0 = all samples
		T getMeanError(unsigned int latestN = 0) const;

		bool getAdaptive() const throw(){ return adaptive; }
    
    	private:

		whiteice::nnetwork<T> nnet;
		const whiteice::dataset<T>& data;

		math::vertex<T> q;
		mutable std::mutex updating_sample;

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

