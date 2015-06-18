/*
 * PTHMC.h
 *
 *  Created on: 18.6.2015
 *      Author: Tomas
 */

#ifndef NEURALNETWORK_PTHMC_H_
#define NEURALNETWORK_PTHMC_H_

#include "HMC.h"
#include <mutex>
#include <thread>

namespace whiteice {

/**
 * Parallel Tempering HMC
 */
template < typename T=whiteice::math::blas_real<float> >
class PTHMC {
public:
	PTHMC(unsigned int deepness, const whiteice::nnetwork<T>& net, const whiteice::dataset<T>& ds, bool adaptive=false, T alpha = T(0.5));
	virtual ~PTHMC();

	bool startSampler();
	bool pauseSampler();
	bool continueSampler();
	bool stopSampler();

	unsigned int getSamples(std::vector< math::vertex<T> >& samples) const;
	unsigned int getNumberOfSamples() const;

	bool getNetwork(bayesian_nnetwork<T>& bnn);

	math::vertex<T> getMean() const;

	// calculates mean error for the latest N samples, 0 = all samples
	T getMeanError(unsigned int latestN = 0) const;

	T getAcceptRate() const throw(){
		if(total_tries <= 1.0)
			return 0.0;
		else
			return (accepts/total_tries);
	}

protected:
	const whiteice::nnetwork<T>& net;
	const whiteice::dataset<T>&  ds;
	const bool adaptive;
	const T alpha;

	std::vector< T > temperature;

	std::mutex sampler_lock;
	std::vector < whiteice::HMC<T>* > hmc;

	std::thread* parallel_tempering_thread;
	volatile bool running;
	volatile bool paused;

	void parallel_tempering();

	double accepts;
	double total_tries;
};

} /* namespace whiteice */


namespace whiteice
{
	extern template class PTHMC< float >;
	extern template class PTHMC< double >;
	extern template class PTHMC< math::blas_real<float> >;
	extern template class PTHMC< math::blas_real<double> >;
};


#endif /* NEURALNETWORK_PTHMC_H_ */
