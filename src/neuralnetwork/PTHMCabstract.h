/*
 * PTHMCabstract.h
 *
 *  Created on: 25.6.2015
 *      Author: Tomas
 */

#ifndef NEURALNETWORK_PTHMCABSTRACT_H_
#define NEURALNETWORK_PTHMCABSTRACT_H_

#include "vertex.h"
#include "HMC_abstract.h"

#include <exception>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <mutex>
#include <memory>


namespace whiteice {

/**
 * Parallel Tempering class for HMC_abstract HMC sampler
 */
template <typename T = math::blas_real<float> >
class PTHMC_abstract {
public:
	// deepness = the number of different temperatures
	PTHMC_abstract(unsigned int deepness, bool adaptive);
	virtual ~PTHMC_abstract();

	// creates HMC_abstract<T> class with a new (PTHMC_abstract deletes it)
	// storeSamples: should created HMC store old samples
	// adaptive : should HMC use adaptive step length
	virtual std::shared_ptr< HMC_abstract<T> > newHMC(bool storeSamples, bool adaptive) = 0;

	bool startSampler();
	bool pauseSampler();
	bool continueSampler();
	bool stopSampler();

	unsigned int getSamples(std::vector< math::vertex<T> >& samples) const;
	unsigned int getNumberOfSamples() const;

	const HMC_abstract<T>& getHMC() const; // returns lowest level sampler (real sampler with no temperature)

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
	const bool adaptive;
	const unsigned int deepness;


	mutable std::mutex sampler_lock;
	std::list < std::shared_ptr< whiteice::HMC_abstract<T> > > hmc;
	bool dynamic_pt; // does we add and remove new temperatures if accept rates are too high/low?

	std::thread* parallel_tempering_thread; // FIXME uses raw pointers
	volatile bool running;
	volatile bool paused;

	void parallel_tempering();

	double accepts;
	double total_tries;


};


extern template class PTHMC_abstract< float >;
extern template class PTHMC_abstract< double >;
extern template class PTHMC_abstract< math::blas_real<float> >;
extern template class PTHMC_abstract< math::blas_real<double> >;



} /* namespace whiteice */

#endif /* NEURALNETWORK_PTHMCABSTRACT_H_ */
