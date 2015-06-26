/*
 * PTHMCabstract.cpp
 *
 *  Created on: 25.6.2015
 *      Author: Tomas
 */

#include "PTHMCabstract.h"
#include "vertex.h"

namespace whiteice {

template <typename T>
PTHMC_abstract<T>::PTHMC_abstract(unsigned int deepness_, bool adaptive_) :
	deepness(deepness_), adaptive(adaptive_)
{
	parallel_tempering_thread = nullptr;
	running = false;
	paused  = false;

	accepts = 0.0;
	total_tries = 0.0;
}


template <typename T>
PTHMC_abstract<T>::~PTHMC_abstract()
{
	stopSampler(); // if needed
}


template <typename T>
bool PTHMC_abstract<T>::startSampler()
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(running) return false; // already running

	hmc.clear();

	try{

		for(unsigned int i=0;i<deepness;i++){
			T temperature = T(((double)i)/((double)(deepness-1)));
			temperature = T(1.0) - temperature;
			bool storeSamples = (i==0);
			// only store samples of the "bottom" sampler [no temperature]
			auto h = newHMC(storeSamples, adaptive);
			h->setTemperature(temperature);
			hmc.push_back(h);
		}

		for(auto h : hmc)
			if(h->startSampler() == false)
				throw std::runtime_error("Cannot start sampler");
	}
	catch(std::exception& e){
		hmc.clear();

		return false;
	}

	// starts parallel tempering thread which swaps samples between different temperature samplers
	try{
		running = true;
		paused = false;
		parallel_tempering_thread = new std::thread(parallel_tempering, this);
	}
	catch(std::exception& e){
		hmc.clear();
		parallel_tempering_thread = nullptr;
	}

	return true;
}


template <typename T>
bool PTHMC_abstract<T>::pauseSampler()
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0) return false; // not running

	bool ok = true;

	for(auto h : hmc){
		ok = ok && h->pauseSampler();
	}

	paused = true;

	return ok;
}


template <typename T>
bool PTHMC_abstract<T>::continueSampler()
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0) return false; // not running

	bool ok = true;

	for(auto h : hmc){
		ok = ok && h->continueSampler();
	}

	paused = false;

	return ok;
}


template <typename T>
bool PTHMC_abstract<T>::stopSampler()
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0) return false; // not running

	running = false;
	paused = false;

	for(auto h : hmc){
		h->stopSampler();
	}

	hmc.clear();

	parallel_tempering_thread->join();
	delete parallel_tempering_thread;
	parallel_tempering_thread = nullptr;

	return true;
}


template <typename T>
unsigned int PTHMC_abstract<T>::getSamples(std::vector< math::vertex<T> >& samples) const
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0) return 0; // not running

	return hmc[0]->getSamples(samples);
}


template <typename T>
unsigned int PTHMC_abstract<T>::getNumberOfSamples() const
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0) return 0; // not running

	return hmc[0]->getNumberOfSamples();
}


// returns lowest level sampler (real sampler with no temperature)
template <typename T>
const HMC_abstract<T>& PTHMC_abstract<T>::getHMC() const
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0)
		throw std::logic_error("No root level HMC");

	return *(hmc[0]);
}


template <typename T>
math::vertex<T> PTHMC_abstract<T>::getMean() const
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0)
		throw std::logic_error("No root level HMC");

	return hmc[0]->getMean();
}


template <typename T>
void PTHMC_abstract<T>::parallel_tempering()
{
	double hz = 1.0;
	unsigned int ms = (unsigned int)(1000.0/hz);

	accepts = 0;
	total_tries = 0;

	// on average we try to jump between every 10 samples so we measure
	// how long it seems take to get 10 samples
	unsigned int delay = 0;
	while(hmc[0]->getNumberOfSamples() < 10 && running){
		std::this_thread::sleep_for(std::chrono::milliseconds(100)); // waits for 100ms
		delay += 100;
	}

	ms = delay;
	hz = 1000.0/ms;


	while(running){
		// pauses sampling for each thread
		// for(auto h : hmc) h->pauseSampler();

		for(int i=hmc.size()-1;i>=1;i--){
			// tries to do MCMC swap between samples, starts from higher T samplers and moves to lower ones
			math::vertex<T> w1, w2;

			hmc[i]->getCurrentSample(w1);
			hmc[i-1]->getCurrentSample(w2);

			T E11 = (hmc[i]->U(w1));
			T E22 = (hmc[i-1]->U(w2));
			T E12 = (hmc[i]->U(w2));
			T E21 = (hmc[i-1]->U(w1));


			T p = math::exp( (E11 + E22) - (E12 - E21) );

			if(T(rand()/(double)RAND_MAX) < p){
				accepts++;

				if(i <= 10){
					std::cout << "SWAP! " << i << " p = " << p << std::endl;
					fflush(stdout);
				}

				hmc[i-1]->setCurrentSample(w2);
				hmc[i]->setCurrentSample(w1);
			}

			total_tries++;
		}

		// continue sampling in each thread
		// for(auto h : hmc) h->continueSampler();

		if(paused){
			while(paused && running){
				std::this_thread::sleep_for(std::chrono::milliseconds(ms));
			}
		}
		else{
			std::this_thread::sleep_for(std::chrono::milliseconds(ms));
		}
	}

}


template class PTHMC_abstract< float >;
template class PTHMC_abstract< double >;
template class PTHMC_abstract< math::blas_real<float> >;
template class PTHMC_abstract< math::blas_real<double> >;


} /* namespace whiteice */
