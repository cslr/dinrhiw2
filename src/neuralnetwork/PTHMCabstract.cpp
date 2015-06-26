/*
 * PTHMCabstract.cpp
 *
 *  Created on: 25.6.2015
 *      Author: Tomas
 */

#include "PTHMCabstract.h"
#include "vertex.h"
#include <chrono>

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

	return hmc.front()->getSamples(samples); // hmc[0]->getSamples(samples);
}


template <typename T>
unsigned int PTHMC_abstract<T>::getNumberOfSamples() const
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0) return 0; // not running

	return hmc.front()->getNumberOfSamples();
}


// returns lowest level sampler (real sampler with no temperature)
template <typename T>
const HMC_abstract<T>& PTHMC_abstract<T>::getHMC() const
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0)
		throw std::logic_error("No root level HMC");

	return *(hmc.front());
}


template <typename T>
math::vertex<T> PTHMC_abstract<T>::getMean() const
{
	std::lock_guard<std::mutex> lock(sampler_lock);
	if(hmc.size() <= 0)
		throw std::logic_error("No root level HMC");

	return hmc.front()->getMean();
}


template <typename T>
void PTHMC_abstract<T>::parallel_tempering()
{
	double hz = 1.0;
	unsigned int ms = (unsigned int)(1000.0/hz);

	// global swap accept probability between all chains:
	// easy way to see if the parallel tempering is working more or less correctly
	accepts = 0;
	total_tries = 0;

	// on average we want to jump between every 10 samples so we measure
	// how long it seems take to get 10 samples
	unsigned int delay = 0;
	while(hmc.front()->getNumberOfSamples() < 10 && running){
		std::this_thread::sleep_for(std::chrono::milliseconds(100)); // waits for 100ms
		delay += 100;
	}

	ms = delay/10;
	hz = 1000.0/ms;

	// algorithm for keeping jump probabilities between HMC samplers good:
	// we measure sample exchange probability between chain i and j, if Pij < 10%
	// then new sampler with temperature = (temperature_i + temperature_j)/2 is added
	// on the other and, if exchange probability Pij is over 50% (?) then j is removed
	// from the chain and jumping will happen between states i and (j+1) from this point on.
	// This is a bit tricky to implement but is rather simple way to keep temperature range
	// healthy so that there is not too many or too small number of chains. Additionally, we
	// always keep at least 2 different temperatures: 1 (bottom) and 0 (top level) and have
	// maximum number of temperatures so that we cannot run out of resources.
	this->dynamic_pt = true;

	std::list< std::list<T> > acceptRate; // keeps track of accept rates between chains
	acceptRate.resize(hmc.size()-1);


	while(running){
		// pauses sampling for each thread
		// for(auto h : hmc) h->pauseSampler();

		math::vertex<T> w1, w2;

		auto startTime = std::chrono::system_clock::now();
		auto arate = acceptRate.begin();
		auto h1 = hmc.begin();
		auto h2 = hmc.begin();
		h2++;

		std::cout << "Number of PT chains: " << hmc.size() << std::endl;

		for(int i=0;i<(hmc.size()-1);){
			// tries to do MCMC swap between samples..

			// not a bulletproof test but skips chains which didn't produce useful data since last check..
			if((h1->get()->getUpdated() == false) && (h2->get()->getUpdated() == false)){
				// no new samples since the last check, skip this one..

				std::cout << "Skipping chains: " << i << " and " << (i+1) << "." << std::endl;

				i++;
				h1++;
				h2++;
				arate++;

				continue;
			}
			else{
				std::cout << "New data from chains : " << i << " and " << (i+1) << "." << std::endl;
			}

			h1->get()->getCurrentSample(w1);
			h2->get()->getCurrentSample(w2);

			h1->get()->setUpdated(false);
			// h2->get()->setUpdated(false);

			T E11 = h1->get()->U(w1);
			T E22 = h2->get()->U(w2);
			T E12 = h1->get()->U(w2);
			T E21 = h2->get()->U(w1);

			T p = math::exp( (E11 + E22) - (E12 + E21) );

			if(p >= T(1.0))
				p = T(1.0);

			if(i <= 0)
				std::cout << "p(" << i << "," << (i+1) << ") = " << p << std::endl;

			if(T(rand()/(double)RAND_MAX) < p){
				accepts++;

				if(i <= 0){
					std::cout << "SWAP! " << i << " p = " << p << std::endl;
					fflush(stdout);
				}

				h2->get()->setCurrentSample(w1);
				h1->get()->setCurrentSample(w2);
			}

			if(dynamic_pt){

				arate->push_back(p);
				while(arate->size() > 10) // keeps only the last 10 accept probabilities
					arate->pop_front();

				// calculates average accept rate between (i) and (i+1)
				T avg_accept_rate = T(0.0);
				for(auto& a : *arate){
					avg_accept_rate += a;
				}
				avg_accept_rate /= T(arate->size());

				if(avg_accept_rate <= T(0.10) && arate->size() >= 10){ // too low accept rate: insert new chain between (i-1) and i
					std::cout << "Too low accept rate between chains " << i << " and " << i+1 << std::endl;
					std::cout << "Temperatures: " << h1->get()->getTemperature() << " and " << h2->get()->getTemperature() << std::endl;
					std::cout << "Accept rate = " << avg_accept_rate << std::endl;

					std::lock_guard<std::mutex> lock(sampler_lock);

					auto h = newHMC(false, adaptive);
					const T temperature = (h1->get()->getTemperature() + h2->get()->getTemperature())/T(2.0);
					h->setTemperature(temperature);

					// randomly selects starting point between neighbouring chains
					if(rand()%1 == 0)
						h->setCurrentSample(w1);
					else
						h->setCurrentSample(w2);

					// invalidates current accept rates h1->h2 (we will have h1->h->h2)
					arate->clear();

					// also generates new empty accept ratio list about accept p-values
					std::list<T> new_aratios_list;

					arate++; // h2->h3 accept rate list
					acceptRate.insert(arate, new_aratios_list); // inserts new empty accept rate list for h->h2
					arate--; // h->h2 (new empty list)
					arate--; // h1->h (invalid now)

					h->startSampler();

					hmc.insert(h2, h); // h1->h->h2

					// tricky here: updates iterators and indexes for the for-loop
					h2--;
					total_tries++;
					// retry at h1: h1->h->h3 now

					continue;
				}
#if 0
				else if(avg_accept_rate >= T(0.90) && arate->size() >= 10){ // too high accept: remove chain h2
					std::cout << "Too high accept rate between chains " << i << " and " << i+1 << std::endl;
					std::cout << "Temperatures: " << h1->get()->getTemperature() << " and " << h2->get()->getTemperature() << std::endl;
					std::cout << "Accept rate = " << avg_accept_rate << std::endl;

					std::lock_guard<std::mutex> lock(sampler_lock);

					auto h = h2;
					h++;

					if(h != hmc.end()){ // h2 is not the last valid element (there is h3 after h2)
						h2->get()->stopSampler();
						h2 = hmc.erase(h2); // moves h2 to h3
						arate->clear(); // invalidates accept rates from h1 to h2
						arate++;
						arate = acceptRate.erase(arate); // removes accept rates h2->h3 because h2 was deleted
						arate--; // moves back to h1 accept rate (now h1->h3)

						total_tries++;

						continue; // rerun checks for h1->h3 jumps
					}
					else if(h1 != hmc.begin()){
						// h2 is the last valid element (highest inverse temperature T=0), remove h1 instead
						// but DO NOT remove h1 if it is the first element of the list [always keep T=0 and T=1]
						arate = acceptRate.erase(arate); // remove h1->h2 accept rate table because there is no h1->h3
						arate--;
						arate->clear(); // invalidates h0->h1 accept rate table because h1 was deleted
						arate++;
						h1->get()->stopSampler();
						h1 = hmc.erase(h1); // remove h1 and make it point to h2 (normal progress forward)

						h2++; // moves h2 to theoretical h3 (end() iterator instead)
						i++;  // does not really matter now but guarantees we stop processing

						total_tries++;

						continue;
					}
					// else [there is only two elements do not remove anything]

				}
#endif
			}

			i++;
			h1++;
			h2++;
			arate++;
			total_tries++;
		}

		auto endTime = std::chrono::system_clock::now();
		auto loopDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

		// continue sampling in each thread
		// for(auto h : hmc) h->continueSampler();

		if(paused){
			while(paused && running){
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
			}
		}
		else{
			if(ms > loopDuration.count())
				std::this_thread::sleep_for(std::chrono::milliseconds(ms - loopDuration.count()));
		}
	}

}


template class PTHMC_abstract< float >;
template class PTHMC_abstract< double >;
template class PTHMC_abstract< math::blas_real<float> >;
template class PTHMC_abstract< math::blas_real<double> >;


} /* namespace whiteice */
