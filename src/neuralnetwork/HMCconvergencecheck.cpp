/*
 * HMCconvergencecheck.cpp
 *
 *  Created on: 17.6.2015
 *      Author: Tomas
 */

#include "HMCconvergencecheck.h"
#include <exception>
#include <stdexcept>


namespace whiteice {

template <typename T>
HMC_convergence_check<T>::HMC_convergence_check(const whiteice::nnetwork<T>& net, const whiteice::dataset<T>& ds, bool adaptive, T alpha) :
		HMC<T>(net, ds, adaptive, alpha)
{
	subhmc = new HMC<T>(net, ds, adaptive, alpha);
}

template <typename T>
HMC_convergence_check<T>::~HMC_convergence_check() {
	if(subhmc) delete subhmc;
}

template <typename T>
bool HMC_convergence_check<T>::startSampler(){

	subhmc->startSampler();
	((HMC<T>*)this)->startSampler();
}


template <typename T>
bool HMC_convergence_check<T>::pauseSampler(){
	subhmc->pauseSampler();
	return ((HMC<T>*)this)->pauseSampler();
}


template <typename T>
bool HMC_convergence_check<T>::continueSampler(){
	subhmc->continueSampler();
	return ((HMC<T>*)this)->continueSampler();
}


template <typename T>
bool HMC_convergence_check<T>::stopSampler(){
	subhmc->stopSampler();
	return ((HMC<T>*)this)->stopSampler();
}


template <typename T>
bool HMC_convergence_check<T>::hasConverged() { // checks for convergence

	std::vector< math::vertex<T> > samples1;
	std::vector< math::vertex<T> > samples2;

	if(this->getSamples(samples1) <= 0 || subhmc->getSamples(samples2) <= 0)
		return false; // no samples

	const unsigned int dim = samples1[0].size();

	if(samples1.size() <= 2*dim || samples2.size() <= 2*dim)
		return false; // not enough samples

	// we calculate N(m, S) parameters and then calculate: continue-valued absolute "KL-divergence"

	math::vertex<T> m1, m2;
	math::matrix<T> S1, S2;

	m1.resize(dim);
	m2.resize(dim);
	S1.resize(dim, dim);
	S2.resize(dim, dim);

	m1.zero();
	m2.zero();
	S1.zero();
	S2.zero();

	for(auto& s : samples1){
		m1 += s;
		S1 += s.outerproduct();
	}

	for(auto& s : samples2){
		m2 += s;
		S2 += s.outerproduct();
	}

	m1 /= T(samples1.size());
	S1 /= T(samples1.size());

	m2 /= T(samples2.size());
	S2 /= T(samples2.size());

	S1 -= m1.outerproduct(); // slow/wastes memory
	S2 -= m2.outerproduct();

	T detS1 = S1.det();
	T detS2 = S2.det();

	S1.inv();
	S2.inv();

	// now we have parameters N(m, S) for distributions in each set
	// we calculate pseudo-entropy
	try{
		T H = T(0.0);

		for(auto& s : samples1)
			H += math::abs(normalprob(s, m2, detS2, S2) - normalprob(s, m1, detS1, S1));

		for(auto& s : samples2)
			H += math::abs(normalprob(s, m1, detS1, S1) - normalprob(s, m2, detS2, S2));

		H /= (samples1.size() + samples2.size());

		const T epsilon = T(0.05); // R => 1.0513

		std::cout << "HMC CONVERGENCE CHECK. R = " << math::exp(H) << std::endl;

		return (H < epsilon);
	}
	catch(std::exception& e){
		return false; // bad data so we cannot have converged yet
	}
}


// calculates log-probability log-p(x|m,S) ~ Normal(m,S)
template <typename T>
T HMC_convergence_check<T>::normalprob(const math::vertex<T>& x, const math::vertex<T>& m,
		const T& detS, const math::matrix<T>& Sinv) const
{
	// const T detS = S.det();
	if(detS <= T(0.0)) // bad distribution we just use some stupid value..
		throw std::invalid_argument("non positive definite covariance matrix");

#if 0
	whiteice::math::matrix<T> Sinv(S);

	if(Sinv.inv() == false)
		throw std::invalid_argument("non positive definite covariance matrix");
#endif


	// T q = math::exp(-T(0.5)*(((x-m)*Sinv*(x-m))[0])) / math::sqrt(detS);
	T q = -T(0.5)*(((x-m)*Sinv*(x-m))[0]) - math::log(math::sqrt(detS));

	// std::cout << "q = " << q << std::endl;

	return q; // not a proper distribution because we do not scale with (1/2*pi)*(dim/2)
}


} /* namespace whiteice */


namespace whiteice
{
	template class HMC_convergence_check< float >;
	template class HMC_convergence_check< double >;
	template class HMC_convergence_check< math::blas_real<float> >;
	template class HMC_convergence_check< math::blas_real<double> >;
};

