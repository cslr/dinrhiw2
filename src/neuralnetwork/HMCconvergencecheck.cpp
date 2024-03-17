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
	// subhmc = new HMC<T>(net, ds, adaptive, alpha);
}

template <typename T>
HMC_convergence_check<T>::~HMC_convergence_check() {
	// if(subhmc) delete subhmc;
}

template <typename T>
bool HMC_convergence_check<T>::startSampler(){
	// subhmc->startSampler();
        return ((HMC<T>*)this)->startSampler();
}


template <typename T>
bool HMC_convergence_check<T>::pauseSampler(){
	// subhmc->pauseSampler();
	return ((HMC<T>*)this)->pauseSampler();
}


template <typename T>
bool HMC_convergence_check<T>::continueSampler(){
	// subhmc->continueSampler();
	return ((HMC<T>*)this)->continueSampler();
}


template <typename T>
bool HMC_convergence_check<T>::stopSampler(){
	// subhmc->stopSampler();
	return ((HMC<T>*)this)->stopSampler();
}


template <typename T>
bool HMC_convergence_check<T>::hasConverged() // checks for convergence
{
	// convergence in terms of aprox. MLE sampler is "tricky"
	// first we are calculating "MLE estimate" of proper pdf so
	// the distribution is incorrect in the first place,
	// second, HMC samplers get stuck into local minima
	// and do not really converge to same solution so what we really
	// get from HMC sampler is "local mode distribution"
	// therefore our convergence estimation must be more modest

	// we calculate sqrt( ||var||^2 / ||mean||^2 ) and decide for
	// convergence when SAMPLE VARIANCE OF MEAN is small enough
	// so that the error in mean when compared to the overall magnitude
	// of mean is small (average value convergeces to normal distribution
	// so one might be able to do real probability calculations here too)

	std::vector< math::vertex<T> > samples;

	if(this->getSamples(samples) <= 0)
		return false; // no samples

	const unsigned int dim = samples[0].size();

	if(samples.size() < dim)
		return false; // not enough samples (we want S to be full rank)

	// we calculate N(m, S) parameters and then calculate

	math::vertex<T> m;
	math::matrix<T> S;

	m.resize(dim);
	S.resize(dim, dim);

	m.zero();
	S.zero();

	for(auto& s : samples){
		m += s;
		S += s.outerproduct();
	}

	m /= T(samples.size());
	S /= T(samples.size());

	S -= m.outerproduct(); // slow/wastes memory [+ should divide by 1/N-1 for sample variance but we don't care]

	// now Average[x] ~ N(m, S/N)

	math::vertex<T> d(dim);

	for(unsigned int i=0;i<dim;i++)
		d[i] = math::sqrt( math::abs(S(i,i))/T(samples.size()) );

	std::cout << "|d| = " << d.norm() << std::endl;
	std::cout << "|m| = " << m.norm() << std::endl;

	T r = d.norm()/m.norm();

	float ratio = 0.0f;
	math::convert(ratio, r);

	printf("hmc convergence check ratio: %f  (%d)\n", ratio, this->getNumberOfSamples());
	std::cout << r << std::endl;
	fflush(stdout);

	if(r < T(0.00001)) // this must be > 1 to guarantee convergence..
		return true;
	else
		return false;


#if 0

	std::vector< math::vertex<T> > samples1;
	std::vector< math::vertex<T> > samples2;

	if(this->getSamples(samples1) <= 0 || subhmc->getSamples(samples2) <= 0)
		return false; // no samples

	const unsigned int dim = samples1[0].size();

	if(samples1.size() <= 2*dim || samples2.size() <= 2*dim)
		return false; // not enough samples

	// we calculate N(m, S) parameters and then calculate

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


	// we calculate difference delta=x-y
	// now Var[x-y] = Var[x]+Var[y]
	// maximum value (one sigma probability) for difference is |(x-y) +- sqrt(var(x)+var(y))|
	// which leads also to maximum value for norm (delta)
	// now this value is compared to norm(m1) and norm(m2) and the ratio between too
	// must be "small enough" for convergence, this is statistically sound comparision although
	// we don't know its exact probability..

	auto delta = m1 - m2;
	for(unsigned int i=0;i<delta.size();i++){
		delta[i] = math::abs(delta[i]) + math::sqrt(S1(i,i)/samples1.size() + S2(i,i)/samples2.size()); // use mean sample variance
	}
	auto emax = delta.norm();

	auto r1 = emax / m1.norm();
	auto r2 = emax / m2.norm();
	auto r = r1 > r2 ? r1 : r2;

	float ratio = 0.0f;
	math::convert(ratio, r);

	printf("hmc convergence check ratio: %f\n", ratio);
	fflush(stdout);

	if(r < T(0.001)) // this must be order of 0.001 to guarantee convergence..
		return true;
	else
		return false;



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

#endif
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
  template class HMC_convergence_check< math::blas_real<float> >;
  template class HMC_convergence_check< math::blas_real<double> >;
};

