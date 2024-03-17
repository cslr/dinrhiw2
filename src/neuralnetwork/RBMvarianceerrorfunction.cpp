/*
 * RBMvarianceerrorfunction.cpp
 *
 *  Created on: 22.6.2015
 *      Author: Tomas
 */

#include "RBMvarianceerrorfunction.h"
#include <assert.h>

namespace whiteice {

template <typename T>
GBRBM_variance_error_function<T>::GBRBM_variance_error_function(const std::vector< math::vertex<T> >& samples__,
		unsigned int numHidden) :
		samples(samples__)
{
	// calculates variance of the data
	if(samples.size() > 0){
		variance.resize(samples[0].size());
		variance.zero();

		auto mean = variance;

		for(auto& s : samples){
			mean += s;
			for(unsigned int i=0;i<s.size();i++)
				variance[i] += s[i]*s[i];
		}

		mean /= T(samples.size());
		variance /= T(samples.size());

		for(unsigned int i=0;i<variance.size();i++)
			variance[i] -= mean[i]*mean[i];

		this->numHidden = numHidden;
	}
}


template <typename T>
GBRBM_variance_error_function<T>::~GBRBM_variance_error_function(){

}


template <typename T>
void GBRBM_variance_error_function<T>::getRealVariance(math::vertex<T>& input) const
{
	for(unsigned int i=0;i<input.size();i++)
		input[i] = math::exp(input[i]);
}


// calculates value of function
template <typename T>
T GBRBM_variance_error_function<T>::operator() (const math::vertex<T>& x) const {
	T value = T(10e6);
	calculate(x, value);
	return value;
}


// calculates value
template <typename T>
T GBRBM_variance_error_function<T>::calculate(const math::vertex<T>& x) const {
	T value = T(10e6);
	calculate(x, value);
	return value;
}


// calculates value
// (optimized version, this is faster because output value isn't copied)
template <typename T>
void GBRBM_variance_error_function<T>::calculate(const math::vertex<T>& x, T& y) const {
	y = T(10e6);

	if(x.size() != variance.size())
		return;

#if 0
	auto input = x;
	for(unsigned int i=0;i<x.size();i++){
		input[i] *= variance[i];
		if(input[i] < 0.01)
			return; // do not allow zero variances..
	}
#endif

	whiteice::GBRBM<T> rbm;
	rbm.resize(x.size(), numHidden);
	rbm.initializeWeights();
	if(rbm.setLogVariance(x) == false)
		return;

	y = rbm.learnWeights(samples, 2, false, false);

	auto var = x;
	this->getRealVariance(var);

	std::cout << y << " = " << var << std::endl;
}


// creates copy of object
template <typename T>
function<math::vertex<T>,T>* GBRBM_variance_error_function<T>::clone() const {
	return new GBRBM_variance_error_function<T>(samples, numHidden);
}


// returns input vectors dimension
template <typename T>
unsigned int GBRBM_variance_error_function<T>::dimension() const  {
	if(samples.size() > 0)
		return samples[0].size();
	else
		return 0;
}


template <typename T>
bool GBRBM_variance_error_function<T>::hasGradient() const  {
	return false;
}


template <typename T>
math::vertex<T> GBRBM_variance_error_function<T>::grad(math::vertex<T>& x) const {
	assert(0);
}


template <typename T>
void GBRBM_variance_error_function<T>::grad(math::vertex<T>& x, math::vertex<T>& y) const {
	assert(0);
}


template class GBRBM_variance_error_function<float>;
template class GBRBM_variance_error_function<double>;
template class GBRBM_variance_error_function< math::blas_real<float> >;
template class GBRBM_variance_error_function< math::blas_real<double> >;


} /* namespace whiteice */
