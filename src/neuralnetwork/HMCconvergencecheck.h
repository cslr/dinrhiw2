/*
 * HMCconvergencecheck.h
 *
 *  Created on: 17.6.2015
 *      Author: Tomas
 */

#ifndef NEURALNETWORK_HMCCONVERGENCECHECK_H_
#define NEURALNETWORK_HMCCONVERGENCECHECK_H_

#include "HMC.h"
#include "matrix.h"
#include <memory>

namespace whiteice {

/**
 * HMC neural network sampler with sampling based convergence check
 */
template <typename T = math::blas_real<float> >
class HMC_convergence_check: public HMC<T> {
public:
	HMC_convergence_check(const whiteice::nnetwork<T>& net, const whiteice::dataset<T>& ds, bool adaptive=false, T alpha = T(0.5));
	virtual ~HMC_convergence_check();

        bool startSampler();
	bool pauseSampler();
	bool continueSampler();
	bool stopSampler();

	bool hasConverged(); // checks for convergence

protected:
	whiteice::HMC<T>* subhmc; // to compare convergence against

	// calculates probability p(x|m,S) ~ Normal(m,S)
	T normalprob(const math::vertex<T>& x, const math::vertex<T>& m, const T& detS, const math::matrix<T>& Sinv) const;
};


} /* namespace whiteice */

namespace whiteice
{
  extern template class HMC_convergence_check< math::blas_real<float> >;
  extern template class HMC_convergence_check< math::blas_real<double> >;
};

#endif /* NEURALNETWORK_HMCCONVERGENCECHECK_H_ */
