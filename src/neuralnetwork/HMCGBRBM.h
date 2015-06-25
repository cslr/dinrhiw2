/*
 * HMCGBRBM.h
 *
 *  Created on: 25.6.2015
 *      Author: Tomas
 */

#ifndef NEURALNETWORK_HMCGBRBM_H_
#define NEURALNETWORK_HMCGBRBM_H_

#include "HMC_abstract.h"
#include "GBRBM.h"

namespace whiteice {

template <typename T = math::blas_real<float> >
class HMC_GBRBM: public HMC_abstract<T> {
public:
	HMC_GBRBM(const std::vector< math::vertex<T> >& samples, unsigned int numHiddenNodes, bool adaptive=false);
	virtual ~HMC_GBRBM();

	GBRBM<T>& getRBM() throw();

	// temperature must be in [0,1] interval
	// 0 = means most freely changing parameters and RBM is just N(m, S) machine
	// 1 = means to fit into data as well as possible and RBM is non-linear P(v) estimator
	bool setTemperature(T temperature);

    // probability functions for hamiltonian MC sampling of
    // P ~ exp(-U(q)) distribution
    virtual T U(const math::vertex<T>& q) const;
    virtual math::vertex<T> Ugrad(const math::vertex<T>& q);

    // a starting point q for the sampler (may not be random)
    virtual void starting_position(math::vertex<T>& q) const;

protected:
    const std::vector< math::vertex<T> >& samples;

    GBRBM<T> rbm;

};


extern template class HMC_GBRBM< float >;
extern template class HMC_GBRBM< double >;
extern template class HMC_GBRBM< math::blas_real<float> >;
extern template class HMC_GBRBM< math::blas_real<double> >;

} /* namespace whiteice */

#endif /* NEURALNETWORK_HMCGBRBM_H_ */
