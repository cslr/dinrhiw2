/*
 * GBRBM.h
 *
 *  Created on: 20.6.2015
 *      Author: Tomas
 */

#ifndef NEURALNETWORK_GBRBM_H_
#define NEURALNETWORK_GBRBM_H_

#include <vector>
#include <random>
#include "vertex.h"
#include "matrix.h"
#include "RNG.h"

namespace whiteice {

/**
 * Implement Gaussian-Bernoulli RBM
 */
template <typename T = math::blas_real<float> >
class GBRBM {
public:
    // creates 1x1 network, used to load some useful network
    GBRBM();
    GBRBM(const GBRBM<T>& rbm);

    // creates 2-layer: V * H network
    GBRBM(unsigned int visible, unsigned int hidden) throw(std::invalid_argument);

    virtual ~GBRBM();

    GBRBM<T>& operator=(const GBRBM<T>& rbm);

    bool resize(unsigned int visible, unsigned int hidden);

    ////////////////////////////////////////////////////////////

    unsigned int getVisibleNodes() const throw();
    unsigned int getHiddenNodes() const throw();

    void getVisible(math::vertex<T>& v) const;
    bool setVisible(const math::vertex<T>& v);

    void getHidden(math::vertex<T>& h) const;
    bool setHidden(const math::vertex<T>& h);

    bool reconstructData(unsigned int iters = 1);
    bool reconstructData(std::vector< math::vertex<T> >& samples, unsigned int iters = 1);

    // reconstruct samples data using v->h->v transform using
    // samples of RBM parameters q ~ p(q|data)
    bool reconstructDataBayesQ(std::vector< math::vertex<T> >& samples,
    		const std::vector< math::vertex<T> >& qparameters);

    bool reconstructDataHidden(unsigned int iters = 1);

    bool sampleHidden(math::vertex<T>& h, const math::vertex<T>& v); // sample from p(h|v)
    bool sampleVisible(math::vertex<T>& v, const math::vertex<T>& h); // sample from p(v|h)

    void getParameters(math::matrix<T>& W, math::vertex<T>& a, math::vertex<T>& b, math::vertex<T>& var) const;
    bool setParameters(const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& var);

    void getVariance(math::vertex<T>& var) const throw();
    bool setVariance(const math::vertex<T>& var) throw();

    bool setLogVariance(const math::vertex<T>& z);
    bool getLogVariance(math::vertex<T>& z) const;

    bool initializeWeights(); // initialize weights to small values

    // calculates single epoch for updating weights using CD-1 and
    // returns reconstruction error
    // EPOCHS control quality of the solution, 1 epoch goes through data once
    // but higher number of EPOCHS mean data calculations can take longer (higher quality)
    T learnWeights(const std::vector< math::vertex<T> >& samples,
    		const unsigned int EPOCHS=1,
    		bool verbose = false, bool learnVariance = false);

    // estimates log(P(samples|params)) of the RBM
    T logProbability(const std::vector< math::vertex<T> >& samples);

    bool setDataStatistics(const std::vector< math::vertex<T> >& samples);

    // samples given number of samples from P(v|params)
    // [GBRBM must have training data (setDataStatistics() call) or the call fails!]
    bool sample(const unsigned int SAMPLES, std::vector< math::vertex<T> >& samples,
    		const std::vector< math::vertex<T> >& statistics_training_data);

    // calculates mean reconstruction error: E[||x - reconstructed(x)||^k], reconstruct(x) = P(x_new|h) <= P(h|x_old)
    T reconstructError(const std::vector< math::vertex<T> >& samples);

    ////////////////////////////////////////////////////////////
    // for HMC sampler (and parallel tempering): calculates energy U(q) and Ugrad(q)

    // set data points parameters for U(q) and Ugrad(q) calculations
    bool setUData(const std::vector< math::vertex<T> >& samples);

    // sets temperature of the GB-RBM U(q) distribution.. As described in Cho et. al (2011) paper
    // 1 means we use RBM and 0 means most degrees of freedom [do not follow RBM parameters]
    bool setUTemperature(const T temperature);
    T getUTemperature();

    unsigned int qsize() const throw(); // size of q vector q = [a, b, z, vec(W)]

    // converts (W, a, b, z) parameters into q vector
    bool convertParametersToQ(const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b,
    		const math::vertex<T>& z, math::vertex<T>& q) const;

    // converts q vector into parameters (W, a, b, z)
    bool convertQToParameters(const math::vertex<T>& q, math::matrix<T>& W, math::vertex<T>& a, math::vertex<T>& b,
    		math::vertex<T>& z) const;

    // sets (W, a, b, z) parameters according to q vector
    bool setParametersQ(const math::vertex<T>& q);

    T U(const math::vertex<T>& q) const throw(); // calculates U(q) = -log(P(data|q))

    T Udiff(const math::vertex<T>& q1, const math::vertex<T>& q2) const;

    math::vertex<T> Ugrad(const math::vertex<T>& q) throw(); // calculates grad(U(q))


    ////////////////////////////////////////////////////////////
    // load & saves RBM data from/to file

    bool load(const std::string& filename) throw();
    bool save(const std::string& filename) const throw();

protected:
    // estimates ratio of Z values of unscaled p(v|params) distributions: Z1/Z2 using AIS Monte Carlo sampling.
    // this is needed by Udiff() which calculates difference of two P(params|v) distributions..
    T log_zratio(const math::vertex<T>& m, const math::vertex<T>& s, // data mean and variance used by the AIS sampler
    			const math::matrix<T>& W1, const math::vertex<T>& a1, const math::vertex<T>& b1, math::vertex<T>& z1,
				const math::matrix<T>& W2, const math::vertex<T>& a2, const math::vertex<T>& b2, math::vertex<T>& z2) const;


    // generates SAMPLES {v,h}-samples from p(v,h|params) using Gibbs sampling (CD) and parallel tempering
    void ais_sampling(std::vector< math::vertex<T> >& vs, const unsigned int SAMPLES,
    		const math::vertex<T>& m, const math::vertex<T>& s,
			const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z) const;

    // parallel tempering annealed importance sampling estimation of logZ and samples v from distribution p(v)
    T ais(T& logZ,
    		const math::vertex<T>& m, const math::vertex<T>& s,
    	    const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z) const;

    T unscaled_log_probability(const math::vertex<T>& v) const;

    T unscaled_log_probability(const math::vertex<T>& v,
    		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z) const;

    // calculates P-ratio: P(data1)/P(data2) using geometric mean of probabilities
    T p_ratio(const std::vector< math::vertex<T> >& data1, const std::vector< math::vertex<T> >& data2);


    T meanEnergy(const std::vector< math::vertex<T> >& samples,
    		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z);

    T E(const math::vertex<T>& v, const math::vertex<T>& h,
    		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z) const;

    T normalrnd() const; // N(0,1)
    math::vertex<T> normalrnd(const math::vertex<T>& m, const math::vertex<T>& v) const; // N(m,v)

    void sigmoid(const math::vertex<T>& input, math::vertex<T>& output) const;

    math::vertex<T> negative_phase_q(const unsigned int SAMPLES,
    		const math::matrix<T>& qW, const math::vertex<T>& qa, const math::vertex<T>& qb, const math::vertex<T>& qz) const;


    math::vertex<T> reconstruct_gbrbm_data(const math::vertex<T>& v,
    		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z,
			unsigned int CDk) const;

    math::vertex<T> reconstruct_gbrbm_hidden(const math::vertex<T>& v,
    		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z,
    		unsigned int CDk);

    T reconstruct_gbrbm_data_error(const std::vector< math::vertex<T> >& samples, unsigned int N,
    		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z,
    		unsigned int CDk);
private:
    // parameters of GB-RBM network

    math::vertex<T> a, b, z;
    math::matrix<T> W;

    // input/output vectors of GB-RBM network
    math::vertex<T> h;
    math::vertex<T> v;

	// used by learnWeights()
	math::vertex<T> data_mean;
	math::vertex<T> data_var;


	// for U(q) and Ugrad(q) calculations (for PT-HMC sampling)
	std::vector< math::vertex<T> > Usamples;
	math::vertex<T> Umean, Uvariance;
	T temperature;

	RNG<T> rng;
};


	extern template class GBRBM< float >;
	extern template class GBRBM< double >;
	extern template class GBRBM< math::blas_real<float> >;
	extern template class GBRBM< math::blas_real<double> >;

} /* namespace whiteice */

#endif /* NEURALNETWORK_GBRBM_H_ */
