/*
 * GBRBM.cpp
 *
 *  Created on: 20.6.2015
 *      Author: Tomas
 */

#include "GBRBM.h"
#include <chrono>
#include <random>
#include <time.h>
#include <math.h>
#include <cmath>

namespace whiteice {

// creates 1x1 network, used to load some useful network
template <typename T>
GBRBM<T>::GBRBM()
{
	W.resize(1,1);
	a.resize(1);
	b.resize(1);
	z.resize(1);

	h.resize(1);
	v.resize(1);

	h.zero();
	v.zero();

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	generator = new std::default_random_engine (seed);
	rng = new std::normal_distribution<>(0, 1); // N(0,1) variables

	initializeWeights();
}

template <typename T>
GBRBM<T>::GBRBM(const GBRBM<T>& rbm)
{
	this->W = rbm.W;
	this->a = rbm.a;
	this->b = rbm.b;
	this->z = rbm.z;

	this->v = rbm.v;
	this->h = rbm.h;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	generator = new std::default_random_engine (seed);
	rng = new std::normal_distribution<>(0, 1); // N(0,1) variables
}

// creates 2-layer: V * H network
template <typename T>
GBRBM<T>::GBRBM(unsigned int visible, unsigned int hidden) throw(std::invalid_argument)
{
    if(visible == 0 || hidden == 0)
      throw std::invalid_argument("invalid network architecture");

    // the last term is always constant: 1 (one)
    v.resize(visible);
    h.resize(hidden);

    a.resize(visible);
    b.resize(hidden);
    z.resize(visible);
    W.resize(visible, hidden);

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	generator = new std::default_random_engine (seed);
	rng = new std::normal_distribution<>(0, 1); // N(0,1) variables

    initializeWeights();
}

template <typename T>
GBRBM<T>::~GBRBM()
{
	// very careful here.. the child GBRBMs must have zero-sized ais_rbm or this will lead to recursion
	// if(ais_rbm.size() > 0) ais_rbm.clear();
	if(generator) delete generator;
	if(rng) delete rng;
}

template <typename T>
GBRBM<T>& GBRBM<T>::operator=(const GBRBM<T>& rbm)
{
	this->W = rbm.W;
	this->a = rbm.a;
	this->b = rbm.b;
	this->z = rbm.z;

	this->v = rbm.v;
	this->h = rbm.h;

	return (*this);
}

template <typename T>
bool GBRBM<T>::resize(unsigned int visible, unsigned int hidden)
{
	if(visible == 0 || hidden == 0)
		return false;

    // the last term is always constant: 1 (one)
    v.resize(visible);
    h.resize(hidden);

    a.resize(visible);
    b.resize(hidden);
    z.resize(visible);
    W.resize(visible, hidden);

    initializeWeights();

    return true;
}

////////////////////////////////////////////////////////////

template <typename T>
unsigned int GBRBM<T>::getVisibleNodes() const throw()
{
	return W.ysize();
}


template <typename T>
unsigned int GBRBM<T>::getHiddenNodes() const throw()
{
	return W.xsize();
}

template <typename T>
void GBRBM<T>::getVisible(math::vertex<T>& v) const
{
	v = this->v;
}

template <typename T>
bool GBRBM<T>::setVisible(const math::vertex<T>& v)
{
	if(v.size() != this->v.size())
		return false;

	this->v = v;

	return true;
}

template <typename T>
void GBRBM<T>::getHidden(math::vertex<T>& h) const
{
	h = this->h;
}

template <typename T>
bool GBRBM<T>::setHidden(const math::vertex<T>& h)
{
	if(h.size() != this->h.size())
		return false;

	this->h = h;

	return true;
}

// number of iterations to simulate the system
// 1 = single step from visible to hidden and back
// from hidden to visible (CD-1)
template <typename T>
bool GBRBM<T>::reconstructData(unsigned int iters)
{
	v = reconstruct_gbrbm_data(v, W, a, b, z, iters);
	return true;
}

template <typename T>
bool GBRBM<T>::reconstructData(std::vector< math::vertex<T> >& samples, unsigned int iters)
{
	for(auto& v : samples){
		v = reconstruct_gbrbm_data(v, W, a, b, z, iters);
	}

	return true;
}


// number of iterations to simulate the system
// 1 = single step from visible to hidden
template <typename T>
bool GBRBM<T>::reconstructDataHidden(unsigned int iters)
{
	h = reconstruct_gbrbm_hidden(v, W, a, b, z, iters);
	return true;
}


// sample from p(h|v)
template <typename T>
bool GBRBM<T>::sampleHidden(math::vertex<T>& h, const math::vertex<T>& v)
{
	if(v.size() != a.size())
		return false;

	auto x = v;
	for(unsigned int i=0;i<v.size();i++)
		x[i] *= math::exp(-z[i]/T(2.0)); // t = S^-0.5 * v

	h.resize(b.size());
	x = x*W + b;

	sigmoid(x, h);

	for(unsigned int i=0;i<h.size();i++){
		T r = T((double)rand())/T((double)RAND_MAX);
		if(r <= h[i]) h[i] = T(1.0);
		else h[i] = T(0.0);
	}

	return true;
}

// sample from p(v|h)
template <typename T>
bool GBRBM<T>::sampleVisible(math::vertex<T>& v, const math::vertex<T>& h)
{
	if(h.size() != b.size())
		return false;

	v.resize(a.size());

	auto mean = W*h; // pseudo-mean requires multiplication by covariance..

	for(unsigned int i=0;i<mean.size();i++){
		v[i] = math::exp(z[i]/T(2.0))*normalrnd() +
				math::exp(z[i]/T(2.0))*mean[i] + a[i];
	}

	return true;
}


template <typename T>
void GBRBM<T>::getParameters(math::matrix<T>& W, math::vertex<T>& a, math::vertex<T>& b, math::vertex<T>& var) const
{
	W = this->W;
	a = this->a;
	b = this->b;

	var.resize(z.size());

	for(unsigned int j=0;j<var.size();j++)
		var[j] = math::exp(this->z[j]);
}

template <typename T>
bool GBRBM<T>::setParameters(const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& var)
{
	if(this->W.ysize() != W.ysize() || this->W.xsize() != W.xsize())
		return false;

	if(this->a.size() != a.size())
		return false;

	if(this->b.size() != b.size())
		return false;

	if(var.size() != this->z.size())
		return false;

	for(unsigned int j=0;j<var.size();j++)
		if(var[j] <= T(0.0)) // variances must be positive!
			return false;

	this->W = W;
	this->a = a;
	this->b = b;

	for(unsigned int j=0;j<var.size();j++)
		z[j] = math::log(var[j]);

	return true;
}


template <typename T>
void GBRBM<T>::getVariance(math::vertex<T>& var) const throw()
{
	var.resize(z.size());

	for(unsigned int j=0;j<var.size();j++)
		var[j] = math::exp(this->z[j]);
}


template <typename T>
bool GBRBM<T>::setVariance(const math::vertex<T>& var) throw()
{
	if(var.size() != z.size())
		return false;

	for(unsigned int j=0;j<var.size();j++)
		if(var[j] <= T(0.0)) // variances must be positive!
			return false;

	for(unsigned int j=0;j<var.size();j++)
		z[j] = math::log(var[j]);

	return true;
}


template <typename T>
bool GBRBM<T>::setLogVariance(const math::vertex<T>& z)
{
	this->z = z;
	return true;
}


template <typename T>
bool GBRBM<T>::getLogVariance(math::vertex<T>& z) const
{
	z = this->z;
	return true;
}


template <typename T>
bool GBRBM<T>::initializeWeights() // initialize weights to small values
{
	a.zero();
	b.zero();
	z.zero(); // initially assume var_i = 1

	for(unsigned int i=0;i<z.size();i++)
		z[i] = math::log(1.0);

	for(unsigned int j=0;j<W.ysize();j++)
		for(unsigned int i=0;i<W.xsize();i++)
			W(j,i) = T(0.1f) * normalrnd();

	return true;
}

// calculates single epoch for updating weights using CD-1 and
// returns reconstruction error
// (keep calculating until there is no improvement anymore)
template <typename T>
T GBRBM<T>::learnWeights(const std::vector< math::vertex<T> >& samples,
		const unsigned int EPOCHS, bool verbose, bool learnVariance)
{
	const unsigned int CDk = 1;

	unsigned int epoch = 0;
	T lrate  = T(0.001);

	if(samples.size() <= 0)
		return T(100000.0); // nothing to do

	std::vector< math::vertex<T> > random_samples; // generates random N(data_mean, datavar*I) data to test against

	data_mean.resize(z.size());
	data_var.resize(z.size());

	// if(verbose)
	{
		for(unsigned int i=0;i<1000;i++){
			auto& s = samples[rand() % samples.size()];
			data_mean += s;
			for(unsigned int i=0;i<s.size();i++)
				data_var[i] += s[i]*s[i];
		}

		data_mean /= T(samples.size());
		data_var  /= T(samples.size());

		for(unsigned int i=0;i<z.size();i++)
			data_var[i] -= data_mean[i]*data_mean[i];

		for(unsigned int s=0;s<1000;s++){
			auto x = data_mean;
			for(unsigned int i=0;i<x.size();i++){
				auto wide = T(2.0)*math::sqrt(data_var[i]);
				x[i] = T(rand()/((double)RAND_MAX))*wide - wide/2 + data_mean[i];
			}

			random_samples.push_back(x);
		}
	}


	std::list<T> errors;

	T error = reconstruct_gbrbm_data_error(samples, 1000, W, a, b, z, CDk);
	auto pratio         = p_ratio(samples, random_samples);
	errors.push_back(math::exp(pratio));

	if(verbose){
		math::vertex<T> var;

		T rerror = reconstruct_gbrbm_data_error(random_samples, 1000, W, a, b, z, CDk);
		auto r_ratio        = (error/rerror);
		//auto logp           = logProbability(samples);


		var = z;
		for(unsigned int i=0;i<var.size();i++) var[i] = math::exp(z[i]);

		std::cout << "START " << epoch <<
			". R: " << error <<
			" R-ratio: " << r_ratio <<
			//" log(P) : " << logp <<
			" P-ratio: " << pratio << " Variance: " << var << std::endl;
	}

	auto best_error = error;
	auto best_a = a;
	auto best_b = b;
	auto best_W = W;
	auto best_z = z;

	bool convergence = false;

	while(!convergence && epoch < EPOCHS)
	{
		math::vertex<T> da(a.size()), db(b.size()), dz(z.size());
		math::matrix<T> dW;
		dW.resize(W.ysize(),W.xsize());
		da.zero();
		db.zero();
		dz.zero();
		dW.zero();

		T N = T(0.0);

		// auto pcd_x = samples[(rand() % samples.size())];


		for(unsigned int i=0;i<1000;i++){
			// goes through data and calculates gradient
			const unsigned int NUM_SAMPLES = 100;

			// negative phase Emodel[gradient]
			std::vector< math::vertex<T> > vs;

			// randomly chooses negative samples either using AIS or CD-1
			if((rand() & 1) == 1){
				ais_sampling(vs, NUM_SAMPLES, data_mean, data_var, W, a, b, z); // gets (v) from the model
			}
			else{
				for(unsigned int j=1;j<NUM_SAMPLES;j++){
					const unsigned int index = rand() % samples.size();
					math::vertex<T> x = samples[index]; // x = visible state

					auto xx = reconstruct_gbrbm_data(x, W, a, b, z, CDk); // gets x ~ p(v) from the model
					vs.push_back(xx);
				}
			}

			math::vertex<T> da(a.size());
			math::vertex<T> db(b.size());
			math::vertex<T> dz(z.size());
			math::matrix<T> dW(W.ysize(), W.xsize());

			// needed?
			da.zero();
			db.zero();
			dz.zero();
			dW.zero();

			math::vertex<T> var(z.size());
			math::vertex<T> varx(z.size());

			for(unsigned int i=0;i<vs.size();i++){
				const auto& x = vs[i];

				for(unsigned int j=0;j<z.size();j++){
					var[j] = math::exp(-z[j]);
					varx[j] = math::exp(-z[j]/T(2.0))*x[j];
				}

				math::vertex<T> y; // y = hidden state
				sigmoid( (varx * W) + b, y);

				// calculates gradients [negative phase]
				math::vertex<T> xa(x - a);
				math::vertex<T> ga(xa); // gradient of a

				for(unsigned int j=0;j<ga.size();j++)
					ga[j] = var[j]*ga[j];

				math::vertex<T> gb(y); // gradient of b
				math::matrix<T> gW = varx.outerproduct(y); // gradient of W

				math::vertex<T> gz(z.size()); // gradient of z

				math::vertex<T> wy = W*y;

				for(unsigned int j=0;j<z.size();j++){
					gz[j] = math::exp(-z[j])*T(0.5)*xa[j]*xa[j] - math::exp(-z[j]/T(2.0))*T(0.5)*x[j]*wy[j];
				}

				da += ga;
				db += gb;
				dz += gz;
				dW += gW;
			}

			da /= T(vs.size());
			db /= T(vs.size());
			dz /= T(vs.size());
			dW /= T(vs.size());

			// calculates gradients [positive phase]: Edata[gradient]
			math::vertex<T> ta(a.size());
			math::vertex<T> tb(b.size());
			math::vertex<T> tz(z.size());
			math::matrix<T> tW(W.ysize(), W.xsize());

			// needed?
			ta.zero();
			tb.zero();
			tz.zero();
			tW.zero();


			for(unsigned int j=0;j<NUM_SAMPLES;j++)
			{
				const unsigned int index = rand() % samples.size();
				math::vertex<T> x = samples[index]; // x = visible state

				auto xa = (x - a);

				for(unsigned int j=0;j<var.size();j++)
					varx[j] = math::exp(-z[j]/T(2.0))*x[j];

				math::vertex<T> y; // y = hidden state
				sigmoid( (varx * W) + b, y);

				for(unsigned int j=0;j<x.size();j++)
					ta[j] += var[j]*xa[j];

				tb += y;
				tW += varx.outerproduct(y);

				auto wy = W*y;
				for(unsigned int j=0;j<z.size();j++){
					tz[j] += math::exp(-z[j])*T(0.5)*xa[j]*xa[j] - math::exp(-z[j]/T(2.0))*T(0.5)*x[j]*wy[j];
				}
			}

			ta /= T(NUM_SAMPLES);
			tb /= T(NUM_SAMPLES);
			tz /= T(NUM_SAMPLES);
			tW /= T(NUM_SAMPLES);


			auto ga = ta - da;
			auto gb = tb - db;
			auto gz = tz - dz;
			auto gW = tW - dW;

			if(learnVariance == false)
				gz.zero();

			// now we have gradients
			{
#if 1
				a = a + lrate*ga;
				b = b + lrate*gb;
				z = z + lrate*gz;
				W = W + lrate*gW;
#else
				// we test different learning rates and pick the one that gives smallest error
				math::vertex<T> a1 = a + T(1.0)*lrate*da;
				math::vertex<T> b1 = b + T(1.0)*lrate*db;
				math::vertex<T> z1 = z + T(1.0)*lrate*dz;
				math::matrix<T> W1 = W + T(1.0)*lrate*dW;
				T error1 = reconstruct_gbrbm_data_error(samples, 10, W1, a1, b1, z1, CDk);

				math::vertex<T> a2 = a + T(1.0/0.9)*lrate*da;
				math::vertex<T> b2 = b + T(1.0/0.9)*lrate*db;
				math::vertex<T> z2 = z + T(1.0/0.9)*lrate*dz;
				math::matrix<T> W2 = W + T(1.0/0.9)*lrate*dW;
				T error2 = reconstruct_gbrbm_data_error(samples, 10, W2, a2, b2, z2, CDk);

				math::vertex<T> a3 = a + T(0.9)*lrate*da;
				math::vertex<T> b3 = b + T(0.9)*lrate*db;
				math::vertex<T> z3 = z + T(0.9)*lrate*dz;
				math::matrix<T> W3 = W + T(0.9)*lrate*dW;
				T error3 = reconstruct_gbrbm_data_error(samples, 10, W3, a3, b3, z3, CDk);

				if(error1 <= error2 && error1 <= error3){
					a = a1;
					b = b1;
					z = z1;
					W = W1;
					lrate = T(1.0)*lrate;
				}
				else if(error2 <= error1 && error2 <= error3){
					a = a2;
					b = b2;
					z = z2;
					W = W2;
					lrate = T(1.0/0.9)*lrate;
				}
				else{
					a = a3;
					b = b3;
					z = z3;
					W = W3;
					lrate = T(0.9)*lrate;
				}
#endif
			}
		}

		{
			error = reconstruct_gbrbm_data_error(samples, 1000, W, a, b, z, CDk);
			auto pratio         = p_ratio(samples, random_samples);
			errors.push_back(math::exp(pratio));

			if(pratio < best_error){
				best_error = pratio;
				best_a = a;
				best_b = b;
				best_W = W;
				best_z = z;
			}

			// check for convergence
			if(errors.size() > 2){
				while(errors.size() > 15)
					errors.pop_front(); // only keeps 10 last epoch samples

				auto me = T(0.0);
				auto ve = T(0.0);

				for(auto& e : errors){
					me += e;
					ve += e*e;
				}

				me /= T(errors.size());
				ve /= T(errors.size());
				ve = ve - me*me;

				auto statistic = sqrt(ve)/me;

				if(statistic <= T(0.05)) // st.dev. is 5% of the mean
					convergence = true;
			}

			if(verbose){
				math::vertex<T> var;

				auto rerror = reconstruct_gbrbm_data_error(random_samples, 1000, W, a, b, z, CDk);
				auto r_ratio        = (error/rerror);
				//auto logp           = logProbability(samples);

				var = z;
				for(unsigned int i=0;i<var.size();i++) var[i] = math::exp(z[i]);

				std::cout << "EPOCH " << epoch <<
						". R: " << error <<
						" R-ratio: " << r_ratio <<
						//" log(P): " << logp <<
						" P-ratio: " << pratio <<
						" Variance: " << var << std::endl;
			}

		}

		epoch++;
	}


	a = best_a;
	b = best_b;
	W = best_W;
	z = best_z;

	// error = reconstruct_gbrbm_data_error(samples, samples.size(), W, a, b, z, CDk);
	// error = -logProbability(samples);

	{
		auto pratio       = p_ratio(samples, random_samples);

		error = -pratio;
	}


	return error;
}



// estimates log(P(samples|params)) of the RBM
template <typename T>
T GBRBM<T>::logProbability(const std::vector< math::vertex<T> >& samples)
{
	// calculates mean and variance from the samples
	if(samples.size() <= 1)
		return T(-INFINITY);

	math::vertex<T> m(samples[0].size()), s(samples[0].size());
	{
		m.zero();
		s.zero();

		for(auto& x : samples){
			m += x;
			for(unsigned int i=0;i<s.size();i++)
				s[i] += x[i]*x[i];
		}

		m /= T(samples.size());
		s /= T(samples.size());

		for(unsigned int i=0;i<s.size();i++)
			s[i] -= m[i]*m[i];

		s *= T(samples.size())/T(samples.size() - 1); // sample variance and not direct variance (divide by N-1 !!)
	}

	// calculates partition function Z
	T logZ = T(0.0);

	ais(logZ, m, s, W, a, b, z);

	T logP = T(0.0);

	// calculates unscaled log-probability of all samples
	for(auto& v : samples){
		T lp = unscaled_log_probability(v, W, a, b, z);
		logP += (lp - logZ)/T(samples.size());
	}

	return logP; // log(P(samples)) [geometric mean of P(v):s]
}


template <typename T>
bool GBRBM<T>::sample(const unsigned int SAMPLES, std::vector< math::vertex<T> >& samples,
		const std::vector< math::vertex<T> >& statistics_training_data)
{
	const auto& stats = statistics_training_data;

	if(stats.size() < 2)
		return false;

	math::vertex<T> m(stats[0].size()), s(stats[0].size());
	{
		m.zero();
		s.zero();

		for(auto& x : stats){
			m += x;
			for(unsigned int i=0;i<s.size();i++)
				s[i] += x[i]*x[i];
		}

		m /= T(stats.size());
		s /= T(stats.size());

		for(unsigned int i=0;i<s.size();i++)
			s[i] -= m[i]*m[i];

		s *= T(stats.size())/T(stats.size() - 1); // sample variance and not direct variance (divide by N-1 !!)
	}

	ais_sampling(samples, SAMPLES, m, s, W, a, b, z);
}


template <typename T>
T GBRBM<T>::reconstructError(const std::vector< math::vertex<T> >& samples)
{
	T error = reconstruct_gbrbm_data_error(samples, samples.size(), W, a, b, z, 1);
	return error;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

// set data points parameters for U(q) and Ugrad(q) calculations
template <typename T>
bool GBRBM<T>::setUData(const std::vector< math::vertex<T> >& samples)
{
	if(samples.size() < 2) return false;

	Usamples = samples;

	// calculates mean and variance of the samples
	Umean.resize(Usamples[0].size());
	Uvariance.resize(Usamples[0].size());

	Umean.zero();
	Uvariance.zero();

	for(auto& s : samples){
		Umean += s;
		for(unsigned int i=0;i<s.size();i++)
			Uvariance[i] += s[i]*s[i];
	}

	Umean /= T(samples.size());
	Uvariance /= T(samples.size());

	for(unsigned int i=0;i<Uvariance.size();i++){
		Uvariance[i] -= Umean[i]*Umean[i];
	}

	// converts to sample variance from the theoretical one (divide by (N-1))
	Uvariance *= T(samples.size())/T(samples.size() - 1);

	return true;
}

template <typename T>
bool GBRBM<T>::setUTemperature(const T temperature){ // sets temperature of the U(q) distribution.. As described in Cho et. al (2011) paper
	this->temperature = temperature;
}


template <typename T>
T GBRBM<T>::getUTemperature()
{
	return temperature;
}

template <typename T>
unsigned int GBRBM<T>::qsize() const throw() // size of q vector q = [a, b, z, vec(W)]
{
	return (a.size() + b.size() + z.size() + W.ysize()*W.xsize());
}


template <typename T>
bool GBRBM<T>::convertUParametersToQ(const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b,
    		const math::vertex<T>& z, math::vertex<T>& q) const
{
	q.resize(a.size()+b.size()+z.size()+W.ysize()*W.xsize());

	q.write_subvertex(a, 0);
	q.write_subvertex(b, a.size());
	q.write_subvertex(z, a.size()+b.size());
	W.save_to_vertex(q, a.size()+b.size()+z.size());

	return true;
}


// sets RBM machine's (W, a, b, z) parameters according to q vector
template <typename T>
bool GBRBM<T>::setParametersQ(const math::vertex<T>& q)
{
	try{
		q.subvertex(a, 0, a.size());
		q.subvertex(b, a.size(), b.size());
		q.subvertex(z, (a.size()+b.size()), z.size());
		math::vertex<T> w(W.ysize()*W.xsize());
		q.subvertex(w, (a.size()+b.size()+z.size()), w.size());
		W.load_from_vertex(w);

		return true;
	}
	catch(std::exception& e){
		return false;
	}
}




template <typename T>
T GBRBM<T>::U(const whiteice::math::vertex<T>& q) const throw() // calculates U(q) = -log(P(data|q))
{
	const unsigned int NUMUSAMPLES = 10; // 1000 seem to work rather well

	// FIXME this is incorrect as the calculation of U requires calculation of proper Z(v) and Z(theta)!

	try{
		// converts q to parameters [a, b, z, W]
		auto qa = a;
		auto qb = b;
		auto qz = z;
		auto qW = W;

		{
			q.subvertex(qa, 0, qa.size());
			q.subvertex(qb, qa.size(), qb.size());
			q.subvertex(qz, (qa.size()+qb.size()), qz.size());
			math::vertex<T> qw(qW.ysize()*qW.xsize());
			q.subvertex(qw, (qa.size()+qb.size()+qz.size()), qw.size());
			qW.load_from_vertex(qw);
		}

		// converts parameters to "temperized versions of themselves" (temperature E [0,1])
		qW = temperature*qW;
		qb = temperature*qb;

		for(unsigned int i=0;i<a.size();i++){
			qa[i] = temperature*qa[i] + (T(1.0) - temperature)*Umean[i];
			T v   = math::exp(qz[i]);
			v     = temperature*v + (T(1.0) - temperature)*Uvariance[i];
			qz[i] = math::log(v);
		}

		// calculates -log(P*(data|q)*p(q)) where P* is unscaled and p(q) is regularizer prior [not used]
		T u = T(0.0);

		for(unsigned int i=0;i<NUMUSAMPLES;i++){
			const unsigned int index = rand() % Usamples.size();
			auto& s = Usamples[index];
			u += -unscaled_log_probability(s, qW, qa, qb, qz);

		}

		u /= T(NUMUSAMPLES);

		// TODO: add some smart priors for the parameters:
		// 1. for qW we could use somekind of generalized Wishart matrix (not xx^t but xy^t)
		// 2. for qa gaussian zero mean may make sense if data is assumed always be preprocessed to have zero mean and variance
		// 3. for qb study binomial distributions and conjugates with p=0 and p=1 equally probable (beta?)
		// 4. for qz use e(qz) ~ chi-squared distribution?

		return (u);
	}
	catch(std::exception& e){
		std::cout << "ERROR: GBRBM::U: unexpected exception: " << e.what() << std::endl;

		T u = T(INFINITY); // error: zero probability: P = exp(-u) = exp(-INFINITY)
		return u;
	}
}


template <typename T>
T GBRBM<T>::Udiff(const math::vertex<T>& q1, const math::vertex<T>& q2) const
{
	const unsigned int NUMUSAMPLES = 10; // 1000 seem to work rather well

	try{
		// converts q to parameters [a, b, z, W]
		auto qa1 = a;
		auto qb1 = b;
		auto qz1 = z;
		auto qW1 = W;

		auto qa2 = a;
		auto qb2 = b;
		auto qz2 = z;
		auto qW2 = W;

		{
			q1.subvertex(qa1, 0, qa1.size());
			q1.subvertex(qb1, qa1.size(), qb1.size());
			q1.subvertex(qz1, (qa1.size()+qb1.size()), qz1.size());
			math::vertex<T> qw1(qW1.ysize()*qW1.xsize());
			q1.subvertex(qw1, (qa1.size()+qb1.size()+qz1.size()), qw1.size());
			qW1.load_from_vertex(qw1);

			q2.subvertex(qa2, 0, qa2.size());
			q2.subvertex(qb2, qa2.size(), qb2.size());
			q2.subvertex(qz2, (qa2.size()+qb2.size()), qz2.size());
			math::vertex<T> qw2(qW2.ysize()*qW2.xsize());
			q2.subvertex(qw2, (qa2.size()+qb2.size()+qz2.size()), qw2.size());
			qW2.load_from_vertex(qw2);
		}

		// converts parameters to "temperized versions of themselves" (temperature E [0,1])
		{
			qW1 = temperature*qW1;
			qb1 = temperature*qb1;

			for(unsigned int i=0;i<a.size();i++){
				qa1[i] = temperature*qa1[i] + (T(1.0) - temperature)*Umean[i];
				T v   = math::exp(qz1[i]);
				v     = temperature*v + (T(1.0) - temperature)*Uvariance[i];
				qz1[i] = math::log(v);
			}

			qW2 = temperature*qW2;
			qb2 = temperature*qb2;

			for(unsigned int i=0;i<a.size();i++){
				qa2[i] = temperature*qa2[i] + (T(1.0) - temperature)*Umean[i];
				T v   = math::exp(qz2[i]);
				v     = temperature*v + (T(1.0) - temperature)*Uvariance[i];
				qz2[i] = math::log(v);
			}
		}

		// back converts temperized versions of parameters into q-vectors (used to calculate negative phase inner product..)
		auto q11 = q1;
		auto q22 = q2;
		{
			q11.write_subvertex(qa1, 0);
			q11.write_subvertex(qb1, qa1.size());
			q11.write_subvertex(qz1, qa1.size()+qb1.size());
			qW1.save_to_vertex(q11, qa1.size()+qb1.size()+qz1.size());

			q22.write_subvertex(qa2, 0);
			q22.write_subvertex(qb2, qa2.size());
			q22.write_subvertex(qz2, qa2.size()+qb2.size());
			qW2.save_to_vertex(q22, qa2.size()+qb2.size()+qz2.size());
		}

		// calculates "negative phase" term
		// auto Ev1 = negative_phase_q(NUMUSAMPLES, qW1, qa1, qb1, qz1); // Ev[grad(F)]
		// auto Ev2 = negative_phase_q(NUMUSAMPLES, qW2, qa2, qb2, qz2);
		// auto aprox_logZratio = T(-0.5)*((q11 - q22)*(Ev1 + Ev2))[0];

		// auto aprox_logZratio = T(0.0); // assumes ratio to be ~ zero..
		//
		// T logZ1 = T(0.0);
		// T logZ2 = T(0.0);
		//
		// ais(logZ1, Umean, Uvariance, qW1, qa1, qb1, qz1);
		// ais(logZ2, Umean, Uvariance, qW2, qa2, qb2, qz2);
		//
		// aprox_logZratio = logZ1 - logZ2;
		//

		auto aprox_logZratio =
				math::log(zratio(Umean, Uvariance, qW1, qa1, qb1, qz1, qW2, qa2, qb2, qz2));

		// calculates -log(P*(data|q)*p(q)) where P* is unscaled (without Z) and p(q) is regularizer prior [not used]
		T u = T(0.0);

		for(unsigned int i=0;i<NUMUSAMPLES;i++){
			const unsigned int index = rand() % Usamples.size();
			auto& s = Usamples[index];

			// free-energy
			auto F1 = -unscaled_log_probability(s, qW1, qa1, qb1, qz1);
			auto F2 = -unscaled_log_probability(s, qW2, qa2, qb2, qz2);

			u += (F1 - F2) + aprox_logZratio;
		}

		u /= T(NUMUSAMPLES);

		// TODO: add some smart priors for the parameters:
		// 1. for qW we could use somekind of generalized Wishart matrix (not xx^t but xy^t)
		// 2. for qa gaussian zero mean may make sense if data is assumed always be preprocessed to have zero mean and variance
		// 3. for qb study binomial distributions and conjugates with p=0 and p=1 equally probable (beta?)
		// 4. for qz use e(qz) ~ chi-squared distribution?

		return (u);
	}
	catch(std::exception& e){
		std::cout << "ERROR: GBRBM::U: unexpected exception: " << e.what() << std::endl;

		T u = T(INFINITY); // error: zero probability: P = exp(-u) = exp(-INFINITY)
		return u;
	}
}


template <typename T>
whiteice::math::vertex<T> GBRBM<T>::Ugrad(const whiteice::math::vertex<T>& q) throw() // calculates grad(U(q))
{
	const unsigned int NUMUSAMPLES = 10; // 1000 seem to work rather well..
	whiteice::math::vertex<T> grad(this->qsize());
	grad.zero();

	try{
		// converts q to parameters [a, b, z, W]
		auto qa = a;
		auto qb = b;
		auto qz = z;
		auto qW = W;

		{
			q.subvertex(qa, 0, qa.size());
			q.subvertex(qb, qa.size(), qb.size());
			q.subvertex(qz, (qa.size()+qb.size()), qz.size());
			math::vertex<T> qw(qW.ysize()*qW.xsize());
			q.subvertex(qw, (qa.size()+qb.size()+qz.size()), qw.size());
			qW.load_from_vertex(qw);
		}

		// converts parameters to "temperized versions of themselves" (temperature E [0,1])
		qW = temperature*qW;
		qb = temperature*qb;

		for(unsigned int i=0;i<a.size();i++){
			qa[i] = temperature*qa[i] + (T(1.0) - temperature)*Umean[i];
			T v   = math::exp(qz[i]);
			v     = temperature*v + (T(1.0) - temperature)*Uvariance[i];
			qz[i] = math::log(v);
		}

		// calculates gradients for the data
		math::vertex<T> ga(a.size());
		math::vertex<T> gb(b.size());
		math::vertex<T> gz(z.size());
		math::matrix<T> gW(W.ysize(), W.xsize());

		ga.zero();
		gb.zero();
		gz.zero();
		gW.zero();

		// grad(U)    = SUM(gradF) - N*Emodel[gradF]
		// grad_a(F)  = -S^-1 (v-a)
		// grad_b(F)  = -h
		// grad_W(F)  = -h(S^-0.5 * v)^T
		// grad_zi(F) = -e(-z[i])*[ 0.5*(v_i - a_i)^2 - 0.5*e(+z[i]/2)*v[i]*(W*h)[i] ]
		// h          = sigmoid(W^t * S^-0.5 * v + b)

		math::vertex<T> invS(qa.size()); // S^-1
		math::vertex<T> invShalf(qa.size()); // S^-0.5
		for(unsigned int i=0;i<qa.size();i++){
			invS[i] = math::exp(-qz[i]);
			invShalf[i] = math::exp(-qz[i]/2);
		}

		// for(auto& v : Usamples){
		for(unsigned int ui=0;ui<NUMUSAMPLES;ui++)
		{
			auto& v = Usamples[rand()%Usamples.size()];
			// calculates positive phase SUM(gradF)

			math::vertex<T> grad_a = (v-qa);
			math::vertex<T> sv(v.size());

			for(unsigned int i=0;i<qa.size();i++){
				grad_a[i] *= invS[i];          // S^-1 (v-a);
				sv[i]      = invShalf[i]*v[i]; // S^-0.5*v
			}

			math::vertex<T> h(qb.size());
			sigmoid(sv*qW + qb, h);            // calculates h

			math::vertex<T> grad_b = h;

			math::matrix<T> grad_W = sv.outerproduct(h);

			math::vertex<T> grad_z(qz.size());

			auto qWh = qW*h;

			for(unsigned int i=0;i<qz.size();i++){
				grad_z[i] = math::exp(-qz[i])*T(0.5)*(v[i]-qa[i])*(v[i]-qa[i]) - T(0.5)*math::exp(-qz[i]/2)*v[i]*qWh[i];
			}

			ga += grad_a;
			gb += grad_b;
			gz += grad_z;
			gW += grad_W;
		}

		ga /= T(NUMUSAMPLES);
		gb /= T(NUMUSAMPLES);
		gz /= T(NUMUSAMPLES);
		gW /= T(NUMUSAMPLES);


		// calculates negative phase N*Emodel[gradF], N = Usamples.size()

		// FIXME actually calculate mean and variance of the estimate Emodel[gradF] and get new samples until sample variance is "small"
		const unsigned int NEGSAMPLES = NUMUSAMPLES;
		{
			std::vector< math::vertex<T> > vs; // negative particles [samples from Pmodel(v)]

			// TODO use AIS to get samples from the model
			// ais_sampling(vs, NEGSAMPLES, Umean, Uvariance, qa, qb, qz, qW);

			// uses CD-1 to get samples [fast]
			for(unsigned int s=0;s<NEGSAMPLES;s++){
				const unsigned int index = rand() % Usamples.size();
				const math::vertex<T>& v = Usamples[index]; // x = visible state

				auto xx = reconstruct_gbrbm_data(v, qW, qa, qb, qz, 1); // gets x ~ p(v) from the model
				vs.push_back(xx);
			}

			const T scaling = T((double)1.0)/T((double)NEGSAMPLES);

			for(auto& v : vs){
				// calculates negative phase N*Emodel[gradF] = N/SAMPLES * SUM( gradF(v_i) )

				math::vertex<T> grad_a = (v-qa);
				math::vertex<T> sv(v.size());

				for(unsigned int i=0;i<qa.size();i++){
					grad_a[i] *= invS[i];          // S^-1 (v-a);
					sv[i]      = invShalf[i]*v[i]; // S^-0.5*v
				}

				math::vertex<T> h(qb.size());
				sigmoid(sv*qW + qb, h);            // calculates h

				math::vertex<T> grad_b = h;

				math::matrix<T> grad_W = sv.outerproduct(h);

				math::vertex<T> grad_z(qz.size());

				auto qWh = qW*h;

				for(unsigned int i=0;i<qz.size();i++){
					grad_z[i] = math::exp(-qz[i])*T(0.5)*(v[i]-qa[i])*(v[i]-qa[i]) - T(0.5)*math::exp(-qz[i]/2)*v[i]*qWh[i];
				}

				// scales the values according to the number of samples

				grad_a *= scaling;
				grad_b *= scaling;
				grad_z *= scaling;
				grad_W *= scaling;

				// this is negative phase so we minus point-wise gradients from the sum variables
				ga -= grad_a;
				gb -= grad_b;
				gz -= grad_z;
				gW -= grad_W;
			}
		}

		// sets variance term to zero... [disables variance learning]
		// gz.zero();

		// converts component gradients [ga,gb,gz,gW] to back to q gradient vector grad(q)

		{
			grad.resize(ga.size()+gb.size()+gz.size()+gW.ysize()*gW.xsize());

			grad.write_subvertex(ga, 0);
			grad.write_subvertex(gb, ga.size());
			grad.write_subvertex(gz, ga.size()+gb.size());
			gW.save_to_vertex(grad, ga.size()+gb.size()+gz.size());
		}

		return (grad);
	}
	catch(std::exception& e){
		std::cout << "ERROR: GBRBM::Ugrad: unexpected exception: " << e.what() << std::endl;
		return (-grad); // zero gradient
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

// load & saves RBM data from/to file
template <typename T>
bool GBRBM<T>::load(const std::string& filename) throw()
{
	return false; // not implemented (yet)
}

template <typename T>
bool GBRBM<T>::save(const std::string& filename) const throw()
{
	return false; // not implemented (yet)
}


//////////////////////////////////////////////////////////////////////////////////////////////////


template <typename T>
T GBRBM<T>::normalrnd() const // N(0,1)
{
	// TODO should protect access using mutex
	return T( (*rng)(*generator) );
}


template <typename T>
math::vertex<T> GBRBM<T>::normalrnd(const math::vertex<T>& m, const math::vertex<T>& v) const
{
	// TODO should protect access using mutex
	math::vertex<T> x(m);

	for(unsigned int i=0;i<x.size();i++)
		x[i] += normalrnd()*math::sqrt(v[i]);

	return x;
}

template <typename T>
void GBRBM<T>::sigmoid(const math::vertex<T>& input, math::vertex<T>& output) const
{
	output.resize(input.size());

	for(unsigned int i=0;i<input.size();i++){
		output[i] = T(1.0)/(T(1.0) + math::exp(-input[i]));
	}
}


// estimates ratio of Z values of unscaled p(v|params) distributions: Z1/Z2 using AIS Monte Carlo sampling.
// this is needed by Udiff() which calculates difference of two P(params|v) distributions..
template <typename T>
T GBRBM<T>::zratio(const math::vertex<T>& m, const math::vertex<T>& s, // data mean and variance used by the AIS sampler
		const math::matrix<T>& W1, const math::vertex<T>& a1, const math::vertex<T>& b1, math::vertex<T>& z1,
		const math::matrix<T>& W2, const math::vertex<T>& a2, const math::vertex<T>& b2, math::vertex<T>& z2) const
{
	std::vector<T> r; // calculated ratios [used to estimate convergence by calculation mean st.dev.]

	int iter = 0;

	while(1){ // repeat until estimate of ratio has converged to have small enough sample variance..
		// uses AIS sampler to get samples from Z2 distribution
		std::vector< math::vertex<T> > vs;
		const unsigned int STEPSIZE=500;

	    ais_sampling(vs, STEPSIZE, m, s, W2, a2, b2, z2);

	    const unsigned int index0 = r.size();
	    r.resize(r.size() + STEPSIZE);

#pragma omp parallel for
	    for(unsigned int index=0;index<STEPSIZE;index++){
	    	auto& v = vs[index];
	        // free-energy
			auto F1 = -unscaled_log_probability(v, W1, a1, b1, z1);
			auto F2 = -unscaled_log_probability(v, W2, a2, b2, z2);

			r[index+index0] = math::exp(F2-F1); // calculates ratio of unscaled probability functions
	    }

	    // calculates mean and estimates how big is it sample variance and waits until it is small enough
	    // Var[1/N SUM(x_i)] = 1/N^2 N*Var[x_i] = Var[x]/N = S^2 / N, where S^2 is sample variance ~ E[X^2] - E[X]^2

	    T mr = T(0.0);
	    T vr = T(0.0);

	    for(auto& s : r){
	    	mr += s;
	    	vr += s*s;
	    }

	    mr /= T(r.size());
	    vr /= T(r.size());
	    vr -= mr*mr;
	    vr *= T((double)r.size()/((double)r.size() - 1.0)); // changes division to 1/N-1 (sample variance)

	    auto sv = math::sqrt(vr / T(r.size())); // st.dev. of the mean

	    if(sv/mr <= T(1.0)){
	    	// mean has maximum of 100% error (relative to mean value)
	    	return mr;
	    }
	    else{
	    	std::cout << "mr = " << mr << " ";
	    	std::cout << "zratio continuing sampling. iteration " << iter
	    			<< " : " << (sv/mr) << std::endl;

	    	if(iter >= 5)
	    		return mr; // maximum of 5 iterations
	    }

	    iter++;
	}
}


// calculates Ev[grad_q(F)]
template <typename T>
math::vertex<T> GBRBM<T>::negative_phase_q(const unsigned int SAMPLES,
		const math::matrix<T>& qW, const math::vertex<T>& qa, const math::vertex<T>& qb, const math::vertex<T>& qz) const
{
	math::vertex<T> grad(qW.ysize()*qW.xsize()+qa.size()+qb.size()+qz.size());
	grad.zero();

	auto ga = qa;
	auto gb = qb;
	auto gz = qz;
	auto gW = qW;

	ga.zero();
	gb.zero();
	gz.zero();
	gW.zero();

	math::vertex<T> invS(qa.size()); // S^-1
	math::vertex<T> invShalf(qa.size()); // S^-0.5
	for(unsigned int i=0;i<qa.size();i++){
		invS[i] = math::exp(-qz[i]);
		invShalf[i] = math::exp(-qz[i]/2);
	}

	// FIXME actually calculate mean and variance of the estimate Emodel[gradF] and get new samples until sample variance is "small"
	{
		std::vector< math::vertex<T> > vs; // negative particles [samples from Pmodel(v)]

		// TODO use AIS to get samples from the model
		// ais_sampling(vs, NEGSAMPLES, Umean, Uvariance, qa, qb, qz, qW);

		// uses CD-1 to get samples [fast]
		for(unsigned int s=0;s<SAMPLES;s++){
			const unsigned int index = rand() % Usamples.size();
			const math::vertex<T>& v = Usamples[index]; // x = visible state

			auto xx = reconstruct_gbrbm_data(v, qW, qa, qb, qz, 1); // gets x ~ p(v) from the model
			vs.push_back(xx);
		}

		const T scaling = T((double)1.0)/T((double)SAMPLES);

		for(auto& v : vs){
			// calculates negative phase N*Emodel[gradF] = N/SAMPLES * SUM( gradF(v_i) )

			math::vertex<T> grad_a = (v-qa);
			math::vertex<T> sv(v.size());

			for(unsigned int i=0;i<qa.size();i++){
				grad_a[i] *= invS[i];          // S^-1 (v-a);
				sv[i]      = invShalf[i]*v[i]; // S^-0.5*v
			}

			math::vertex<T> h(qb.size());
			sigmoid(sv*qW + qb, h);            // calculates h

			math::vertex<T> grad_b = h;

			math::matrix<T> grad_W = sv.outerproduct(h);

			math::vertex<T> grad_z(qz.size());

			auto qWh = qW*h;

			for(unsigned int i=0;i<qz.size();i++){
				grad_z[i] = math::exp(-qz[i])*T(0.5)*(v[i]-qa[i])*(v[i]-qa[i]) - T(0.5)*math::exp(-qz[i]/2)*v[i]*qWh[i];
			}

			// scales the values according to the number of samples

			grad_a *= scaling;
			grad_b *= scaling;
			grad_z *= scaling;
			grad_W *= scaling;

			ga += grad_a;
			gb += grad_b;
			gz += grad_z;
			gW += grad_W;
		}
	}


	// converts component gradients [ga,gb,gz,gW] to back to q gradient vector grad(q)

	{
		grad.resize(ga.size()+gb.size()+gz.size()+gW.ysize()*gW.xsize());

		grad.write_subvertex(ga, 0);
		grad.write_subvertex(gb, ga.size());
		grad.write_subvertex(gz, ga.size()+gb.size());
		gW.save_to_vertex(grad, ga.size()+gb.size()+gz.size());
	}

	return (-grad);
}


// generates SAMPLES {v,h}-samples from p(v,h|params) using AIS
template <typename T>
void GBRBM<T>::ais_sampling(std::vector< math::vertex<T> >& vs, const unsigned int SAMPLES,
		const math::vertex<T>& m, const math::vertex<T>& s,
		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z) const
{
	std::vector< math::vertex<T> > v, h;

	std::vector< GBRBM<T> > ais_rbm;
	const unsigned int NTemp = 100; // number of different temperatures (values below <100, or below 10 do not work very well)..

	math::vertex<T> vz(z.size());
	for(unsigned int i=0;i<vz.size();i++)
		vz[i] = math::exp(z[i]); // sigma^2

	if(ais_rbm.size() != NTemp){
		ais_rbm.resize(NTemp);
		for(unsigned int i=0;i<NTemp;i++)
			ais_rbm[i].resize(a.size(), b.size());
	}

	for(unsigned int i=0;i<NTemp;i++){
		const T beta = T(i/((double)(NTemp - 1)));
		ais_rbm[i].setParameters(beta*W,  beta*a + (T(1.0)-beta)*m, beta*b, beta*vz + (T(1.0)-beta)*s);
	}


	std::vector<T> logRs;
	unsigned int iter = 0;

	{
		vs.resize(SAMPLES);

#pragma omp parallel for
		for(unsigned int i=0;i<SAMPLES;i++){
			math::vertex<T> vv;

#pragma omp critical
			{
				vv = normalrnd(m, s); // generates N(m,s) distributed variable [level 0]
			}

			math::vertex<T> hh;

			for(unsigned int j=1;j<(NTemp-1);j++){
				// T(v,v') transition operation
				ais_rbm[j].sampleHidden(hh, vv);
				ais_rbm[j].sampleVisible(vv, hh);
			}

			{
				vs[i] = vv;
			}
		}

	}

}

// calculates log(P-ratio): log(P(data1)/P(data2)) using geometric mean of probabilities
template <typename T>
T GBRBM<T>::p_ratio(const std::vector< math::vertex<T> >& data1, const std::vector< math::vertex<T> >& data2)
{
	// log(R) = log(P(data1)) - log(P(data2))

	T logPdata1 = T(0.0);
	T logPdata2 = T(0.0);

	if(data1.size() <= 0 ||  data2.size() <= 0)
		return T(0.0);

	for(const auto& v : data1)
		logPdata1 += unscaled_log_probability(v);

	logPdata1 /= T(data1.size());

	for(const auto& v : data2)
		logPdata2 += unscaled_log_probability(v);

	logPdata2 /= T(data2.size());

	return (logPdata1 - logPdata2);
}


// estimates partition function Z and samples v ~ p(v) for a given GBRBM(W,a,b,z)
// using Parallel Tempering Annihilated Importance Sampling
template <typename T>
T GBRBM<T>::ais(T& logZ,
		const math::vertex<T>& m, const math::vertex<T>& s,
	    const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z) const
{
	// parallel tempering RBM stack
	std::vector< GBRBM<T> > rbm;
	const unsigned int NTemp = 1000; // number of different temperatures (10.000 might be a good choice)

	math::vertex<T> vz(z.size());
	for(unsigned int i=0;i<vz.size();i++)
		vz[i] = math::exp(z[i]); // sigma^2

	rbm.resize(NTemp);
	for(unsigned int i=0;i<NTemp;i++){
		const T beta = T(i/((double)(NTemp - 1)));
		rbm[i].resize(this->getVisibleNodes(), this->getHiddenNodes());
		rbm[i].setParameters(beta*W,  beta*a + (T(1.0)-beta)*m, beta*b, beta*vz + (T(1.0)-beta)*s);
	}

	const unsigned int SAMPLES = 1000; // base number of samples

	std::vector<T> logRs;
	unsigned int iter = 0;

	do{

		for(unsigned int i=0;i<SAMPLES;i++){
			auto vv = normalrnd(m, s); // generates N(m,s) distributed variable [level 0]
			math::vertex<T> h;

			T logR = rbm[1].unscaled_log_probability(vv) - rbm[0].unscaled_log_probability(vv);

			for(unsigned int j=1;j<(NTemp-1);j++){
				// T(v,v') transition operation
				rbm[j].sampleHidden(h, vv);
				rbm[j].sampleVisible(vv, h);

				logR += rbm[(j+1)].unscaled_log_probability(vv) - rbm[j].unscaled_log_probability(vv);
			}

			// v.push_back(vv); // do not store samples?
			logRs.push_back(logR);
		}

		// calculates mean because only it is guaranteed by theory to give correct values
		// also causes expected error of logR by calculating sample variance and calculates its percentage of the mean (var/mean)
		T logR = T(0.0);
		T varR = T(0.0);
		for(const auto& lr : logRs){
			auto r = math::exp(lr);
			logR +=  r / T(logRs.size());
			varR += (r * r) / T(logRs.size());
		}

		varR = varR - logR*logR;
		varR /= T(logRs.size()); // sample variance

		auto pError = T(100.0)*math::sqrt(varR)/logR;

		std::cout << iter << ": Expected error in R (Z) (%): " << pError << std::endl;
		iter++;

		if(pError < 5.0)
			break; // only stop until statistics tell we have 5% error
	}
	while(1);

#if 0
	// calculates final logR (mean value) [we use geometric mean because it is easiest to calculate]
	T logR = T(0.0);
	for(const auto& lr : logRs)
		logR += lr / T(logRs.size());
#endif
	// calculates mean because only it is guaranteed by theory to give correct values
	// also causes expected error of logR by calculating sample variance and calculates its percentage of the mean (var/mean)
	T logR = T(0.0);
	T varR = T(0.0);
	for(const auto& lr : logRs){
		auto r = math::exp(lr);
		logR +=  r / T(logRs.size());
		varR += (r * r) / T(logRs.size());
	}

	varR = varR - logR*logR;
	varR /= T(logRs.size()); // sample variance

	std::cout << "Expected error in R (Z) (%): " << T(100.0)*math::sqrt(varR)/logR << std::endl;

	logR = math::log(logR); // transforms back to logarithms


	// logR = log(Z) - log(Z0), Z0 = Z of N(m,s) distribution
	T NlogZ = T(m.size()/2.0)*math::log(T(2.0)*M_PI);

	for(unsigned int i=0;i<m.size();i++)
		NlogZ += T(0.5)*z[i];

	logZ = logR + NlogZ;
	// logZ = NlogZ - logR;

	return logZ;
}


template <typename T>
T GBRBM<T>::unscaled_log_probability(const math::vertex<T>& v) const
{
	return unscaled_log_probability(v, W, a, b, z);
}


template <typename T>
T GBRBM<T>::unscaled_log_probability(const math::vertex<T>& v,
    const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z) const
{
	auto va = v - a;
	auto sv(v);

	T sva = T(0.0);

	math::vertex<T> s(z.size());
	for(unsigned int i=0;i<s.size();i++){
		s[i] = math::exp(-z[i]);
		sv[i] *= math::sqrt(s[i]);
		sva += va[i]*va[i]*s[i];
	}

	auto alpha = sv*W + b;

	T sum = T(0.0);
	for(unsigned int i=0;i<b.size();i++){
		const T exp = math::exp(alpha[i]);
		double t = 0.0;
		whiteice::math::convert(t, exp);

		if(std::isinf(t)){
			sum += alpha[i]; // approximate very large terms: log(1+x) = log(x)
		}
		else{
			sum += math::log(T(1.0) + exp);
		}
	}

	T result = T(-0.5)*sva + sum;

	return result;
}


// calculates mean energy of the samples
template <typename T>
T GBRBM<T>::meanEnergy(const std::vector< math::vertex<T> >& samples,
		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z)
{
	math::vertex<T> d;
	math::vertex<T> dd;
	d.resize(z.size());
	dd.resize(z.size());

	for(unsigned int i=0;i<z.size();i++){
		d[i]  = math::exp(-z[i]);
		dd[i] = math::sqrt(math::exp(z[i]));
	}

	T e = T(0.0);

	for(auto& s : samples){
		math::vertex<T> h;
		sampleHidden(h, s);

		e += E(s, h, W, a, b, z);
	}

	e /= T(samples.size());

	return e;
}

// calculates energy function E(v,h|params)
template <typename T>
T GBRBM<T>::E(const math::vertex<T>& v, const math::vertex<T>& h,
		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z) const
{
	T e = T(0.0);

	math::matrix<T> C, Chalf;
	C.resize(z.size(),z.size());
	Chalf.resize(z.size(),z.size());
	C.zero();
	Chalf.zero();
	for(unsigned int i=0;i<z.size();i++){
		C(i,i) = math::exp(-z[i]);
		Chalf(i,i) = math::sqrt(C(i,i));
	}

	e = (T(0.5)*(v-a)*C*(v-a) - v*Chalf*W*h - b*h)[0];

	return e;
}




template <typename T>
math::vertex<T> GBRBM<T>::reconstruct_gbrbm_data(const math::vertex<T>& v,
		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z,
		unsigned int CDk) const
{
	auto x = v;

	math::vertex<T> d;
	math::vertex<T> dd;
	d.resize(z.size());
	dd.resize(z.size());

	for(unsigned int i=0;i<z.size();i++){
		d[i]  = math::exp(-z[i]/T(2.0));
		dd[i] = math::sqrt(math::exp(z[i]));
	}

	for(unsigned int l=0;l<CDk;l++){
		for(unsigned int i=0;i<x.size();i++)
			x[i] = d[i]*x[i];

		auto hx = x*W + b;

		math::vertex<T> h;

		sigmoid(hx, h);

		for(unsigned int i=0;i<h.size();i++){
			T r = T( rand()/((double)RAND_MAX) );
			if(r <= h[i]) h[i] = T(1.0);
			else h[i] = T(0.0);
		}

		auto mean = W*h; // pseudo-mean

		for(unsigned int i=0;i<mean.size();i++){
			x[i] = dd[i]*normalrnd() + dd[i]*mean[i] + a[i];
		}
	}

	return x;
}


template <typename T>
math::vertex<T> GBRBM<T>::reconstruct_gbrbm_hidden(const math::vertex<T>& v,
		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z,
		unsigned int CDk)
{
	auto x = v;

	math::vertex<T> d;
	math::vertex<T> dd;
	d.resize(z.size());
	dd.resize(z.size());

	for(unsigned int i=0;i<z.size();i++){
		d[i]  = math::exp(-z[i]/T(2.0));
		dd[i] = math::sqrt(math::exp(z[i]));
	}

	math::vertex<T> h;

	for(unsigned int l=0;l<CDk;l++){
		for(unsigned int i=0;i<x.size();i++)
			x[i] = d[i]*x[i];

		auto hx = x*W + b;

		sigmoid(hx, h);

		for(unsigned int i=0;i<h.size();i++){
			T r = T( rand()/((double)RAND_MAX) );
			if(r <= h[i]) h[i] = T(1.0);
			else h[i] = T(0.0);
		}

		auto mean = W*h;

		for(unsigned int i=0;i<mean.size();i++){
			x[i] = dd[i]*normalrnd() + dd[i]*mean[i] + a[i];
		}
	}

	return h;
}


template <typename T>
T GBRBM<T>::reconstruct_gbrbm_data_error(const std::vector< math::vertex<T> >& samples, unsigned int N,
		const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z,
		unsigned int CDk)
{
	T error = T(0.0);

	if(samples.size() <= 0)
		return error;

	for(unsigned int n=0;n<N;n++){
		const unsigned int index = rand() % samples.size();
		auto& s = samples[index];

		auto delta = s - reconstruct_gbrbm_data(s, W, a, b, z, CDk);
		auto nrm   = delta.norm();
		nrm = math::pow(nrm, T(2.0));

		error += nrm/delta.size();
	}

	error /= N;

	return error;
}


template class GBRBM< float >;
template class GBRBM< double >;
template class GBRBM< math::blas_real<float> >;
template class GBRBM< math::blas_real<double> >;



} /* namespace whiteice */
