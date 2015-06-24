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

    initializeWeights();
}

template <typename T>
GBRBM<T>::~GBRBM()
{
	delete generator;
	delete rng;
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

	math::vertex<T> data_mean(z.size());
	math::vertex<T> data_var(z.size());
	std::vector< math::vertex<T> > random_samples; // generates random N(data_mean, datavar*I) data to test against

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

	errors.push_back(error);

	if(verbose){
		math::vertex<T> var;

		T rerror = reconstruct_gbrbm_data_error(random_samples, 1000, W, a, b, z, CDk);
		auto r_ratio        = (error/rerror);
		auto pratio         = p_ratio(samples, random_samples);
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
			const unsigned int index = rand() % samples.size();

			math::vertex<T> x = samples[index]; // x = visible state

			std::vector< math::vertex<T> > vs;
#if 1
			pt_sampling(vs, a.size() > b.size() ? a.size() : b.size(),
					data_mean, data_var); // gets (v) from the model
#else
			auto xx = reconstruct_gbrbm_data(x, W, a, b, z, CDk); // gets x ~ p(v) from the model
			vs.push_back(xx);
#endif
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

#if 0
			x = reconstruct_gbrbm_data(x, W, a, b, z, CDk); // gets x ~ p(v) from the model

			math::vertex<T> var(z.size());
			math::vertex<T> varx(z.size());

			for(unsigned int j=0;j<z.size();j++){
				var[j] = math::exp(-z[j]);
				varx[j] = var[j]*x[j];
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

			for(unsigned int j=0;j<z.size();j++)
				gz[j] = var[j]*(T(0.5)*xa[j]*xa[j] - x[j]*wy[j]);
#endif

			// calculates gradients [positive phase]
			x = samples[index];
			auto xa = (x - a);

			for(unsigned int j=0;j<var.size();j++)
				varx[j] = math::exp(-z[j]/T(2.0))*x[j];

			math::vertex<T> y; // y = hidden state
			sigmoid( (varx * W) + b, y);

			for(unsigned int j=0;j<x.size();j++)
				da[j] = var[j]*xa[j] - da[j];

			db = y - db;
			dW = varx.outerproduct(y) - dW;

			auto wy = W*y;
			for(unsigned int j=0;j<z.size();j++){
				dz[j] = math::exp(-z[j])*T(0.5)*xa[j]*xa[j] - math::exp(-z[j]/T(2.0))*T(0.5)*x[j]*wy[j] - dz[j];
				// dz[j] = var[j]*(T(0.5)*xa[j]*xa[j] - x[j]*wy[j]) - dz[j];
			}

			if(learnVariance == false)
				dz.zero();


			// now we have gradients
			{
#if 1
				a = a + lrate*da;
				b = b + lrate*db;
				z = z + lrate*dz;
				W = W + lrate*dW;
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

				std::cout << "CONVERGENCE: " << statistic << std::endl;

				if(statistic <= T(0.5)) // st.dev. is 5% of the mean
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
	T logZ;

	ais(logZ, m, s, W, a, b, z);

	T logP = T(0.0);

	// calculates unscaled log-probability of all samples
	for(auto& v : samples){
		T lp = unscaled_log_probability(v, W, a, b, z);
		logP += (lp - logZ)/T(samples.size());
	}

	return logP; // log(P(samples)) [geometric mean of P(v):s]
}


////////////////////////////////////////////////////////////

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
T GBRBM<T>::normalrnd() // N(0,1)
{
	return T( (*rng)(*generator) );
}


template <typename T>
math::vertex<T> GBRBM<T>::normalrnd(const math::vertex<T>& m, const math::vertex<T>& s)
{
	math::vertex<T> x(m);

	for(unsigned int i=0;i<x.size();i++)
		x[i] += normalrnd()*math::sqrt(s[i]);

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


// generates SAMPLES {v,h}-samples from p(v,h|params) using AIS
template <typename T>
void GBRBM<T>::pt_sampling(std::vector< math::vertex<T> >& vs, const unsigned int SAMPLES,
		const math::vertex<T>& m, const math::vertex<T>& s)
{
	// parallel tempering RBM stack
	std::vector< GBRBM<T> > rbm;
	std::vector< math::vertex<T> > v, h;

	const unsigned int NTemp = 100; // number of different temperatures

	math::vertex<T> vz(z.size());
	for(unsigned int i=0;i<vz.size();i++)
		vz[i] = math::exp(z[i]); // sigma^2

	rbm.resize(NTemp);
	for(unsigned int i=0;i<NTemp;i++){
		const T beta = T(i/((double)(NTemp - 1)));
		rbm[i].resize(this->getVisibleNodes(), this->getHiddenNodes());
		rbm[i].setParameters(beta*W,  beta*a + (T(1.0)-beta)*m, beta*b, beta*vz + (T(1.0)-beta)*s);
	}


	std::vector<T> logRs;
	unsigned int iter = 0;

	{

		for(unsigned int i=0;i<SAMPLES;i++){
			auto vv = normalrnd(m, s); // generates N(m,s) distributed variable [level 0]
			math::vertex<T> hh;

			for(unsigned int j=1;j<(NTemp-1);j++){
				// T(v,v') transition operation
				rbm[j].sampleHidden(hh, vv);
				rbm[j].sampleVisible(vv, hh);
			}

			vs.push_back(vv);
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
	    const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b, const math::vertex<T>& z)
{
	// parallel tempering RBM stack
	std::vector< GBRBM<T> > rbm;
	const unsigned int NTemp = 10000; // number of different temperatures

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
	for(unsigned int i=0;i<b.size();i++)
		sum += math::log(T(1.0) + math::exp(alpha[i]));

	return T(-0.5)*sva + sum;
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
