/*
 * HMM.cpp
 *
 *  Created on: 5.7.2015
 *      Author: Tomas
 */

#include "HMM.h"
#include <vector>
#include <list>

using namespace whiteice::math;

namespace whiteice {

  HMM::HMM(unsigned int visStates, unsigned int hidStates) throw(std::logic_error) :
    numVisible(visStates), numHidden(hidStates)
  {
    if(numVisible == 0 || numHidden == 0)
      throw std::logic_error("whiteice::HMM ctor - number of visible or hidden states cannot be zero");
    
    precision = 128; // 128 bits [2 x double]
    
    ph.resize(numHidden);
    
    for(auto& i : ph){
      i.setPrecision(precision);
      i = 0.0;
    }
    
    A.resize(numHidden);
    for(auto& a : A){
      a.resize(numHidden);
      for(auto& aa : a){
	aa.setPrecision(precision);
	aa = 0.0;
      }
    }
    
    B.resize(numHidden);
    for(auto& b : B){
      b.resize(numHidden);
      for(auto& c : b){
	c.resize(numVisible);
	for(auto& d: c){
	  d.setPrecision(precision);
	  d = 0.0;
	}
      }
    }
    
  }
  
  HMM::~HMM()
  {
    // nothing to do
  }
  

  /**
   * Sets arbitrary precision number's
   * precision for calculations
   * TODO set good default value so finetuning is rarely needed
   */
  bool HMM::setPrecision(unsigned int prec)
  {
    this->precision = prec;
    
    ph.resize(numHidden);
    
    for(auto& i : ph){
      i.setPrecision(precision);
    }
    
    A.resize(numHidden);
    for(auto& a : A){
      a.resize(numHidden);
      for(auto& aa : a){
	aa.setPrecision(precision);
      }
    }
    
    B.resize(numHidden);
    for(auto& b : B){
      b.resize(numHidden);
      for(auto& c : b){
	c.resize(numVisible);
	for(auto& d: c){
	  d.setPrecision(precision);
	}
      }
    }
    
    return true;
  }
  

  /**
   * Sets ph, A, and B to random (initial) values before optimization
   */
  void HMM::randomize()
  {
    realnumber sum(0.0, precision);

    // pi
    for(auto& p : ph){
      p = rng.uniform();
      sum += p;
    }

    for(auto& p : ph)
      p /= sum;

    
    // A
    for(unsigned int i=0;i<numHidden;i++){
      sum = 0.0;

      for(unsigned int j=0;j<numHidden;j++){
	A[i][j] = rng.uniform();
	sum += A[i][j];
      }

      for(unsigned int j=0;j<numHidden;j++){
	A[i][j] /= sum;
      }
    }

    // B
    for(unsigned int i=0;i<numHidden;i++){
      for(unsigned int j=0;j<numHidden;j++){
	sum = 0.0;
	
	for(unsigned int k=0;k<numVisible;k++){
	  B[i][j][k] = rng.uniform();
	  sum += B[i][j][k];
	}

	for(unsigned int k=0;k<numVisible;k++){
	  B[i][j][k] /= sum;
	}

      }
    }
    
  }
  
  
  /**
   * trains HMM parameters from discrete observational states
   * (unsigned integers are state numbers) using
   * Expectation Maximization (EM) algorithm
   *
   * returns log(probability) of training data
   */
  double HMM::train(const std::vector<unsigned int>& observations) throw (std::invalid_argument)
  {
    // uses Baum-Welch algorithm
    
    bool converged = false;
    std::list<realnumber> pdata;
    
    realnumber plast(0.0, precision);
    
    
    while(!converged) // keeps calculating EM-algorithm for parameter estimation
    {
	
      // first calculates alpha and beta
      const unsigned int T = observations.size();
      set::vector< std::vector<realnumber> > alpha(T+1), beta(T+1);
      
      for(auto& a : alpha){
	a.resize(numHidden);
	for(auto& ai : a)
	  ai.setPrecision(precision);
	a = ph;
      }
      
      // forward procedure (alpha)
      for(unsigned int t=1;t<=T;t++){
	auto& a  = alpha[t-1];
	auto& an = alpha[t];
	auto& o  = observations[t-1];
	
	for(unsigned int j=0;j<an.size();j++){
	  an[j] = 0.0;
	  for(unsigned int i=0;i<a.size();i++){
	    
	    if(o >= B[i][j].size())
	      throw std::invalid_argument("HMM::train() - alpha calculations: observed state out of range");
	    
	    an[j] += a[i]*A[i][j]*B[i][j][o];
	  }
	}
      }
      
      
      for(auto& b : beta){
	b.resize(numHidden);
	for(auto& bi : b){
	  bi.setPrecision(precision);
	  bi = 1.0;
	}
      }
      
      // backward procedure (beta)
      for(unsigned int t=(T+1);t>=1;t--){
	auto& b  = beta[t];
	auto& bp = beta[t-1];
	auto& o  = observations[t-1];
	
	for(unsigned int i=0;i<numHidden;i++){
	  bp[i] = 0.0;
	  for(unsigned int j=0;j<numHidden;j++){
	    
	    if(o >= B[i][j].size())
	      throw std::invalid_argument("HMM::train() - beta calculations: observed state out of range");
	    
	    bp[i] += A[i][j] * B[i][j][o] * b[j];
	  }
	}
      }
      
      // now we have both alpha and beta and we calculate p
      std::vector< std::vector < std::vector<realnumber> > > p(T);
      
      for(auto& pij : p){
	pij.resize(numHidden);
	for(auto& pj : pij){
	  pj.resize(numHidden);
	  for(auto& v: pj){
	    v.setPrecision(precision);
	    v = 0.0;
	  }
	}
      }
      
      
      for(unsigned int t=1;t<=T;t++){
	realnumber ab(0.0, precision);
	
	// divisor
	for(unsigned int m=0;m<numHidden;m++)
	  ab += alpha[t-1][m]*beta[t-1][m];
	
	auto& pt = p[t-1];
	auto& o  = observations[t-1];
	
	for(unsigned int i=0;i<numHidden;i++){
	  for(unsigned int j=0;j<numHidden;j++){
	    
	    if(o >= B[i][j].size())
	      throw std::invalid_argument("HMM::train() - p(i,j) calculations: observed state out of range");
	    
	    pt[i][j] = alpha[t-1][i] * A[i][j] * B[i][j][o] * beta[t][j];
	    pt[i][j] /= ab;
	  }
	}
      }
      
      
      // now we have p[t][i][j] and we calculate y[t][i]
      std::vector< std::vector<realnumber> > y(T);
      
      for(auto& yt : y){
	yt.resize(numHidden);
	for(auto& yti : yt){
	  yti.setPrecision(precision);
	  yti = 0.0;
	}
      }
      
      for(unsigned unsigned int t=1;t<=T;t++){
	for(unsigned int i=0;i<numHidden;i++){
	  auto& yti = y[t-1][i];
	  yti = 0.0;
	  
	  for(unsigned int j=0;j<numHidden;j++)
	    yti += p[t-1][i][j];
	}
      }
      
      //////////////////////////////////////////////////////////////////////
      // now we can calculate new parameter values based on EM
      
      // pi
      for(unsigned int i=0;i<numHidden;i++){
	const unsigned int t = 1;
	ph[i] = y[t-1][i];
      }
      
      // state transitions A[i][j]
      for(unsigned int i=0;i<numHidden;i++){
	for(unsigned int j=0;j<numHidden;j++){
	  
	  realnumber sp(0.0, precision);
	  realnumber sy(0.0, precision);
	  
	  for(unsigned int t=1;t<=T;t++){
	    sp += p[t-1][i][j];
	    sy += y[t-1][t];
	  }
	  
	  A[i][j] = sp/sy;
	}
      }
      
      // visible state probabilities B[i][j][k]
      for(unsigned int i=0;i<numHidden;i++){
	for(unsigned int j=0;j<numHidden;j++){
	  for(unsigned int k=0;k<numVisible;k++){
	    realnumber spk(0.0, precision);
	    realnumber sp(0.0, precision);
	    
	    for(unsigned int t=1;t<=T;t++){
	      if(observations[t-1] == k) spk += p[t-1][i][j];
	      sp += p[t-1][i][j];
	    }
	    
	    B[i][j][k] = spk/sp;
	  }
	}
      }
      
      
      // now we have new parameters: A, B, ph
      // still calculates probability of observations [using previous parameter values]
      // as E[p(o)] = p(observations)**(1/length(observations)) is used to measure convergence
      
      realnumber po(0.0, precision);
      for(unsigned int i=0;i<numHidden;i++){
	const unsigned int t = T+1;
	po += alpha[t-1][i];
      }
      po = pow(po, 1.0/observations.size());
      
      plast = po;
      pdata.push_back(po);
      
      
      std::cout << "Observations log(probability): " << log(po).getDouble() << std::endl;
      
      // estimates convergence
      {
	if(pdata.size() < 10)
	  continue; // needs at least 10 data points
	
	while(pdata.size() > 20)
	  pdata.pop_front();
	
	// calculates mean and st.dev. and decides for convergence if st.dev/mean <= 0.05
	// (works for positive values)
	
	realnumber m(0.0, precision);
	realnumber s(0.0, precision);
	
	for(auto& p : pdata){
	  m += p/pdata.size();
	  s += (p*p)/pdata.size();
	}
	
	s -= m*m;
	s = sqrt(s);
	
	auto r = s/m;
	
	if(r.toDouble() <= 0.05){
	  converged = true;
	}
	
      }
      
    }
    
    
    if(plast > 0.0)
      plast = log(plast);
    
    return plast.getDouble();
  }
  

  
  /**
   * samples given length observation stream from HMM
   */
  bool HMM::sample(const unsigned int numberOfObservations,
		   std::vector<unsigned int>& observations) const
  {
    observations.clear();
    
    unsigned int h = sample(ph);
    
    // h is initial hidden state sampled from p(h)
    for(unsigned int i=0;i<numberOfObservations;i++){
      auto hprev = h;
      h = sample(A[h]); // samples new h from state transition distribution A[i] (i->j)
      
      // emits observation/visible state k with probability B[hprev][h][k]
      unsigned int k = sample(B[hprev][h]);
      
      observations.push_back(k);
    }
    
    return true;
  }

  
  /**
   * helper function: samples 1d discrete distribution
   * according to probability p(k) returns k (index) of variable chosen
   */
  unsigned int HMM::sample(const std::vector<realnumber>& p) const
  {
    double u = rng.uniform();
    realnumber cp(0.0, precision);
    
    for(unsigned int h=0;h<p.size();h++){
      cp += p[h];
      
      if(u <= cp)
	return h;
    }
    
    return (p.size()-1);
  }
  

  /**
   * finds maximum likelihood hidden states describing observations
   * as well as possible by using Viterbi algorithm:
   * max(h) p(v|h)
   *
   * returns log(probability) of the optimum hidden states
   */
  double HMM::ml_states(std::vector<unsigned int>& hidden,
			const std::vector<unsigned int>& observations) const throw (std::invalid_argument)
  {
    std::vector<realnumber> d(ph);
    std::vector<realnumber> dnext(ph);
    
    realnumber phidden(0.0, precision);
    
    for(unsigned int t=1;t<=(T+1);t++){
      for(unsigned int j=0;j<numHidden;j++){
	realnumber max(0.0, precision);
	unsigned int max_i = 0;
	auto& o = observations[t-1];

	if(o >= B[hh][h].size())
	  throw std::invalid_argument("HMM::ml_states() - observed state out of range");

	for(unsigned int i=0;i<numHidden;i++){
	  auto t = d[i] * A[i][j] * B[i][j][o];

	  // NOTE: this does not properly handle ties so the first best path found is used.
	  
	  if(t > max){
	    max = t;
	    max_i = i;
	  }
	}
	
	dnext[j] = max;
	phidden  = max;
	hidden.push_back(max_i);
      }
    }

    auto logp = log(phidden);
    
    return logp.getDouble();
  }



  /*
   * calculations log(probability) of observations
   */
  double HMM::logprobability(std::vector<unsigned int>& observations) const throw (std::invalid_argument)
  {
    // uses forward procedure
    
    std::vector<realnumber> alpha(ph);
    std::vector<realnumber> anext(ph);
    
    const unsigned int T = observations.size();
    
    for(unsigned int t=1;t<=(T+1);t++){
      for(unsigned int h=0;h<numHidden;h++){
	anext[h] = 0.0f;
	for(unsigned int hh=0;hh<numHidden;hh++){
	  auto& o = observations[t-1];
	  
	  if(o >= B[hh][h].size())
	    throw std::invalid_argument("HMM::logprobability() - observed state out of range");
	  
	  anext[h] += alpha[hh]*A[hh][h]*B[hh][h][o];
	}
      }
      
      alpha = anext;
    }
    
    realnumber po(0.0, precision);
    
    for(auto& a : alpha)
      po += a;
    
    po = log(po);
    
    return po.getDouble();
  }



} /* namespace whiteice */
