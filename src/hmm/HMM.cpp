/*
 * HMM.cpp
 *
 *  Created on: 5.7.2015
 *      Author: Tomas
 */

#include "HMM.h"

#include <vector>
#include <list>
#include <set>

#include "dataset.h"



using namespace whiteice::math;

namespace whiteice {

  HMM::HMM(unsigned int visStates, unsigned int hidStates) throw(std::logic_error)
  {
    this->numVisible = visStates;
    this->numHidden  = hidStates;

    if(numVisible == 0 || numHidden == 0)
      throw std::logic_error("whiteice::HMM ctor - number of visible or hidden states cannot be zero");
    
    precision = 256; // 128 bits [2 x double]
    
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

    this->randomize();
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

    normalize_parameters();
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
    unsigned int iteration = 0;
    realnumber plast(0.0, precision);

    {
      printf("ITER %d. Log(probability) = %f\n", iteration, logprobability(observations));
    }

    
    while(!converged) // keeps calculating EM-algorithm for parameter estimation
    {
	
      // first calculates alpha and beta
      const unsigned int T = observations.size();
      std::vector< std::vector<realnumber> > alpha(T+1), beta(T+2);
      
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
	    
	    if(o >= B[i][j].size()){
	      char buffer[128];
	      sprintf(buffer,
		      "HMM::train() - alpha calculations: observed state out range (%d)", o);
	      throw std::invalid_argument(buffer);
	    }
	    
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
	    
	    if(o >= B[i][j].size()){
	      char buffer[128];
	      sprintf(buffer,
		      "HMM::train() - beta calculations: observed state out range (%d)", o);
	      throw std::invalid_argument(buffer);
	    }

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

	realnumber zero(0.0, precision);

	if(ab == zero){
	  throw std::invalid_argument("HMM::train() - out of floating point precision\n");
	}
	
	auto& pt = p[t-1];
	auto& o  = observations[t-1];
	
	for(unsigned int i=0;i<numHidden;i++){
	  for(unsigned int j=0;j<numHidden;j++){
	    
	    if(o >= B[i][j].size()){
	      char buffer[128];
	      sprintf(buffer,
		      "HMM::train() - p(i,j) calculations: observed state out range (%d)", o);
	      throw std::invalid_argument(buffer);
	    }
	    
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
      
      for(unsigned int t=1;t<=T;t++){
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
	    sy += y[t-1][i];
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
      po = pow(po, 1.0/((double)observations.size()));
      
      plast = po;
      pdata.push_back(po);
      
      iteration++;
      
      printf("ITER %d. Log(probability) = %f\n", iteration, log(po).getDouble());
      
      
      // estimates convergence
      {
	if(pdata.size() < 10)
	  continue; // needs at least 10 data points
	
	while(pdata.size() > 30)
	  pdata.pop_front();
	
	// calculates mean and st.dev. and decides for convergence if st.dev/mean <= 0.05
	// (works for positive values)
	
	realnumber m(0.0, precision);
	realnumber s(0.0, precision);
	
	for(auto& p : pdata){
	  m += p;
	  s += (p*p);
	}

	m /= (double)pdata.size();
	s /= (double)pdata.size();

	s -= m*m;

	s = abs(s);
	s = sqrt(s);
	
	auto r = s/m;
	
	if(r.getDouble() <= 0.00001){
	  converged = true;
	}
	
      }
      
    }

    normalize_parameters();
    
    
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

    const unsigned int T = observations.size()-1;
    
    for(unsigned int t=1;t<=(T+1);t++){
      for(unsigned int j=0;j<numHidden;j++){
	realnumber max(0.0, precision);
	unsigned int max_i = 0;
	auto& o = observations[t-1];

	for(unsigned int i=0;i<numHidden;i++){
	  
	  if(o >= B[i][j].size())
	    throw std::invalid_argument("HMM::ml_states() - observed state out of range");
	  
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




  double HMM::next_state(const unsigned int currentState,
			 unsigned int& nextState,
			 const unsigned int observation) const
    throw (std::invalid_argument)
  {
    if(currentState > ph.size()){
      throw std::invalid_argument("HMM::next_state() - currentState out of bounds");
    }
    
    std::vector<realnumber> dnext(numHidden);
    realnumber sum(0.0, precision);

    for(unsigned int j=0;j<numHidden;j++){
      const auto& o = observation;
      const unsigned int i = currentState;
	  
      if(o >= B[i][j].size())
	throw std::invalid_argument("HMM::next_state() - observed state out of range");
      realnumber t(0.0, precision);
      t = A[i][j] * B[i][j][o];

      dnext[j].setPrecision(precision);
      dnext[j] = t;
      sum += t;
    }

    for(unsigned int j=0;j<numHidden;j++){
      dnext[j] /= sum;
    }

    nextState = sample(dnext);

    auto logp = log(dnext[nextState]);
    
    return logp.getDouble();
  }

  



  /*
   * calculations log(probability) of observations
   */
  double HMM::logprobability(const std::vector<unsigned int>& observations) const throw (std::invalid_argument)
  {
    // uses forward procedure
    
    std::vector<realnumber> alpha(ph);
    std::vector<realnumber> anext(ph);
    
    const unsigned int T = observations.size();
    
    for(unsigned int t=1;t<=(T);t++){
      for(unsigned int h=0;h<numHidden;h++){
	anext[h] = 0.0f;
	for(unsigned int hh=0;hh<numHidden;hh++){
	  auto& o = observations[t-1];
	  
	  if(o >= B[hh][h].size()){
	    char buffer[128];
	    sprintf(buffer,
		    "HMM::logprobability() - observed state out of range (%d)", o);
	    throw std::invalid_argument(buffer);
	  }
	  
	  anext[h] += alpha[hh]*A[hh][h]*B[hh][h][o];
	}
      }
      
      alpha = anext;
    }
    
    realnumber po(0.0, precision);
    
    for(auto& a : alpha)
      po += a;
    
    po = pow(po, 1.0/((double)observations.size()));
    
    po = log(po);
    
    return po.getDouble();
  }

#define HMM_VERSION_CFGSTR "HMM_VERSION"
#define HMM_ARCH_CFGSTR    "HMM_ARCH"
#define HMM_PI_CFGSTR      "HMM_PARAM_PI"
#define HMM_PARAM_A_CFGSTR "HMM_PARAM_A"
#define HMM_PARAM_B_CFGSTR "HMM_PARAM_B"

  bool HMM::load(const std::string& filename) throw()
  {
    try{
      whiteice::dataset<double> configuration;
      math::vertex<double> data;

      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      
      if(configuration.load(filename) == false)
	return false;

      int versionid = 0;
      
      // checks version
      {
	data = configuration.accessName(HMM_VERSION_CFGSTR, 0);
	ints.resize(data.size());
	for(unsigned int i=0;i<data.size();i++){
	  math::convert(ints[i], data[i]);
	}
	
	if(ints.size() != 1)
	  return false;
	
	versionid = ints[0];
	
	ints.clear();
      } 
      
      if(versionid != 1000) // v1.0 datafile
	return false;

      // gets architecture
      std::vector<unsigned int> arch;
      
      {
	data = configuration.accessName(HMM_ARCH_CFGSTR, 0);
	ints.resize(data.size());
	for(unsigned int i=0;i<data.size();i++){
	  math::convert(ints[i], data[i]);
	}
	
	if(ints.size() < 3)
	  return false;
	
	arch.resize(ints.size());
	
	for(unsigned int i=0;i<ints.size();i++){
	  if(ints[i] <= 0) return false;
	  arch[i] = (unsigned int)ints[i];
	}
      }

      // gets PI parameter
      std::vector<double> PI; 
      
      {
	data = configuration.accessName(HMM_PI_CFGSTR, 0);
	
	if(data.size() != arch[1])
	  return false; // arch[1] == numHidden states

	PI.resize(arch[1]);
	for(unsigned int i=0;i<arch[1];i++)
	  PI[i] = data[i];
      }

      // gets A parameter
      std::vector< std::vector< double > > loadA;

      {
	data = configuration.accessName(HMM_PARAM_A_CFGSTR, 0);

	if(data.size() != arch[1]*arch[1])
	  return false;

	unsigned int index = 0;

	loadA.resize(arch[1]);
	for(unsigned int j=0;j<arch[1];j++){
	  loadA[j].resize(arch[1]);
	  for(unsigned int i=0;i<arch[1];i++,index++){
	    loadA[j][i] = data[index];
	  }
	}
	  
      }

      // gets B parameter
      std::vector< std::vector< std::vector< double > > > loadB;
      
      {
	data = configuration.accessName(HMM_PARAM_B_CFGSTR, 0);

	if(data.size() != arch[1]*arch[1]*arch[0])
	  return false;

	unsigned int index = 0;

	loadB.resize(arch[1]);
	for(unsigned int j=0;j<arch[1];j++){
	  loadB[j].resize(arch[1]);
	  for( unsigned int i=0;i<arch[1];i++){
	    loadB[j][i].resize(arch[0]);
	    for(unsigned int k=0;k<arch[0];k++, index++){
	      loadB[j][i][k] = data[index];
	    }
	  }
	}
      }


      // now we have all parameters, copies them to actual parameters
      numVisible = arch[0];
      numHidden  = arch[1];
      precision  = arch[2];

      ph.resize(numHidden);
      for(unsigned int i=0;i<ph.size();i++){
	ph[i].setPrecision(precision);
	ph[i] = PI[i];
      }

      A.resize(numHidden);
      for(unsigned int j=0;j<numHidden;j++){       
	A[j].resize(numHidden);
	for(unsigned int i=0;i<numHidden;i++){
	  A[j][i].setPrecision(precision);
	  A[j][i] = loadA[j][i];
	}
      }

      B.resize(numHidden);
      for(unsigned int j=0;j<numHidden;j++){
	B[j].resize(numHidden);
	for(unsigned int i=0;i<numHidden;i++){
	  B[j][i].resize(numVisible);
	  for(unsigned int k=0;k<numVisible;k++){
	    B[j][i][k].setPrecision(precision);
	    B[j][i][k] = loadB[j][i][k];
	  }
	}
      }
      
      
      return true;
      
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      return false;
    }
  }

  
  bool HMM::save(const std::string& filename) const throw()
  {
    try{
      whiteice::dataset<double> configuration;
      math::vertex<double> data;
      
      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      
      // writes version information
      {
	ints.push_back(1000);
	
	configuration.createCluster(HMM_VERSION_CFGSTR, ints.size());
	data.resize(ints.size());
	for(unsigned int i=0;i<ints.size();i++)
	  data[i] = ints[i];
	
	configuration.add(configuration.getCluster(HMM_VERSION_CFGSTR), data);
	
	ints.clear();
      }
      
      
      // writes architecture information
      {
	ints.push_back(numVisible);
	ints.push_back(numHidden);
	ints.push_back(precision);
	
	configuration.createCluster(HMM_ARCH_CFGSTR, ints.size());
	data.resize(ints.size());
	for(unsigned int i=0;i<ints.size();i++)
	  data[i] = ints[i];
	
	configuration.add(configuration.getCluster(HMM_ARCH_CFGSTR), data);
	
	ints.clear();
      }
      
      // writes parameter PI
      {
	configuration.createCluster(HMM_PI_CFGSTR, ph.size());
	data.resize(ph.size());
	for(unsigned int i=0;i<ph.size();i++)
	  data[i] = ph[i].getDouble();
	
	configuration.add(configuration.getCluster(HMM_PI_CFGSTR), data);
      }
      
      // writes parameter vec(A)
      {
	const unsigned int size = numHidden*numHidden;
	
	configuration.createCluster(HMM_PARAM_A_CFGSTR, size);
	data.resize(size);
	unsigned int index = 0;
	for(unsigned int j=0;j<numHidden;j++)
	  for(unsigned int i=0;i<numHidden;i++, index++)
	    data[index] = A[j][i].getDouble();
	
	configuration.add(configuration.getCluster(HMM_PARAM_A_CFGSTR), data);
      }
      
      // writes parameter vec(B)
      {
	const unsigned int size = numHidden*numHidden*numVisible;
	
	configuration.createCluster(HMM_PARAM_B_CFGSTR, size);
	data.resize(size);
	unsigned int index = 0;
	for(unsigned int j=0;j<numHidden;j++)
	  for(unsigned int i=0;i<numHidden;i++)
	    for(unsigned int k=0;k<numVisible;k++,index++)
	      data[index] = B[j][i][k].getDouble();
	
	configuration.add(configuration.getCluster(HMM_PARAM_B_CFGSTR), data);
      }

      return configuration.save(filename);
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      
      return false;
    }
  }
    

  // normalizes parameters by ordering hidden states according to probabilities
  void HMM::normalize_parameters()
  {
    auto newA = A;
    auto newP = ph;
    auto newB = B;

    // uses hidden state transitions to order states
    std::vector<unsigned int> order;
    std::set<unsigned int> states;

    for(unsigned int i=0;i<numHidden;i++)
      states.insert(i);

    while(states.size() > 0){
      realnumber max = 0.0;
      unsigned int state = 0;

      for(auto s : states){
	if(A[s][s] > max){
	  max = A[s][s];
	  state = s;
	}
      }

      states.erase(states.find(state));
      order.push_back(state);
    }

    // PI
    for(unsigned int i=0;i<order.size();i++){
      newP[i] = ph[order[i]];
    }

    // A
    for(unsigned int i=0;i<order.size();i++){
      for(unsigned int j=0;j<numHidden;j++){
	for(unsigned int i=0;i<numHidden;i++){
	  newA[j][i] = A[order[j]][order[i]];
	}
      }
    }

    // B
    for(unsigned int i=0;i<order.size();i++){
      for(unsigned int j=0;j<numHidden;j++){
	for(unsigned int i=0;i<numHidden;i++){
	  for(unsigned int k=0;k<numVisible;k++){
	    newB[j][i][k] = B[order[j]][order[i]][k];
	  }
	}
      }
    }
    

    ph = newP;
    A = newA;
    B = newB;
    
  }
  

} /* namespace whiteice */
