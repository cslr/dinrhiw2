
#include "RNN_RBM.h"
#include "matrix.h"
#include "vertex.h"
#include "dataset.h"
#include "bayesian_nnetwork.h"

#include <stdio.h>


namespace whiteice
{

  
  template <typename T>
  RNN_RBM<T>::RNN_RBM(unsigned int dimVisible,
		      unsigned int dimHidden,
		      unsigned int dimRecurrent)
  {
    this->dimVisible = dimVisible;
    this->dimHidden  = dimHidden;
    this->dimRecurrent = dimRecurrent;

    if(dimVisible == 0 || dimHidden == 0 || dimRecurrent == 0)
      throw std::invalid_argument("RNN_RBM ctor: zero dimension not possible");

    // initializes neural network
    {
      // NN inputs are visible elements v and recurrent elements
      // NN outputs are two RBM bias elements and recurrent elements
      std::vector<unsigned int> arch;
      arch.push_back(dimVisible + dimRecurrent);
      arch.push_back(dimVisible + dimHidden + dimRecurrent);
      arch.push_back(dimVisible + dimHidden + dimRecurrent);

      nn.setArchitecture(arch, whiteice::nnetwork<T>::halfLinear);
      
      nn.randomize();
    }

    // initializes BBRBM
    {      
      rbm.resize(dimVisible, dimHidden);
      
      rbm.initializeWeights();
    }

    // sets synthesization variables
    {
      synthIsInitialized = false;
      vprev.resize(dimVisible);
      rprev.resize(dimRecurrent);
      
      vprev.zero();
      rprev.zero();
    }
    
  }
  

  template <typename T>
  RNN_RBM<T>::RNN_RBM(const whiteice::RNN_RBM<T>& rbm)
  {
    this->dimVisible = rbm.dimVisible;
    this->dimHidden  = rbm.dimHidden;
    this->dimRecurrent = rbm.dimRecurrent;

    this->nn = rbm.nn;
    this->rbm = rbm.rbm;

    this->synthIsInitialized = rbm.synthIsInitialized;
    this->vprev = rbm.vprev;
    this->rprev = rbm.rprev;
  }
  
  
  template <typename T>
  RNN_RBM<T>::~RNN_RBM()
  {
    synthIsInitialized = false;
  }
  

  template <typename T>
  RNN_RBM<T>& RNN_RBM<T>::operator=(const whiteice::RNN_RBM<T>& rbm)
  {
    this->dimVisible = rbm.dimVisible;
    this->dimHidden  = rbm.dimHidden;
    this->dimRecurrent = rbm.dimRecurrent;

    this->nn = rbm.nn;
    this->rbm = rbm.rbm;

    this->synthIsInitialized = rbm.synthIsInitialized;
    this->vprev = rbm.vprev;
    this->rprev = rbm.rprev;

    return (*this);
  }


  template <typename T>
  unsigned int RNN_RBM<T>::getVisibleDimensions() const
  {
    return dimVisible;
  }

  
  template <typename T>
  unsigned int RNN_RBM<T>::getHiddenDimensions() const
  {
    return dimHidden;
  }


  template <typename T>
  unsigned int RNN_RBM<T>::getRecurrentDimensions() const
  {
    return dimRecurrent;
  }


  template <typename T>
  const whiteice::nnetwork<T>& RNN_RBM<T>::getRNN() const
  {
    return nn;
  }


  template <typename T>
  const whiteice::BBRBM<T>& RNN_RBM<T>::getRBM() const
  {
    return rbm;
  }
  
  
  // optimizes data likelihood using N-timseries,
  // which are i step long and have dimVisible elements e
  // timeseries[N][i][e]
  template <typename T>
  bool RNN_RBM<T>::optimize(const std::vector< std::vector< whiteice::math::vertex<T> > >& timeseries)
  {
    // some checks for input validity
    if(timeseries.size() <= 0) return false;
    if(timeseries[0].size() <= 0) return false;
    if(timeseries[0][0].size() != dimVisible) return false;
    
    bool running = true;
    const unsigned int CDk = 1;
    T epsilon = T(0.01); // step length
    
    unsigned int iterations = 0;
    std::list<T> errors;

    
    while(running){

      whiteice::math::matrix<T> grad_W(dimHidden, dimVisible);
      whiteice::math::vertex<T> grad_w(nn.exportdatasize());
      unsigned int numgradients = 0;

      grad_W.zero();
      grad_w.zero();
      

      for(unsigned int n=0;n<timeseries.size();n++){

	whiteice::math::vertex<T> r(dimRecurrent);
	r.zero();
	
	whiteice::math::vertex<T> v(dimVisible);
	v.zero();

	whiteice::math::matrix<T> ugrad(dimVisible + dimHidden + dimRecurrent,
					nn.exportdatasize());
	ugrad.zero();

	for(unsigned int i=0;i<timeseries[n].size();i++){
	  whiteice::math::vertex<T> input(dimVisible + dimRecurrent);
	  input.write_subvertex(v, 0);
	  input.write_subvertex(r, dimVisible);

	  whiteice::math::vertex<T> output;
	  
	  nn.calculate(input, output);

	  whiteice::math::vertex<T> a(dimVisible), b(dimHidden);

	  output.subvertex(a, 0, dimVisible);
	  output.subvertex(b, dimVisible, dimHidden);

	  v = timeseries[n][i]; // visible element

	  whiteice::math::vertex<T> vstar(dimVisible), hstar(dimHidden);
	  whiteice::math::vertex<T> h(dimHidden);

	  // uses CD-k to calculate v* and h* (CD-k estimates) and h response to v
	  {
	    rbm.setBValue(b);
	    rbm.setAValue(a);

	    rbm.setVisible(v);
	    rbm.reconstructData(1);
	    rbm.getHidden(h);
	    
	    rbm.setVisible(v);
	    rbm.reconstructData(2*CDk);

	    rbm.getVisible(vstar);
	    rbm.getHidden(hstar);
	  }
	  

	  // calculates error gradients of recurrent neural network
	  {
	    // du(n)/dw = df/dw + df/dr * Gr * du(n-1)/dw, Gr matrix selects r
	    
	    whiteice::math::matrix<T> fgrad_w;
	    
	    nn.gradient(input, fgrad_w);

	    whiteice::math::matrix<T> fgrad_input;

	    nn.gradient_value(input, fgrad_input);

	    whiteice::math::matrix<T> fgrad_r(nn.output_size(), dimRecurrent);

	    fgrad_input.submatrix(fgrad_r,
				  dimVisible+dimHidden, 0,
				  dimRecurrent, nn.output_size());

	    whiteice::math::matrix<T> ugrad_r(dimRecurrent, nn.exportdatasize());

	    ugrad.submatrix(ugrad_r,
			    0, dimVisible+dimHidden,
			    nn.exportdatasize(), dimRecurrent);
	    
	    ugrad = fgrad_w + fgrad_r * ugrad_r;
	  }

	  
	  // calculates gradients of log(probability)
	  {
	    // calculates rbm W weights gradient: h*v^T - E[h*v^T]
	    // TODO optimize computations
	    auto gW = (h.outerproduct(v) - hstar.outerproduct(vstar)); 

	    // calculates RNN weights gradient:
	    // dlog(p)/dw = dlog(p)/da * da/dw + dlog(p)/db * db/dw

	    // da/dw
	    whiteice::math::matrix<T> ugrad_a(dimVisible, nn.exportdatasize());

	    ugrad.submatrix(ugrad_a,
			    0, 0,
			    nn.exportdatasize(), dimVisible);

	    // db/dw
	    whiteice::math::matrix<T> ugrad_b(dimHidden, nn.exportdatasize());

	    ugrad.submatrix(ugrad_b,
			    0, dimVisible,
			    nn.exportdatasize(), dimHidden);

	    auto dlogp_a = v - vstar;
	    auto dlogp_b = h - hstar;

	    auto dlogp_w = dlogp_a * ugrad_a + dlogp_b * ugrad_b;

	    grad_W += gW;
	    grad_w += dlogp_w;
	    
	    numgradients++;
	  }
	  
	  output.subvertex(r, dimVisible + dimHidden, dimRecurrent);	  
	}

      }


      // after we have calculated gradients through time-series
      // updates parameters can computes reconstruction error and reports it
      if(numgradients > 0)
      {
	grad_W /= T(numgradients);
	grad_w /= T(numgradients);

	T error = T(0.0);

	{
	  whiteice::math::matrix<T> W;
	  whiteice::math::vertex<T> w;

	  whiteice::BBRBM<T> rbm1(rbm), rbm2(rbm), rbm3(rbm);
	  whiteice::nnetwork<T> nn1(nn), nn2(nn), nn3(nn);

	  W = rbm.getWeights();
	  W += epsilon*grad_W;
	  rbm1.setWeights(W);

	  W = rbm.getWeights();
	  W += T(0.90)*epsilon*grad_W;
	  rbm2.setWeights(W);

	  W = rbm.getWeights();
	  W += T(1.0/0.90)*epsilon*grad_W;
	  rbm3.setWeights(W);
	  
	  nn.exportdata(w);
	  w += epsilon*grad_w;
	  nn1.importdata(w);

	  nn.exportdata(w);
	  w += T(0.90)*epsilon*grad_w;
	  nn2.importdata(w);

	  nn.exportdata(w);
	  w += T(1.0/0.9)*epsilon*grad_w;
	  nn3.importdata(w);
	  
	  T error1 = reconstructionError(rbm1, nn1, timeseries);
	  T error2 = reconstructionError(rbm2, nn2, timeseries);
	  T error3 = reconstructionError(rbm3, nn3, timeseries);

	  if(error1 <= error2 && error1 <= error3){
	    error = error1;
	  }
	  else if(error2 <= error1 && error2 <= error3){
	    error = error2;
	    epsilon = T(0.90)*epsilon;
	  }
	  else{ // error3 is the smallest
	    error = error3;
	    epsilon = T(1.0/0.90)*epsilon;
	  }

	  
	  W = rbm.getWeights();
	  W += epsilon*grad_W;
	  rbm.setWeights(W);
	}

	printf("RNN_RBM ITER %d RECONSTRUCTION ERROR: %f EPSILON: %f\n",
	       iterations, error.c[0], epsilon.c[0]);
	fflush(stdout);

	errors.push_back(error);
	
      }
      else{
	return false; // something is wrong..
      }

      
      // checks for stopping criteria
      // (error standard deviation is 1% of mean value)
      {
	while(errors.size() > 20)
	  errors.pop_front();

	if(errors.size() >= 10){
	  T me = T(0.0), se = T(0.0);
	  
	  for(const auto& e : errors){
	    me += abs(e) / T(errors.size());
	    se += e*e / T(errors.size());
	  }

	  se = sqrt(abs(se - me*me)); // st.dev.

	  T ratio = se / (me + T(10e-10));

	  printf("STOPPING CRITERIA RATIO: %f\n", ratio.c[0]);
	  fflush(stdout);

	  if(ratio <= T(0.001)*epsilon){
	    running = false; // stop computation
	  }
	  
	}
      }
      
      iterations++;
    }

    
    return (iterations > 0);
  }

  
  // resets timeseries synthetization parameters
  template <typename T>
  void RNN_RBM<T>::synthStart()
  {
    vprev.resize(dimVisible);
    rprev.resize(dimRecurrent);
    vprev.zero();
    rprev.zero();

    synthIsInitialized = true;
  }

  
  // synthesizes next timestep by using the model
  template <typename T>
  bool RNN_RBM<T>::synthNext(whiteice::math::vertex<T>& vnext)
  {
    if(synthIsInitialized == false) return false;
	
    const unsigned int CDk = 4;
    
    
    whiteice::math::vertex<T> input(dimVisible + dimRecurrent);
    input.write_subvertex(vprev, 0);
    input.write_subvertex(rprev, dimVisible);
    
    whiteice::math::vertex<T> output;
    
    nn.calculate(input, output);
    
    whiteice::math::vertex<T> a(dimVisible), b(dimHidden);
    
    output.subvertex(a, 0, dimVisible);
    output.subvertex(b, dimVisible, dimHidden);

    whiteice::math::vertex<T> vstar(dimVisible);
    
    // uses CD-k to calculate v* (CD-k estimate)
    {
      rbm.setBValue(b);
      rbm.setAValue(a);


      rbm.setVisible(vprev);
      rbm.reconstructData(2*CDk);
      
      rbm.getVisible(vstar);
      
      vnext = vstar;
    }

    output.subvertex(rprev, dimVisible + dimHidden, dimRecurrent);

    synthIsInitialized = false;
    
    return true;
  }

  
  // synthesizes N next candidates using the probabilistic model
  template <typename T>
  bool RNN_RBM<T>::synthNext(unsigned int N, std::vector< whiteice::math::vertex<T> >& vnext)
  {
  if(synthIsInitialized == false) return false;
    
    const unsigned int CDk = 4;
    
    
    whiteice::math::vertex<T> input(dimVisible + dimRecurrent);
    input.write_subvertex(vprev, 0);
    input.write_subvertex(rprev, dimVisible);
    
    whiteice::math::vertex<T> output;
    
    nn.calculate(input, output);
    
    whiteice::math::vertex<T> a(dimVisible), b(dimHidden);
    
    output.subvertex(a, 0, dimVisible);
    output.subvertex(b, dimVisible, dimHidden);

    whiteice::math::vertex<T> vstar(dimVisible);
    
    // uses CD-k to calculate v* (CD-k estimate)
    {
      rbm.setBValue(b);
      rbm.setAValue(a);

      for(unsigned int i=0;i<N;i++){
	rbm.setVisible(vprev);
	rbm.reconstructData(2*CDk);
	
	rbm.getVisible(vstar);

	vnext.push_back(vstar);
      }
    }

    output.subvertex(rprev, dimVisible + dimHidden, dimRecurrent);

    synthIsInitialized = false;
    
    return true;
  }

  
  // selects given v as the next step in time-series
  // (needed to be called before calling again synthNext())
  template <typename T>
  bool RNN_RBM<T>::synthSetNext(whiteice::math::vertex<T>& v)
  {
    vprev = v;
    synthIsInitialized = true;

    return true;
  }


  template <typename T>
  bool RNN_RBM<T>::save(const std::string& basefilename) const
  {
    
    // saves generic variables
    {
      whiteice::dataset<T> conf;
      whiteice::math::vertex<T> v;

      if(conf.createCluster("dimensions", 1) == false) return false;
      v.resize(1);
      v.zero();

      v[0] = T(this->dimVisible);
      if(conf.add(0, v) == false) return false;

      v[0] = T(this->dimHidden);
      if(conf.add(0, v) == false) return false;

      v[0] = T(this->dimRecurrent);
      if(conf.add(0, v) == false) return false;

      v[0] = T((int)(this->synthIsInitialized));
      if(conf.add(0, v) == false) return false;

      if(conf.createCluster("vprev", this->dimVisible) == false) return false;
      if(conf.add(1, this->vprev) == false) return false;

      if(conf.createCluster("rprev", this->dimRecurrent) == false) return false;
      if(conf.add(2, this->rprev) == false) return false;

      if(conf.save(basefilename) == false) return false;
    }

    // saves model data
    {
      char buffer[256];
      snprintf(buffer, 256, "%s.rnn", basefilename.c_str());

      whiteice::bayesian_nnetwork<T> bnet;
      if(bnet.importNetwork(this->nn) == false) return false;

      if(bnet.save(buffer) == false) return false;

      snprintf(buffer, 256, "%s.bbrbm", basefilename.c_str());

      if(this->rbm.save(buffer) == false) return false;
    }

    return true;
  }


  template <typename T>
  bool RNN_RBM<T>::load(const std::string& basefilename)
  {
    // tries to load RNN_RBM

    unsigned int dimVisible;
    unsigned int dimHidden;
    unsigned int dimRecurrent;
    bool synthIsInitialized;

    whiteice::math::vertex<T> vprev;
    whiteice::math::vertex<T> rprev;

    // generic variables
    {
      whiteice::dataset<T> conf;
      whiteice::math::vertex<T> v;

      if(conf.load(basefilename) == false) return false;
      if(conf.getNumberOfClusters() != 3) return false;

      if(conf.size(0) != 4) return false;
      if(conf.dimension(0) != 1) return false;

      v.resize(1);
      v.zero();

      v = conf.access(0, 0); // dimVisible
      if(v[0] < T(0.0)) return false;
      whiteice::math::convert(dimVisible, floor(v[0]));

      v = conf.access(0, 1); // dimHidden
      if(v[0] < T(0.0)) return false;
      whiteice::math::convert(dimHidden, floor(v[0]));
      
      v = conf.access(0, 2); // dimRecurrent
      if(v[0] < T(0.0)) return false;
      whiteice::math::convert(dimRecurrent, floor(v[0]));
      
      v = conf.access(0, 3); // synthIsInitialized
      if(v[0] < T(0.0)) return false;
      int tmp = 0;
      whiteice::math::convert(tmp, floor(v[0]));
      synthIsInitialized = (bool)tmp;

      v.resize(dimVisible); // vprev
      if(conf.dimension(1) != dimVisible) return false;
      v = conf.access(1, 0);
      vprev = v;

      v.resize(dimRecurrent); // vprev
      if(conf.dimension(2) != dimRecurrent) return false;
      v = conf.access(2, 0);
      rprev = v;
    }

    // tries to load model data
    whiteice::nnetwork<T> nn;
    whiteice::BBRBM<T> rbm;
    
    {
      whiteice::bayesian_nnetwork<T> bnet;
      std::vector< whiteice::math::vertex<T> > weights;

      char buffer[256];
      snprintf(buffer, 256, "%s.rnn", basefilename.c_str());

      if(bnet.load(buffer) == false) return false;

      if(bnet.exportSamples(nn, weights, 0) == false) return false;
      if(weights.size() <= 0) return false;

      if(nn.input_size() != dimVisible + dimRecurrent) return false;
      if(nn.output_size() != dimVisible + dimHidden + dimRecurrent) return false;

      if(nn.importdata(weights[0]) == false) return false;

      
      snprintf(buffer, 256, "%s.bbrbm", basefilename.c_str());

      if(rbm.load(buffer) == false) return false;

      if(rbm.getVisibleNodes() != dimVisible) return false;
      if(rbm.getHiddenNodes() != dimHidden) return false;
    }

    // data loaded successfully: sets global variables
    {
      this->dimVisible = dimVisible;
      this->dimHidden  = dimHidden;
      this->dimRecurrent = dimRecurrent;
      
      this->synthIsInitialized = synthIsInitialized;      
      this->vprev = vprev;
      this->rprev = rprev;

      this->nn = nn;
      this->rbm = rbm;
    }

    return true;
  }
  

  template <typename T>
  T RNN_RBM<T>::reconstructionError(whiteice::BBRBM<T>& rbm,
				    whiteice::nnetwork<T>& nn, 
				    const std::vector< std::vector< whiteice::math::vertex<T> > >& timeseries) const
  {
    if(timeseries.size() <= 0) return T(0.0);
    if(timeseries[0].size() <= 0) return T(0.0);
    if(timeseries[0][0].size() != dimVisible) return T(INFINITY);

    T error = T(0.0);
    unsigned int counter = 0;
    const unsigned int CDk = 1;
    

    for(unsigned int n=0;n<timeseries.size();n++){
      
      whiteice::math::vertex<T> r(dimRecurrent);
      r.zero();
      
      whiteice::math::vertex<T> v(dimVisible);
      v.zero();
      
      for(unsigned int i=0;i<timeseries[n].size();i++){
	whiteice::math::vertex<T> input(dimVisible + dimRecurrent);
	input.write_subvertex(v, 0);
	input.write_subvertex(r, dimVisible);

	whiteice::math::vertex<T> output;
	
	nn.calculate(input, output);

	whiteice::math::vertex<T> a(dimVisible), b(dimHidden);
	
	output.subvertex(a, 0, dimVisible);
	output.subvertex(b, dimVisible, dimHidden);

	v = timeseries[n][i]; // visible element

	whiteice::math::vertex<T> vstar(dimVisible);
	
	// uses CD-k to calculate v* (CD-k estimate)
	{
	  rbm.setBValue(b);
	  rbm.setAValue(a);
	  
	  rbm.setVisible(v);
	  rbm.reconstructData(2*CDk);
	  
	  rbm.getVisible(vstar);
	}

	error += (v - vstar).norm();
	counter++;
	  
	output.subvertex(r, dimVisible + dimHidden, dimRecurrent);	
      }
    }

    error /= T(counter);

    return error;
  }



  template class RNN_RBM< whiteice::math::blas_real<float> >;
  template class RNN_RBM< whiteice::math::blas_real<double> >;
}
