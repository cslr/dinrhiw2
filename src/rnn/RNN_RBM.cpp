
#include "RNN_RBM.h"
#include "matrix.h"
#include "vertex.h"


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
    
  }
  
  
  template <typename T>
  RNN_RBM<T>::~RNN_RBM()
  {
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
    const T epsilon = T(0.0001);
    
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
	  v = timeseries[n][i]; // visible element for the next timestep
	}

      }


      // after we have calculated gradients through time-series
      // updates parameters can computes reconstruction error and reports it
      if(numgradients > 0)
      {
	grad_W /= T(numgradients);
	grad_w /= T(numgradients);

	whiteice::math::matrix<T> W;
	whiteice::math::vertex<T> w;
	
	W = rbm.getWeights();
	W += epsilon*grad_W;
	rbm.setWeights(W);

	nn.exportdata(w);
	w += epsilon*grad_w;
	nn.importdata(w);

	T error = reconstructionError(timeseries);

	printf("RNN_RBM ITER %d RECONSTRUCTION ERROR: %f\n",
	       iterations, error.c[0]);
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

	  if(ratio <= T(0.01)){
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
    // TODO implement me!
  }

  
  // synthesizes next timestep by using the model
  template <typename T>
  bool RNN_RBM<T>::synthNext(whiteice::math::vertex<T>& vnext)
  {
    return false; // TODO implement me!
  }

  
  // synthesizes N next candidates using the probabilistic model
  template <typename T>
  bool RNN_RBM<T>::synthNext(unsigned int N, std::vector< whiteice::math::vertex<T> >& vnext)
  {
    return false; // TODO implement me!
  }

  
  // selects given v as the next step in time-series
  // (needed to be called before calling again synthNext())
  template <typename T>
  bool RNN_RBM<T>::synthSetNext(whiteice::math::vertex<T>& v)
  {
    return false; // TODO implement me!
  }
  

  template <typename T>
  T RNN_RBM<T>::reconstructionError(const std::vector< std::vector< whiteice::math::vertex<T> > >& timeseries)
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
	v = timeseries[n][i]; // visible element for the next timestep
      }
    }

    error /= T(counter);

    return error;
  }



  template class RNN_RBM< whiteice::math::blas_real<float> >;
  template class RNN_RBM< whiteice::math::blas_real<double> >;
}
