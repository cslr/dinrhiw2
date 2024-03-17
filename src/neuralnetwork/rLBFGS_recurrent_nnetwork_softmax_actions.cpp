
#include "rLBFGS_recurrent_nnetwork_softmax_actions.h"
#include "deep_ica_network_priming.h"

#include "eig.h"
#include "EnsembleMeans.h"


namespace whiteice
{

  template <typename T>
  rLBFGS_recurrent_nnetwork_softmax_actions<T>::rLBFGS_recurrent_nnetwork_softmax_actions(const nnetwork<T>& nn,
							  const dataset<T>& d,
							  bool overfit):
    whiteice::math::LBFGS<T>(overfit),
    net(nn), data(d)
  {
    assert(data.getNumberOfClusters() == 3);

    // checks network has correct architecture
    {
      const unsigned int RDIM = net.input_size()-data.dimension(0);

      assert(RDIM >= 1);
      
      assert(net.input_size() == data.dimension(0)+RDIM);
      assert(net.output_size() == data.dimension(1)+RDIM);
    }
    

    // divides data in episoids to to training and testing sets
    ///////////////////////////////////////////////
    {
      dtrain = data;
      dtest  = data;
      
      dtrain.clearData(0);
      dtrain.clearData(1);
      dtrain.clearData(2);
      
      dtest.clearData(0);
      dtest.clearData(1);
      dtest.clearData(2);
      
      for(unsigned int e=0;e<data.size(3);e++){
	math::vertex<T> range = data.access(3, e);

	assert(range.size() == 2);
	assert(range[0] < range[1] && range[0] < data.size(0) && range[1] <= data.size(0));
	unsigned int r1, r2;

	whiteice::math::convert(r1, range[0]);
	whiteice::math::convert(r2, range[1]);

	const unsigned int LEN = (r2 - r1);
	
	const unsigned int r = (rng.rand() % 4);
	
	if(r != 0){ // training dataset 75% of cases go here

	  const unsigned int start = dtrain.size(0);

	  range[0] = start;
	  range[1] = start+LEN;

	  for(unsigned int i=r1;i<r2;i++){
	    math::vertex<T> in  = data.access(0,i);
	    math::vertex<T> out = data.access(1,i);
	    
	    dtrain.add(0, in,  true);
	    dtrain.add(1, out, true);
	  }

	  dtrain.add(2, range, true);
	}
	else{

	  const unsigned int start = dtest.size(0);

	  range[0] = start;
	  range[1] = start+LEN;

	  for(unsigned int i=r1;i<r2;i++){
	    math::vertex<T> in  = data.access(0,i);
	    math::vertex<T> out = data.access(1,i);
	    
	    dtest.add(0, in,  true);
	    dtest.add(1, out, true);
	  }

	  dtest.add(2, range, true);
	}
	
      }
      
      // we cannot never have zero training or testing set size
      // in such a small cases (very little data) we just use
      // all the data both for training and testing and overfit
      if(dtrain.size(0) == 0 || dtest.size(0) == 0 || overfit){
	dtrain = data;
	dtest  = data;
      }
    }
    
  }

  
  template <typename T>
  rLBFGS_recurrent_nnetwork_softmax_actions<T>::~rLBFGS_recurrent_nnetwork_softmax_actions()
  {
  }


  template <typename T>
  T rLBFGS_recurrent_nnetwork_softmax_actions<T>::getError(const math::vertex<T>& x) const
  {
    T e = T(0.0f);

    { // recurrent neural network structure, we assume episode start/end is given in the 3rd cluster
      
      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);

#pragma omp parallel shared(e)
      {
	math::vertex<T> err, correct;
	T esum = T(0.0f);
	
	const unsigned int INPUT_DATA_DIM = dtest.dimension(0);
	const unsigned int OUTPUT_DATA_DIM = dtest.dimension(1);
	const unsigned int RDIM = nnet.output_size() - OUTPUT_DATA_DIM;

	math::vertex<T> input, output, output_r;
	input.resize(dtest.dimension(0)+RDIM);
	output.resize(dtest.dimension(1)+RDIM);
	output_r.resize(RDIM);
	err.resize(dtest.dimension(1));
	
	// E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(auto)
	for(unsigned int episode=0;episode<dtest.size(2);episode++){

	  math::vertex<T> range = dtest.access(2,episode);

	  unsigned int start = 0; 
	  unsigned int length = 0;

	  whiteice::math::convert(start, range[0]);
	  whiteice::math::convert(length, range[1]);
	  
	  input.zero();
	  
	  // recurrency: feebacks output back to inputs and
	  //             calculates error
	  for(unsigned int i = start;i<length;i++){
	    input.write_subvertex(dtest.access(0, i), 0);
	    
	    nnet.calculate(input, output);

	    output.subvertex(output_r, dtest.dimension(1), RDIM);
	    assert(input.write_subvertex(output_r, INPUT_DATA_DIM));

	    output.subvertex(err, 0, dtest.dimension(1));
	    //nnet.softmax_output(err, 0, err.size());

	    // assumes there is no postprocessing in dataset of output values

	    correct = dtest.access(1, i);

	    esum += nnet.kl_divergence(err, 0, err.size(), correct)
	      - entropy_regularizer*nnet.entropy(err, 0, err.size());
	  }
	  
	}
	
#pragma omp critical
	{
	  e += esum; // per each recurrency
	}
	
      }
      
    }
    
    e /= T( (float)dtest.size(0) ); // per N
    
    return e;
  }

  

  template <typename T>
  T rLBFGS_recurrent_nnetwork_softmax_actions<T>::U(const math::vertex<T>& x) const
  {
    T e = T(0.0f);

    { // recurrent neural network

      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);
      
#pragma omp parallel shared(e)
      {
	//whiteice::nnetwork<T> nnet(this->net);
	//nnet.importdata(x);
	
	math::vertex<T> err, correct;
	T esum = T(0.0f);
	
	const unsigned int INPUT_DATA_DIM = dtrain.dimension(0);
	const unsigned int OUTPUT_DATA_DIM = dtrain.dimension(1);
	const unsigned int RDIM = nnet.output_size() - OUTPUT_DATA_DIM;

	math::vertex<T> input, output, output_r;
	input.resize(dtrain.dimension(0)+RDIM);
	output.resize(dtrain.dimension(1)+RDIM);
	output_r.resize(RDIM);
	err.resize(dtrain.dimension(1));
	
	// E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(auto)
	for(unsigned int episode=0;episode<dtrain.size(2);episode++){

	  math::vertex<T> range = dtrain.access(2,episode);

	  unsigned int start = 0; 
	  unsigned int length = 0;

	  whiteice::math::convert(start, range[0]);
	  whiteice::math::convert(length, range[1]);
	  
	  input.zero();
	  
	  // recurrency: feebacks output back to inputs and
	  //             calculates error
	  for(unsigned int i = start;i<length;i++){
	    input.write_subvertex(dtrain.access(0, i), 0);
	    
	    //nnet.input() = input;
	    //nnet.calculate(false);
	    nnet.calculate(input, output);

	    output.subvertex(output_r, dtrain.dimension(1), RDIM);
	    assert(input.write_subvertex(output_r, INPUT_DATA_DIM));

	    output.subvertex(err, 0, dtrain.dimension(1));

	    correct = dtrain.access(1, i);

	    esum += nnet.kl_divergence(err, 0, err.size(), correct)
	      - entropy_regularizer*nnet.entropy(err, 0, err.size());
	  }
	  
	}
	
#pragma omp critical
	{
	  e += esum; // per each recurrency
	}
	
      }
      
    }
    
    e /= T(dtrain.size(0));
    
#if 1
    {
      // regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
      auto err = T(0.5)*alpha*(x*x)[0];
      e += err;
    }
#endif
    
    return (e);    
  }


  // clipped gradient values which otherwise cause exploding gradients in LBFGS optimization
  template <typename T>
  void rLBFGS_recurrent_nnetwork_softmax_actions<T>::box_values(math::matrix<T>& GRAD) const
  {
#pragma omp parallel for schedule(static)
    for(unsigned int i=0;i<GRAD.size();i++){
      if(GRAD[i] > T(1e4f)) GRAD[i] = T(1e4f);
      else if(GRAD[i] < T(-1e4f)) GRAD[i] = T(-1e4);
    }
  }

    
  template <typename T>
  math::vertex<T> rLBFGS_recurrent_nnetwork_softmax_actions<T>::Ugrad(const math::vertex<T>& x) const
  {

    {
      // recurrent neural network!
      math::vertex<T> sumgrad;
      sumgrad = x;
      sumgrad.zero();

      const unsigned int INPUT_DATA_DIM = dtrain.dimension(0);
      const unsigned int OUTPUT_DATA_DIM = dtrain.dimension(1);
      const unsigned int RDIM = net.output_size() - OUTPUT_DATA_DIM;
      const unsigned int RDIM2 = net.input_size() - INPUT_DATA_DIM;
      assert(RDIM == RDIM2);
      assert(dtrain.size(0) == dtrain.size(1));

      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);
      
#pragma omp parallel shared(sumgrad)
      {
	math::vertex<T> grad1, grad2;
	math::vertex<T> sgrad;
	sgrad = x;
	sgrad.zero();
	grad1 = x;
	grad2 = x;

	math::vertex<T> input, output, output_r;
	input.resize(dtrain.dimension(0)+RDIM);
	output_r.resize(RDIM);

	math::matrix<T> UGRAD;
	UGRAD.resize(dtrain.dimension(1)+RDIM, nnet.gradient_size());

	math::matrix<T> URGRAD;
	URGRAD.resize(RDIM, nnet.gradient_size());

	//math::matrix<T> UYGRAD;
	//UYGRAD.resize(dtrain.dimension(1), nnet.gradient_size());

	math::matrix<T> FGRAD;
	FGRAD.resize(dtrain.dimension(1)+RDIM, nnet.gradient_size());

	math::matrix<T> FRGRAD;
	FRGRAD.resize(RDIM, nnet.output_size());

	math::matrix<T> FGRADTMP;
	FGRADTMP.resize(dtrain.dimension(1)+RDIM, RDIM);

#pragma omp for nowait schedule(auto)
	for(unsigned int episode=0;episode<dtrain.size(2);episode++){
	  
	  math::vertex<T> range = dtrain.access(2,episode);

	  unsigned int START = 0; 
	  unsigned int END = 0;

	  whiteice::math::convert(START, range[0]);
	  whiteice::math::convert(END, range[1]);

	  assert(START < dtrain.size(0) && START < dtrain.size(1));
	  assert(START < END);
	  assert(END <= dtrain.size(0) && END <= dtrain.size(1));
	  
	  UGRAD.zero();
	  grad1.zero();
	  grad2.zero();
	  
	  input.zero();
	  
	  for(unsigned int i=START;i<END;i++){
	    assert(input.write_subvertex(dtrain.access(0,i), 0) == true);
	      
	    assert(nnet.jacobian(input, FGRAD) == true);
	    // df/dw (dtrain.dimension(1)+RDIM, nnet.gradient_size())

	    {
	      assert(nnet.gradient_value(input, FGRADTMP) == true);
	      // df/dinput (dtrain.dimension(1)+RDIM,dtrain.dimension(0)+RDIM)

	      // df/dr
	      assert(FGRADTMP.submatrix(FRGRAD,
					dtrain.dimension(0), 0,
					RDIM, nnet.output_size()) == true);

	      // KAPPA_r = I

	      // df/dr (dtrain.dimension(1)+RDIM, RDIM)
	      // dU/dw (dtrain.dimension(1)+RDIM, nnet.gradient_size())

	      // KAPPA_r operation to UGRAD to select only R terms
	      assert(UGRAD.submatrix(URGRAD,
				     0,dtrain.dimension(1),
				     nnet.gradient_size(), RDIM) == true);
	    }

	    // dU(n+1)/dw = df/dw + df/dr * KAPPA_r * dU(n)/dw
	    UGRAD = FGRAD + FRGRAD*URGRAD;

	    // clipped gradients!!
	    box_values(UGRAD);

	    assert(nnet.calculate(input, output) == true);
	    
	    { // calculate error gradient value for E(i=0)..E(N) terms
	      
	      assert(nnet.kl_divergence_gradient_j(output, 0, dtrain.dimension(1),
						   dtrain.access(1,i),
						   UGRAD,
						   grad1));

	      assert(nnet.entropy_gradient_j(output, 0, dtrain.dimension(1),
					     UGRAD,
					     grad2));
		     

	      sgrad += grad1 - entropy_regularizer*grad2;
	    }

	    assert(output.subvertex(output_r, dtrain.dimension(1), RDIM));
	    assert(input.write_subvertex(output_r, INPUT_DATA_DIM));
	  }

	}
	
#pragma omp critical
	{
	  sumgrad += sgrad;
	}
	
      }

      sumgrad /= T(dtrain.size(0));
      
#if 1
      {
	// regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
	
	sumgrad += alpha*x;
      }
#endif
      
      return sumgrad;
    }


  }
  
  
  template <typename T>
  bool rLBFGS_recurrent_nnetwork_softmax_actions<T>::heuristics(math::vertex<T>& x) const
  {
    return true;
  }
  
  
  template class rLBFGS_recurrent_nnetwork_softmax_actions< math::blas_real<float> >;
  template class rLBFGS_recurrent_nnetwork_softmax_actions< math::blas_real<double> >;

  
};
