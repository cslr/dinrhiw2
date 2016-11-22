
#include "rLBFGS_nnetwork.h"
#include "deep_ica_network_priming.h"

#include "eig.h"


namespace whiteice
{

  template <typename T>
  rLBFGS_nnetwork<T>::rLBFGS_nnetwork(const nnetwork<T>& nn,
				    const dataset<T>& d,
				    const unsigned int deepness_,
				    bool overfit,
				    bool negativefeedback) :
    whiteice::math::LBFGS<T>(overfit), deepness(deepness_),
    net(nn), data(d)
  {
    this->negativefeedback = negativefeedback;

    assert(data.getNumberOfClusters() == 2);

    // checks network has correct architecture
    if(deepness > 1)
    {
      assert(net.input_size() == data.dimension(0)+data.dimension(1));
    }
    else{
      assert(net.input_size() == data.dimension(0));
    }

    assert(net.output_size() == data.dimension(1));
    
    // divides data to to training and testing sets
    ///////////////////////////////////////////////
    {
      
      dtrain = data;
      dtest  = data;
      
      dtrain.clearData(0);
      dtrain.clearData(1);
      dtest.clearData(0);
      dtest.clearData(1);
      
      
      for(unsigned int i=0;i<data.size(0);i++){
	const unsigned int r = (rand() & 1);
	
	if(r == 0){
	  math::vertex<T> in  = data.access(0,i);
	  math::vertex<T> out = data.access(1,i);
	  
	  dtrain.add(0, in,  true);
	  dtrain.add(1, out, true);
	}
	else{
	  math::vertex<T> in  = data.access(0,i);
	  math::vertex<T> out = data.access(1,i);
	  
	  dtest.add(0, in,  true);
	  dtest.add(1, out, true);
	}
      }
      
      // we cannot never have zero training or testing set size
      // in such a small cases (very little data) we just use
      // all the data both for training and testing and overfit
      if(dtrain.size(0) == 0 || dtest.size(0) == 0){
	dtrain = data;
	dtest  = data;
      }
    }
    
  }

  
  template <typename T>
  rLBFGS_nnetwork<T>::~rLBFGS_nnetwork()
  {
  }


  template <typename T>
  T rLBFGS_nnetwork<T>::getError(const math::vertex<T>& x) const
  {
    T e = T(0.0f);

    if(deepness <= 1){
    
#pragma omp parallel shared(e)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> err;
	T esum = T(0.0f);
	
	
	// E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(dynamic)
	for(unsigned int i=0;i<dtest.size(0);i++){
	  nnet.input() = dtest.access(0, i);
	  nnet.calculate(false);
	  err = dtest.access(1, i) - nnet.output();
	  
	  err = (err*err);
	  esum += T(0.5f)*err[0];
	}
	
#pragma omp critical
	{
	  e += esum;
	}
	
      }
    }
    else{ // recurrent neural network structure
      
#pragma omp parallel shared(e)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> err;
	T esum = T(0.0f);
	
	
	// E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(dynamic)
	for(unsigned int i=0;i<dtest.size(0);i++){
	  math::vertex<T> input;
	  input.resize(dtest.dimension(0)+dtest.dimension(1));
	  input.zero();
	  input.write_subvertex(dtest.access(0,i), 0);

	  // recurrency: feebacks output back to inputs and
	  //             calculates error
	  for(unsigned int d = 0;d<deepness;d++){
	    nnet.input() = input;
	    nnet.calculate(false);	    
	    input.write_subvertex(nnet.output(), dtest.dimension(0));
	  }
	  
	  err = dtest.access(1, i) - nnet.output();
	  err = (err*err);
	  esum += T(0.5f)*err[0];
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
  T rLBFGS_nnetwork<T>::U(const math::vertex<T>& x) const
  {
    T e = T(0.0f);

    if(deepness <= 1){
#pragma omp parallel
      {
	whiteice::nnetwork<T> nnet(this->net);
	
	nnet.importdata(x);
	math::vertex<T> err;
	T esum = T(0.0f);
	
	// E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(dynamic)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  nnet.input() = dtrain.access(0, i);
	  nnet.calculate(false);
	  err = dtrain.access(1, i) - nnet.output();
	  err = (err*err); // /T(dtrain.size(0));
	  esum += T(0.5f)*err[0];
	}
	
#pragma omp critical
	{
	  e += esum;
	}
	
      }
      
    }
    else{ // recurrent neural network
      
#pragma omp parallel shared(e)
      {
	whiteice::nnetwork<T> nnet(this->net);
	
	nnet.importdata(x);
	math::vertex<T> err;
	T esum = T(0.0f);
	
	// E = SUM 0.5*e(i)^2
#pragma omp for nowait schedule(dynamic)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  math::vertex<T> input;
	  input.resize(nnet.input_size());
	  input.zero();
	  assert(input.write_subvertex(dtrain.access(0,i), 0));

	  // recurrency: feebacks output back to inputs and
	  //             calculates error
	  for(unsigned int d = 0;d<deepness;d++){
	    nnet.input() = input;
	    nnet.calculate(false);
	    err = dtrain.access(1, i) - nnet.output();
	    
	    err = (err*err);
	    esum += T(0.5f)*err[0];

	    assert(input.write_subvertex(nnet.output(), nnet.output_size()));
	  }
	  
	}
	
#pragma omp critical
	{
	  e += esum;
	}
	
      }
      
    }
      

    {
	T alpha = T(0.01);   // regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
	auto err = T(0.5)*alpha*(x*x);
	e += err[0];
    }


    e /= T(dtrain.size(0));
    
    return (e);    
  }

  
  template <typename T>
  math::vertex<T> rLBFGS_nnetwork<T>::Ugrad(const math::vertex<T>& x) const
  {
    math::vertex<T> sumgrad;
    sumgrad = x;
    sumgrad.zero();

    if(deepness <= 1){

      math::matrix<T> sigma2;
      sigma2.resize(net.output_size(), net.output_size());
      sigma2.zero();
      
      math::vertex<T> m(net.output_size());
      m.zero();
    
      // positive phase/gradient
#pragma omp parallel shared(sumgrad)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> sgrad, grad, err;
	
	math::matrix<T> sum_sigma2;
	sum_sigma2.resize(net.output_size(), net.output_size());
	sum_sigma2.zero();
	
	math::vertex<T> sum_m(net.output_size());
	sum_m.zero();
	
	sgrad = x;
	sgrad.zero();
	
#pragma omp for nowait schedule(dynamic)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  nnet.input() = dtrain.access(0, i);
	  nnet.calculate(true);
	  err = dtrain.access(1,i) - nnet.output();
	  
	  if(nnet.gradient(err, grad) == false){
	    std::cout << "gradient failed." << std::endl;
	    assert(0); // FIXME
	  }
	  
	  sum_sigma2 += err.outerproduct(err);
	  sum_m += err; // should not be calculated??
	  
	  sgrad += grad; // /T(dtrain.size(0));
	}
	
#pragma omp critical
	{
	  sumgrad += sgrad;
	  sigma2 += sum_sigma2;
	  m += sum_m;
	}
	
      }
      
      sigma2 /= T((double)dtrain.size(0));
      m /= T((double)dtrain.size(0));
      sigma2 -= m.outerproduct(m);
      
      // sigma2 is error covariance matrix
      // (we assume there are only few output dimensions)
      // we need to sample from N(O, Sigma2) solve X*sqrt(D)*n
      auto D = sigma2;
      math::matrix<T> X;
      
      while(symmetric_eig(D, X) == false){
	for(unsigned int i=0;i<m.size();i++)
	  sigma2(i,i) += T(1.0); // dummy regularizer..
	D = sigma2;
      }
      
      sigma2 = D;
      
      for(unsigned int i=0;i<m.size();i++){
	sigma2(i,i) = sqrt(abs(sigma2(i,i)));
      }
      
      sigma2 = X*sigma2;
      
      
      // negative phase/gradient
#pragma omp parallel shared(sumgrad)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> sgrad, grad, err;
	
	sgrad = x;
	sgrad.zero();
	
#pragma omp for nowait schedule(dynamic)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  // generates negative particle
	  auto x = dtrain.access(0, rng.rand() % dtrain.size(0));
	  
	  nnet.input() = x;
	  nnet.calculate(true);
	  auto y = nnet.output();
	  
	  math::vertex<T> n(y.size());
	  rng.normal(n);
	  y += sigma2 * n; // adds properly correlated noise..
	  
	  err = y - nnet.output();
	  
	  if(nnet.gradient(err, grad) == false){
	    std::cout << "gradient failed." << std::endl;
	    assert(0); // FIXME
	  }
	  
	  sgrad -= grad; // /T(dtrain.size(0));
	}
	
#pragma omp critical
	{
	  sumgrad += sgrad;
	}
	
      }

    }
    else{ // recurrent neural network!

      std::vector< math::matrix<T> > sigma2;
      std::vector< math::vertex<T> > m;
      sigma2.resize(deepness);
      m.resize(deepness);

      for(unsigned int d=0;d<deepness;d++){
	sigma2[d].resize(net.output_size(), net.output_size());
	sigma2[d].zero();
	m[d].resize(net.output_size());
	m[d].zero();
      }

      
      // positive phase/gradient
#pragma omp parallel shared(sumgrad) shared(sigma2) shared(m)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> sgrad, grad, err;
	
	std::vector< math::matrix<T> > sum_sigma2;
	sum_sigma2.resize(deepness);

	std::vector< math::vertex<T> > sum_m;
	sum_m.resize(deepness);

	for(unsigned int d=0;d<deepness;d++){
	  sum_sigma2[d].resize(net.output_size(), net.output_size());
	  sum_sigma2[d].zero();
	  sum_m[d].resize(net.output_size());
	  sum_m[d].zero();
	}
	
	
	sgrad = x;
	sgrad.zero();
	
#pragma omp for nowait schedule(dynamic)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  math::vertex<T> input;
	  input.resize(net.input_size());
	  input.zero();
	  input.write_subvertex(dtrain.access(0,i), 0);

	  if(rng.rand() & 1) // trains to use correct input (50% chance)
	    input.write_subvertex(dtrain.access(1,i), dtrain.dimension(0));

	  for(unsigned int d=0;d<deepness;d++){
	    nnet.input() = input;
	    nnet.calculate(true);
	    err = dtrain.access(1,i) - nnet.output();
	    
	    if(nnet.gradient(err, grad) == false){
	      std::cout << "gradient failed." << std::endl;
	      assert(0); // FIXME
	    }
	    
	    sum_sigma2.at(d) += err.outerproduct(err);
	    // sum_m.at(d) += err;
	    
	    sgrad += grad; // /T(dtrain.size(0));

	    input.write_subvertex(nnet.output(), dtrain.dimension(0));
	  }
	  
	}
	
#pragma omp critical
	{
	  sumgrad += sgrad;

	  for(unsigned int d=0;d<deepness;d++){ 
	    sigma2[d] += sum_sigma2[d];
	    m[d] += sum_m[d];
	  }
	}
	
      }


      for(unsigned int d=0;d<deepness;d++){
      
	sigma2[d] /= T((double)dtrain.size(0));
	m[d] /= T((double)dtrain.size(0));
	sigma2[d] -= m[d].outerproduct(m[d]);
	
	// sigma2 is error covariance matrix
	// (we assume there are only few output dimensions)
	// we need to sample from N(O, Sigma2) solve X*sqrt(D)*n
	auto D = sigma2[d];
	math::matrix<T> X;
	
	while(symmetric_eig(D, X) == false){
	  for(unsigned int i=0;i<m.size();i++)
	    sigma2[d](i,i) += T(1.0); // dummy regularizer..
	  D = sigma2[d];
	}
      
	sigma2[d] = D;
	
	for(unsigned int i=0;i<m[d].size();i++){
	  sigma2[d](i,i) = sqrt(abs(sigma2[d](i,i)));
	}
	
	sigma2[d] = X*sigma2[d];
      }
      
      
      // negative phase/gradient
#pragma omp parallel shared(sumgrad)
      {
	whiteice::nnetwork<T> nnet(this->net);
	nnet.importdata(x);
	
	math::vertex<T> sgrad, grad, err;
	
	sgrad = x;
	sgrad.zero();
	
#pragma omp for nowait schedule(dynamic)
	for(unsigned int i=0;i<dtrain.size(0);i++){
	  // generates negative particle
	  
	  const unsigned int index = rng.rand() % dtrain.size(0);
	  auto x = dtrain.access(0, index);
	  
	  math::vertex<T> input;
	  input.resize(dtrain.dimension(0)+dtrain.dimension(1));
	  input.zero();
	  input.write_subvertex(x, 0);
	  
	  if(rng.rand() & 1) // trains to use correct input (50% chance)
	    input.write_subvertex(dtrain.access(1,index), dtrain.dimension(0));

	  for(unsigned int d=0;d<deepness;d++){
	    nnet.input() = input;
	    nnet.calculate(true);
	    auto y = nnet.output();
	    
	    math::vertex<T> n(y.size());
	    rng.normal(n);
	    y += sigma2[d] * n; // adds properly correlated noise..
	    
	    err = y - nnet.output();
	    
	    if(nnet.gradient(err, grad) == false){
	      std::cout << "gradient failed." << std::endl;
	      assert(0); // FIXME
	    }
	  
	    sgrad -= grad; // /T(dtrain.size(0));

	    // input.write_subvertex(nnet.output(), dtrain.dimension(0));
	    input.write_subvertex(y, dtrain.dimension(0)); // INCLUDE NOISE IN FEEDBACK??
	  }
	}
	
#pragma omp critical
	{
	  sumgrad += sgrad;
	}
	
      }      

      
    }


    {
      T alpha = T(0.01f);
      sumgrad += alpha*x;
    }
    
    sumgrad /= T(dtrain.size(0));

	  
    // sumgrad.normalize();
    
    return (sumgrad);
  }
  
  
  template <typename T>
  bool rLBFGS_nnetwork<T>::heuristics(math::vertex<T>& x) const
  {
    if(negativefeedback){
      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);
      
      T alpha = T(0.5f);
      negative_feedback_between_neurons(nnet, dtrain, alpha);
      
      nnet.exportdata(x);
    }

#if 0
    // heuristic: for linear neuron weights to be always
    //            same as their non-linear pairs. this means signal is
    //            always transferred without problems to the next layer
    {
      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);
      
      for(unsigned int l=0;l<(nnet.getLayers()-1);l++){ // do not process the last layer..
	whiteice::math::vertex<T> b;
	whiteice::math::matrix<T> W;

	nnet.getBias(b, l);
	nnet.getWeights(W, l);
	
	for(unsigned int n=0;n<nnet.getNeurons(l);n += 2){
	  b[n] = b[n+1];
	  for(unsigned int j=0;j<W.xsize();j++){
	    W(n,j) = W(n+1,j);
	  }
	}

	nnet.setBias(b,l);
	nnet.setWeights(W,l);
      }
	    
    }
#endif

    return true;
  }
  
  
  template class rLBFGS_nnetwork< float >;
  template class rLBFGS_nnetwork< double >;
  template class rLBFGS_nnetwork< math::blas_real<float> >;
  template class rLBFGS_nnetwork< math::blas_real<double> >;

  
};
