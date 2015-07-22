
#include "LBFGS_nnetwork.h"
#include "deep_ica_network_priming.h"

namespace whiteice
{

  template <typename T>
  LBFGS_nnetwork<T>::LBFGS_nnetwork(const nnetwork<T>& nn, const dataset<T>& d, bool overfit, bool negativefeedback) :
    whiteice::math::LBFGS<T>(overfit), net(nn), data(d)
  {
	  this->negativefeedback = negativefeedback;

	  // divides data to to training and testing sets
	  ///////////////////////////////////////////////

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

  
  template <typename T>
  LBFGS_nnetwork<T>::~LBFGS_nnetwork()
  {
  }


  template <typename T>
  T LBFGS_nnetwork<T>::getError(const math::vertex<T>& x) const
  {
    whiteice::nnetwork<T> nnet(this->net);
    nnet.importdata(x);
    
    math::vertex<T> err;
    T e = T(0.0f);
    
    // E = SUM 0.5*e(i)^2
    for(unsigned int i=0;i<dtest.size(0);i++){
      // std::cout << "data in  = " << dtest.access(0, i) << std::endl;
      // std::cout << "data out = " << dtest.access(1, i) << std::endl;
      
      nnet.input() = dtest.access(0, i);
      nnet.calculate(false);
      err = dtest.access(1, i) - nnet.output();
      
      err = (err*err);
      e += T(0.5f)*err[0];
    }
    
    std::cout << "LBFGS::getError() = " << e << std::endl;
    
    e /= T( (float)dtest.size(0) ); // per N
    
    return e;
  }
  

  template <typename T>
  T LBFGS_nnetwork<T>::U(const math::vertex<T>& x) const
  {
    whiteice::nnetwork<T> nnet(this->net);
    
    nnet.importdata(x);
    
    math::vertex<T> err;
    T e = T(0.0f);

    // E = SUM 0.5*e(i)^2
    for(unsigned int i=0;i<dtrain.size(0);i++){
      nnet.input() = dtrain.access(0, i);
      nnet.calculate(false);
      err = dtrain.access(1, i) - nnet.output();
      err = (err*err); // /T(dtrain.size(0));
      e += T(0.5f)*err[0];
    }

#if 1    
    {
      T alpha = T(0.01);   // regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
      err = T(0.5)*alpha*(x*x);
      e += err[0];
    }
#endif

    e /= T(dtrain.size(0));
    
    return (e);    
  }

  
  template <typename T>
  math::vertex<T> LBFGS_nnetwork<T>::Ugrad(const math::vertex<T>& x) const
  {
	  whiteice::nnetwork<T> nnet(this->net);

	  math::vertex<T> sumgrad, grad, err;

	  nnet.importdata(x);

	  sumgrad = x;
	  sumgrad.zero();

	  for(unsigned int i=0;i<dtrain.size(0);i++){
		  nnet.input() = dtrain.access(0, i);
		  nnet.calculate(true);
		  err = dtrain.access(1,i) - nnet.output();

		  if(nnet.gradient(err, grad) == false){
			  std::cout << "gradient failed." << std::endl;
			  assert(0); // FIXME
		  }

		  sumgrad += grad; // /T(dtrain.size(0));
	  }

#if 1
	  {
		  T alpha = T(0.01f);
		  sumgrad += alpha*x;
	  }
#endif
	  
	  sumgrad /= T(dtrain.size(0));
	  
	  // sumgrad.normalize();
	  
	  return (sumgrad);
  }
  
  
  template <typename T>
  bool LBFGS_nnetwork<T>::heuristics(math::vertex<T>& x) const
  {
    if(negativefeedback){
      whiteice::nnetwork<T> nnet(this->net);
      nnet.importdata(x);
      
      T alpha = T(0.5f);
      negative_feedback_between_neurons(nnet, dtrain, alpha);
      
      nnet.exportdata(x);
    }

    return true;
  }
  
  
  template class LBFGS_nnetwork< float >;
  template class LBFGS_nnetwork< double >;
  template class LBFGS_nnetwork< math::blas_real<float> >;
  template class LBFGS_nnetwork< math::blas_real<double> >;

  
};
