
#include "BFGS_nnetwork.h"


namespace whiteice
{

  template <typename T>
  BFGS_nnetwork<T>::BFGS_nnetwork(const nnetwork<T>& nn, const dataset<T>& d) :
    net(nn), data(d)
  {
    
  }

  
  template <typename T>
  BFGS_nnetwork<T>::~BFGS_nnetwork()
  {
  }


  template <typename T>
  T BFGS_nnetwork<T>::getError() const
  {
    whiteice::nnetwork<T> nnet(this->net);

    math::vertex<T> x;
    T y;
    unsigned int iters;
    this->getSolution(x, y, iters);
    nnet.importdata(x);
    
    math::vertex<T> err;
    T e = T(0.0f);

    // E = SUM 0.5*e(i)^2
    for(unsigned int i=0;i<data.size(0);i++){
      nnet.input() = data.access(0, i);
      nnet.calculate(false);
      err = data.access(1, i) - nnet.output();
      T inv = T(1.0f/err.size());
      err = inv*(err*err);
      e += T(0.5f)*err[0];
    }
    
    e /= T( (float)data.size(0) ); // per N

    return e;
  }
  

  template <typename T>
  T BFGS_nnetwork<T>::U(const math::vertex<T>& x) const
  {
    whiteice::nnetwork<T> nnet(this->net);
    
    nnet.importdata(x);
    
    math::vertex<T> err;
    T e = T(0.0f);

    // E = SUM 0.5*e(i)^2
    for(unsigned int i=0;i<data.size(0);i++){
      nnet.input() = data.access(0, i);
      nnet.calculate(false);
      err = data.access(1, i) - nnet.output();
      // T inv = T(1.0f/err.size());
      err = (err*err);
      e += T(0.5f)*err[0];
    }
    
    // e /= T( (float)data.size(0) ); // per N

    {
      T alpha = T(0.01);   // regularizer exp(-0.5*||w||^2) term, w ~ Normal(0,I)
      err = alpha*(x*x);
      e += x[0];
    }
    
    
    return (e);    
  }

  
  template <typename T>
  math::vertex<T> BFGS_nnetwork<T>::Ugrad(const math::vertex<T>& x) const
  {
    whiteice::nnetwork<T> nnet(this->net);
    
    T ninv = T(1.0f); // T(1.0f/data.size(0));
    math::vertex<T> sumgrad, grad, err;

    nnet.importdata(x);

    for(unsigned int i=0;i<data.size(0);i++){
      nnet.input() = data.access(0, i);
      nnet.calculate(true);
      err = data.access(1,i) - nnet.output();
      
      if(nnet.gradient(err, grad) == false){
	std::cout << "gradient failed." << std::endl;
	assert(0); // FIXME
      }
      
      if(i == 0)
	sumgrad = ninv*grad;
      else
	sumgrad += ninv*grad;
    }

    T alpha = T(0.01f);

    sumgrad += alpha*x;

    // TODO: is this really correct gradient to use
    // (we want to use: 0,5*SUM e(i)^2 + alpha*w^2
    
    
    return (sumgrad);    
  }
  
  
  
  template class BFGS_nnetwork< float >;
  template class BFGS_nnetwork< double >;
  template class BFGS_nnetwork< math::blas_real<float> >;
  template class BFGS_nnetwork< math::blas_real<double> >;

  
};
