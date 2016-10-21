/*
 * BBRBM.cpp
 *
 *  Created on: 22.6.2015
 *      Author: Tomas
 */

#include "BBRBM.h"

namespace whiteice {

template <typename T>
BBRBM<T>::BBRBM()
{
  h.resize(1);
  v.resize(1);
  
  W.resize(1,1);
  a.resize(1);
  b.resize(1);

  initializeWeights();
}

template <typename T>
BBRBM<T>::BBRBM(const BBRBM<T>& rbm)
{
  this->v = rbm.v;
  this->h = rbm.h;

  this->W = rbm.W;
  this->a = rbm.a;
  this->b = rbm.b;
}

// creates 2-layer: V * H network
template <typename T>
BBRBM<T>::BBRBM(unsigned int visible, unsigned int hidden) throw(std::invalid_argument)
{
  if(visible == 0 || hidden == 0)
    throw std::invalid_argument("invalid network architecture");
    
  // the last term is always constant: 1 (one)
  v.resize(visible);
  h.resize(hidden);
  
  W.resize(hidden, visible);
  a.resize(visible);
  b.resize(hidden);
    
  initializeWeights();    
}

template <typename T>
BBRBM<T>::~BBRBM()
{

}


template <typename T>
BBRBM<T>& BBRBM<T>::operator=(const BBRBM<T>& rbm)
{
  this->v = rbm.v;
  this->h = rbm.h;

  this->W = rbm.W;
  this->a = rbm.a;
  this->b = rbm.b;

  return (*this);
}


template <typename T>
bool BBRBM<T>::resize(unsigned int visible, unsigned int hidden)
{
  if(visible == 0 || hidden == 0)
    return false;
    
  // the last term is always constant: 1 (one)
  v.resize(visible);
  h.resize(hidden);
  
  W.resize(hidden, visible);
  a.resize(visible);
  b.resize(hidden);
  
  initializeWeights(); // overwrites old weights..

  return true;
}

////////////////////////////////////////////////////////////
template <typename T>
void BBRBM<T>::getVisible(math::vertex<T>& v) const
{
  v = this->v;
}

template <typename T>
bool BBRBM<T>::setVisible(const math::vertex<T>& v)
{
  if(v.size() != this->v.size())
    return false;

  this->v = v;

  return true;
}


template <typename T>
void BBRBM<T>::getHidden(math::vertex<T>& h) const
{
  h = this->h;
}

template <typename T>
bool BBRBM<T>::setHidden(const math::vertex<T>& h)
{
  if(h.size() != this->h.size())
    return false;

  this->h = h;
  
  return true;
}

template <typename T>
bool BBRBM<T>::reconstructData(unsigned int iters)
{
  if(iters == 0) return false;
  
  math::matrix<T> Wt = W;
  Wt.transpose();
  
  while(iters > 0){
    h = W*v + b;
    
    // 1. hidden units: calculates sigma(a_j)
    for(unsigned int j=0;j<(h.size()-0);j++){
      T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
      T r = rng.uniform();
      
      if(aj > r) h[j] = T(1.0); // discretization step
      else       h[j] = T(0.0);
    }
    
    iters--;
    if(iters <= 0) return true;
    
    v = Wt*h + a;
    
    // 1. visible units: calculates sigma(a_j)
    for(unsigned int j=0;j<(v.size()-0);j++){
      T aj = T(1.0)/(T(1.0) + math::exp(-v[j]));
      T r = rng.uniform();
      
      if(aj > r) v[j] = T(1.0); // discretization step
      else       v[j] = T(0.0);
    }
    
    iters--;
    if(iters <= 0) return true;
  }
  
  return true;
}

template <typename T>
bool BBRBM<T>::reconstructData(std::vector< math::vertex<T> >& samples,
			       unsigned int iters)
{
  for(auto& s : samples){
    if(this->setVisible(s) == false)
      return false;
    
    if(this->reconstructData(2*iters) == false)
      return false;

    this->getVisible(s);
  }

  return true;
}

template <typename T>
bool BBRBM<T>::reconstructDataHidden(unsigned int iters)
{
  if(iters == 0) return false;
  
  math::matrix<T> Wt = W;
  Wt.transpose();
  
  while(iters > 0){
    v = Wt*h + a;
    
    // 1. visible units: calculates sigma(a_j)
    for(unsigned int j=0;j<(v.size()-0);j++){
      T aj = T(1.0)/(T(1.0) + math::exp(-v[j]));
      T r = rng.uniform();
      
      if(aj > r) v[j] = T(1.0); // discretization step
      else       v[j] = T(0.0);
    }
    
    iters--;
    if(iters <= 0) return true;
    
    h = W*v + b;
    
    // 1. hidden units: calculates sigma(a_j)
    for(unsigned int j=0;j<(h.size()-0);j++){
      T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
      T r = rng.uniform();
      
      if(aj > r) h[j] = T(1.0); // discretization step
      else       h[j] = T(0.0);
    }
    
    iters--;
    if(iters <= 0) return true;
  }
    
  return true;
}

template <typename T>
void BBRBM<T>::getParameters(math::matrix<T>& W,
			     math::vertex<T>& a,
			     math::vertex<T>& b) const
{
  W = this->W;
  a = this->a;
  b = this->b;
}


template <typename T>
bool BBRBM<T>::initializeWeights() // initialize weights to small values
{
  for(unsigned int j=0;j<W.ysize();j++){
    for(unsigned int i=0;i<W.xsize();i++){
      T r = rng.uniform();
      r   = T(0.02)*r - T(0.01);
      W(j,i) = r;
    }
  }
    
  for(unsigned int j=0;j<a.size();j++){
    T r = rng.uniform();
    r   = T(0.02)*r - T(0.01);
    a[j] = r;
  }
  
  for(unsigned int j=0;j<b.size();j++){
    T r = rng.uniform();
    r   = T(0.02)*r - T(0.01);
    b[j] = r;
  }

  return true;
}

// returns remaining reconstruction error in samples
template <typename T>
T BBRBM<T>::learnWeights(const std::vector< math::vertex<T> >& samples,
			 const unsigned int EPOCHS,
			 bool verbose)
{
  // implements traditional CD-k (k=10) learning algorithm,
  // which can be used as a reference point
  // and returns reconstruction error as the modelling error..
  
  const unsigned int CDk = 10; // CD-10 algorithm
  T lambda = T(0.01);          // initial learning rate
    
  math::matrix<T> PW(W);
  math::matrix<T> NW(W);
  math::vertex<T> pa(a), na(a);
  math::vertex<T> pb(b), nb(b);

  math::matrix<T> Wt = W;
  Wt.transpose();

  T latestError = reconstructionError(samples, 1000, a, b, W);
  
  if(verbose){
    double error = 0.0;
    math::convert(error, latestError);
    
    printf("%d/%d START. R: %f\n", 0, EPOCHS, error);
    fflush(stdout);
  }

  if(samples.size() <= 0) return latestError;
  if(samples[0].size() != v.size())
    return latestError; // silent failure

  const unsigned int NUMSAMPLES = 10;
  const unsigned int EPOCHSAMPLES = 1000;



  
  for(unsigned int e=0;e<EPOCHS;e++){

    for(unsigned int es=0;es<EPOCHSAMPLES;es++){

      // calculates positive gradient from NUMSAMPLES examples Pdata(v))
      PW.zero();
      pa.zero();
      pb.zero();
      
      // Pdata
      for(unsigned int i=0;i<NUMSAMPLES;i++){
	const unsigned int index = rng.rand() % samples.size();
	auto v = samples[index];
	
	auto h = W*v + b;
	
	// hidden units
	for(unsigned int j=0;j<h.size();j++)
	  h[j] = T(1.0)/(T(1.0) + math::exp(-h[j]));
	
	PW += h.outerproduct(v)/T(NUMSAMPLES);
	pa += v/T(NUMSAMPLES);
	pb += h/T(NUMSAMPLES);
      }
    

      // samples from Pmodel(v) approximatedly
      std::vector< math::vertex<T> > modelsamples; // model samples

      for(unsigned int i=0;i<NUMSAMPLES;i++)
      {
	const unsigned int index = rng.rand() % samples.size();
	auto v = samples[index];

	
	auto h = W*v + b;
	
	// 1. hidden units: calculates sigma(a_j)
	for(unsigned int j=0;j<(h.size()-0);j++){
	  T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	  T r = T(rand())/T(RAND_MAX);
	  
	  if(aj > r) h[j] = T(1.0); // discretization step
	  else       h[j] = T(0.0);
	}
	
	
	for(unsigned int l=0;l<CDk;l++){
	  v = Wt*h + a;
	  
	  // 1. visible units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(v.size()-0);j++){
	    T aj = T(1.0)/(T(1.0) + math::exp(-v[j]));
	    T r = T(rand())/T(RAND_MAX);
	    
	    if(aj > r) v[j] = T(1.0); // discretization step
	    else       v[j] = T(0.0);
	  }
	  
	  h = W*v + b;
	  
	  // 2. hidden units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(h.size()-0);j++){
	    T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	    T r = T(rand())/T(RAND_MAX);
	    
	    if(aj > r) h[j] = T(1.0); // discretization step
	    else       h[j] = T(0.0);
	  }
	  
	}

	modelsamples.push_back(v);
      }
    
      // calculates negative gradient by using samples from Pmodel(v)
      NW.zero();
      na.zero();
      nb.zero();
      
      // Pmodel
      for(unsigned int i=0;i<NUMSAMPLES;i++){
	const unsigned int index = rng.rand() % modelsamples.size();
	auto v = modelsamples[index];
	
	auto h = W*v + b;
	
	// hidden units
	for(unsigned int j=0;j<h.size();j++)
	  h[j] = T(1.0)/(T(1.0) + math::exp(-h[j]));
	
	NW += h.outerproduct(v)/T(NUMSAMPLES);
	na += v/T(NUMSAMPLES);
	nb += h/T(NUMSAMPLES);
      }
    
      // updates weights according to CD rule + ADAPTIVE LEARNING RATE
      {
	// W += lambda*(P - N);
	
	T coef1 = T(1.0);
	T coef2 = T(1.0/0.9);
	T coef3 = T(0.9);
	
	auto W1 = W + coef1*lambda*(PW - NW);
	auto W2 = W + coef2*lambda*(PW - NW);
	auto W3 = W + coef3*lambda*(PW - NW);
	
	auto a1 = a + coef1*lambda*(pa - na);
	auto a2 = a + coef2*lambda*(pa - na);
	auto a3 = a + coef3*lambda*(pa - na);
	
	auto b1 = b + coef1*lambda*(pb - nb);
	auto b2 = b + coef2*lambda*(pb - nb);
	auto b3 = b + coef3*lambda*(pb - nb);	
	
	T error1 = reconstructionError(samples, 10, a1, b1, W1);
	T error2 = reconstructionError(samples, 10, a2, b2, W2);
	T error3 = reconstructionError(samples, 10, a3, b3, W3);
	
	if(error2 < error1 && error2 < error3){
	  W = W2;
	  a = a2;
	  b = b2;
	  lambda *= coef2;
	}
	else if(error3 < error2 && error3 < error1){
	  W = W3;
	  a = a3;
	  b = b3;
	  lambda *= coef3;
	}
	else{
	  W = W1;
	  a = a1;
	  b = b1;
	  lambda *= coef1;
	}
      }
    
      Wt = W;
      Wt.transpose();
    }
  
    // calculates mean reconstruction error in samples and looks for convergence
    latestError = reconstructionError(samples, 1000, a, b, W);
    
    if(verbose){
      double error = 0.0;
      math::convert(error, latestError);
      
      printf("%d/%d EPOCH. R: %f\n", e, EPOCHS, error);
      fflush(stdout);
    }

  }
  
  
  return latestError;
  
  // alternatively we could look at delta parameters when trying to decide if we have converged
  // return (math::frobenius_norm(originalW)/originalW.size()); 
}


template <typename T>
T BBRBM<T>::reconstructionError(const std::vector< math::vertex<T> >& samples,
				unsigned int N, // number of samples to use from samples to estimate reconstruction error
				const math::vertex<T>& a,
				const math::vertex<T>& b,			
				const math::matrix<T>& W) const throw() // weight matrix (parameters) to use
{
  T error = T(0.0);

  for(unsigned int n=0;n<N;n++){
    const auto& s = samples[n];

    error += reconstructionError(s, a, b, W) / T(N);
  }

  return error;
}


// weight matrix (parameters) to use
template <typename T>
T BBRBM<T>::reconstructionError(const math::vertex<T>& s,
				const math::vertex<T>& a,
				const math::vertex<T>& b,
				const math::matrix<T>& W) const throw()
{
  try{
    math::matrix<T> Wt = W;
    Wt.transpose();
    
    math::vertex<T> v(this->v);
    math::vertex<T> h(this->h);

    v = s;

    {
      h = W*v + b;
      
      // 1. hidden units: calculates sigma(a_j)
      for(unsigned int j=0;j<(h.size()-0);j++){
	T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	T r = rng.uniform();
	
	if(aj > r) h[j] = T(1.0); // discretization step
	else       h[j] = T(0.0);
      }

      v = Wt*h + a;
      
      // 1. visible units: calculates sigma(a_j)
      for(unsigned int j=0;j<(v.size()-0);j++){
	T aj = T(1.0)/(T(1.0) + math::exp(-v[j]));
	T r = rng.uniform();
	
	if(aj > r) v[j] = T(1.0); // discretization step
	else       v[j] = T(0.0);
      }
      
    }

    auto delta = v - s;
    return delta.norm(); // reconstruction error for this vector
  }
  catch(std::exception& e){
    std::cout << "Uncaught exception (reconstructionError()): " << e.what() << std::endl;
    return T(INFINITY); // incorrect data - infinity error
  }
}


////////////////////////////////////////////////////////////
// TODO implement me! (implement LBFGS_BBRBM after gradient descent one works)

template <typename T>
bool BBRBM<T>::setUData(const std::vector< math::vertex<T> >& samples)
{
  return false;
}

template <typename T>
unsigned int BBRBM<T>::qsize() const throw() // size of q vector q = [vec(W)]
{
  return (W.xsize()*W.ysize());
}
  
// converts (W) parameters into q vector
template <typename T>
bool BBRBM<T>::convertParametersToQ(const math::matrix<T>& W, math::vertex<T>& q) const
{
  return false;
}
  
// converts q vector into parameters (W, a, b)
template <typename T>
bool BBRBM<T>::convertQToParameters(const math::vertex<T>& q, math::matrix<T>& W) const
{
  return false;
}
  
// sets (W) parameters according to q vector
template <typename T>
bool BBRBM<T>::setParametersQ(const math::vertex<T>& q)
{
  return false;
}

template <typename T>
bool BBRBM<T>::getParametersQ(math::vertex<T>& q) const
{
  return false;
}
  
template <typename T>
T BBRBM<T>::U(const math::vertex<T>& q) const throw()
{
  return T(-INFINITY);
}

template <typename T>
math::vertex<T> BBRBM<T>::Ugrad(const math::vertex<T>& q) throw()
{
  math::vertex<T> grad;
  grad.resize(this->qsize());
  grad.zero();

  return grad;
}

////////////////////////////////////////////////////////////

// load & saves RBM data from/to file
template <typename T>
bool BBRBM<T>::load(const std::string& filename) throw()
{
  return false;
}

template <typename T>
bool BBRBM<T>::save(const std::string& filename) const throw()
{
  return false;
}

template class BBRBM< float >;
template class BBRBM< double >;
template class BBRBM< math::blas_real<float> >;
template class BBRBM< math::blas_real<double> >;

} /* namespace whiteice */
