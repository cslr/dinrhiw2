/*
 * BBRBM.cpp
 *
 *  Created on: 22.6.2015
 *      Author: Tomas Ukkonen
 */

#include "BBRBM.h"
#include "dataset.h"
#include "linear_ETA.h"

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

  this->Usamples = rbm.Usamples;
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
  this->Usamples = rbm.Usamples;

  return (*this);
}

template <typename T>
bool BBRBM<T>::operator==(const BBRBM<T>& rbm) const
{
  if(this->v != rbm.v) return false;
  if(this->h != rbm.h) return false;

  if(this->W != rbm.W) return false;
  if(this->a != rbm.a) return false;
  if(this->b != rbm.b) return false;
  if(this->Usamples != rbm.Usamples) return false;

  return true;
}

template <typename T>
bool BBRBM<T>::operator!=(const BBRBM<T>& rbm) const
{
  if(this->v == rbm.v && this->h == rbm.h && 
     this->W == rbm.W && this->a == rbm.a &&
     this->b == rbm.b && this->Usamples == rbm.Usamples)
    return false;

  return true;
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
unsigned int BBRBM<T>::getVisibleNodes() const
{
  return v.size();
}

template <typename T>
unsigned int BBRBM<T>::getHiddenNodes() const
{
  return h.size();
}
  
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
math::vertex<T> BBRBM<T>::getBValue() const
{
  return b;
}

template <typename T>
math::vertex<T> BBRBM<T>::getAValue() const
{
  return a;
}

template <typename T>
math::matrix<T> BBRBM<T>::getWeights() const
{
  return W;
}

template <typename T>
bool BBRBM<T>::reconstructData(unsigned int iters)
{
  if(iters == 0) return false;
  
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

    v = h*W + a;
    
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

#if 0
  math::matrix<T> Wt = W; // OPTIMIZE ME/FIX ME creates Wt matrix
  Wt.transpose();
#endif
  
  while(iters > 0){
#if 0
    v = Wt*h + a;
#endif
    v = h*W + a;
    
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
  // implements traditional CD-k (k=2) learning algorithm,
  // which can be used as a reference point
  // and returns reconstruction error as the modelling error..
  
  const unsigned int CDk = 2;  // CD-k algorithm (was 10!)
  T lambda = T(0.01);          // initial learning rate
    
  math::matrix<T> PW(W);
  math::matrix<T> NW(W);
  math::vertex<T> pa(a), na(a);
  math::vertex<T> pb(b), nb(b);

#if 0
  math::matrix<T> Wt = W;
  Wt.transpose();
#endif

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

    whiteice::linear_ETA<double> eta;
    eta.start(0.0, (double)EPOCHSAMPLES);
    
    for(unsigned int es=0;es<EPOCHSAMPLES;es++){

      eta.update(es);

      if(verbose){
	printf("\r                                                                             \r");
	printf("EPOCH SAMPLES %d/%d [ETA: %.2f minutes]", es, EPOCHSAMPLES, eta.estimate()/60.0);
	fflush(stdout);
      }

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
	
	// PW += h.outerproduct(v)/T(NUMSAMPLES);
	T scaling = T(1.0)/T(NUMSAMPLES);
	addouterproduct(PW, scaling, h, v);
	
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
	  T r = rng.uniform();
	  
	  if(aj > r) h[j] = T(1.0); // discretization step
	  else       h[j] = T(0.0);
	}
	
	
	for(unsigned int l=0;l<CDk;l++){
#if 0
	  v = Wt*h + a;
#endif
	  v = h*W + a;
	  
	  // 1. visible units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(v.size()-0);j++){
	    T aj = T(1.0)/(T(1.0) + math::exp(-v[j]));
	    T r = rng.uniform();
	    
	    if(aj > r) v[j] = T(1.0); // discretization step
	    else       v[j] = T(0.0);
	  }
	  
	  h = W*v + b;
	  
	  // 2. hidden units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(h.size()-0);j++){
	    T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	    T r = rng.uniform();
	    
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
	
	// NW += h.outerproduct(v)/T(NUMSAMPLES);
	T scaling = T(1.0)/T(NUMSAMPLES);
	addouterproduct(NW, scaling, h, v);
	
	
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

#if 0
      Wt = W;
      Wt.transpose();
#endif
    }
  
    // calculates mean reconstruction error in samples and looks for convergence
    latestError = reconstructionError(samples, 1000, a, b, W);
    
    if(verbose){
      double error = 0.0;
      math::convert(error, latestError);

      printf("\n");
      printf("%d/%d EPOCH. R: %f\n", e+1, EPOCHS, error);
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
    const unsigned int index = rng.rand() % samples.size();
    const auto& s = samples[index];

    error += reconstructionError(s, a, b, W) / T(N*s.size());
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
#if 0
    math::matrix<T> Wt = W;
    Wt.transpose();
#endif
    
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

#if 0
      v = Wt*h + a;
#endif
      v = h*W + a;
      
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
  if(samples.size() == 0) return false;
  if(samples[0].size() != a.size()) return false;
  
  this->Usamples = samples;
  
  return true;
}

template <typename T>
unsigned int BBRBM<T>::qsize() const throw() // size of q vector q = [vec(W)]
{
  return (a.size() + b.size() + W.xsize()*W.ysize());
}
  
// converts (W) parameters into q vector
template <typename T>
bool BBRBM<T>::convertParametersToQ(const math::matrix<T>& W,
				    const math::vertex<T>& a,
				    const math::vertex<T>& b, 
				    math::vertex<T>& q) const
{
  q.resize(a.size() + b.size() + W.ysize()*W.xsize());

  q.write_subvertex(a, 0);
  q.write_subvertex(b, a.size());
  W.save_to_vertex(q, a.size()+b.size());

  return true;
}
  
// converts q vector into parameters (W, a, b)
template <typename T>
bool BBRBM<T>::convertQToParameters(const math::vertex<T>& q,
				    math::matrix<T>& W,
				    math::vertex<T>& a,
				    math::vertex<T>& b) const 
				    
{
  a.resize(this->a.size());
  b.resize(this->b.size());
  W.resize(this->W.ysize(), this->W.xsize());
  
  if(q.size() != (a.size()+b.size()+W.ysize()*W.xsize()))
    return false;
  
  try{
    q.subvertex(a, 0, a.size());
    q.subvertex(b, a.size(), b.size());
    math::vertex<T> w(W.ysize()*W.xsize());
    q.subvertex(w, (a.size()+b.size()), w.size());
    W.load_from_vertex(w);
    
    return true;
  }
  catch(std::exception& e){
    return false;
  }
}
  
// sets (W) parameters according to q vector
template <typename T>
bool BBRBM<T>::setParametersQ(const math::vertex<T>& q)
{
  return convertQToParameters(q, W, a, b);
}

template <typename T>
bool BBRBM<T>::getParametersQ(math::vertex<T>& q) const
{
  return convertParametersToQ(W, a, b, q);
}

// keeps parameters within sane levels
// (clips overly large parameters and NaNs)
template <typename T>
void BBRBM<T>::safebox(math::vertex<T>& a, math::vertex<T>& b,
		       math::matrix<T>& W) const
{
  for(unsigned int i=0;i<a.size();i++){
    if(whiteice::math::isnan(a[i])) a[i] = T(0.0); //printf("anan"); }
    if(a[i] < T(-10e10)) a[i] = T(-10e10); //printf("aclip"); }
    if(a[i] > T(+10e10)) a[i] = T(+10e10); //printf("aclip"); }
  }

  for(unsigned int i=0;i<b.size();i++){
    if(whiteice::math::isnan(b[i])) b[i] = T(0.0); //printf("bnan"); }
    if(b[i] < T(-10e10)) b[i] = T(-10e10); //printf("bclip"); }
    if(b[i] > T(+10e10)) b[i] = T(+10e10); //printf("bclip"); }
  }

  for(unsigned int j=0;j<W.ysize();j++){
    for(unsigned int i=0;i<W.xsize();i++){
      if(whiteice::math::isnan(W(j,i))) W(j,i) = T(0.0); //printf("Wnan"); }
      if(W(j,i) < T(-10e10)) W(j,i) = T(-10e10); //printf("Wclip"); }
      if(W(j,i) > T(+10e10)) W(j,i) = T(+10e10); //printf("Wclip"); }
    }
  }
}
  
template <typename T>
T BBRBM<T>::U(const math::vertex<T>& q) const throw()
{
  math::vertex<T> a;
  math::vertex<T> b;
  math::matrix<T> W;

  convertQToParameters(q, W, a, b);
  safebox(a, b, W);
  // currently uses only reconstruction error to estimate U(p)
  return reconstructionError(Usamples, 1000, a, b, W);
}

template <typename T>
math::vertex<T> BBRBM<T>::Ugrad(const math::vertex<T>& q) throw()
{
  math::vertex<T> a;
  math::vertex<T> b;
  math::matrix<T> W;
  
  convertQToParameters(q, W, a, b);
  safebox(a, b, W);

  // calculates gradient vector
  math::vertex<T> ga(a);
  math::vertex<T> gb(b);
  math::matrix<T> gW(W);

  math::vertex<T> pa(a);
  math::vertex<T> pb(b);
  math::matrix<T> PW(W);

  math::vertex<T> na(a);
  math::vertex<T> nb(b);
  math::matrix<T> NW(W);

  const unsigned int NUMSAMPLES = 100;
  const unsigned int CDk = 1;

  {
    // calculates positive gradient from NUMSAMPLES examples Pdata(v))
    PW.zero();
    pa.zero();
    pb.zero();
    
    // Pdata
    for(unsigned int i=0;i<NUMSAMPLES;i++){
      const unsigned int index = rng.rand() % Usamples.size();
      auto v = Usamples[index];
      
      auto h = W*v + b;
      
      // hidden units
      for(unsigned int j=0;j<h.size();j++)
	h[j] = T(1.0)/(T(1.0) + math::exp(-h[j]));
      
      // PW += h.outerproduct(v)/T(NUMSAMPLES);
      T scaling = T(1.0)/T(NUMSAMPLES);
      addouterproduct(PW, scaling, h, v);
	
      pa += v/T(NUMSAMPLES);
      pb += h/T(NUMSAMPLES);
    }
    

    // samples from Pmodel(v) approximatedly
    std::vector< math::vertex<T> > modelsamples; // model samples
    
    for(unsigned int i=0;i<NUMSAMPLES;i++)
    {
      const unsigned int index = rng.rand() % Usamples.size();
      auto v = Usamples[index];
      
      
      auto h = W*v + b;
      
      // 1. hidden units: calculates sigma(a_j)
      for(unsigned int j=0;j<(h.size()-0);j++){
	T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	T r = rng.uniform();
	
	if(aj > r) h[j] = T(1.0); // discretization step
	else       h[j] = T(0.0);
      }
	
      
      for(unsigned int l=0;l<CDk;l++){
	v = h*W + a;
	
	// 1. visible units: calculates sigma(a_j)
	for(unsigned int j=0;j<(v.size()-0);j++){
	  T aj = T(1.0)/(T(1.0) + math::exp(-v[j]));
	  T r = rng.uniform();
	  
	  if(aj > r) v[j] = T(1.0); // discretization step
	  else       v[j] = T(0.0);
	}
	
	h = W*v + b;
	
	// 2. hidden units: calculates sigma(a_j)
	for(unsigned int j=0;j<(h.size()-0);j++){
	  T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	  T r = rng.uniform();
	  
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
      auto v = modelsamples[i];
      
      auto h = W*v + b;
      
      // hidden units
      for(unsigned int j=0;j<h.size();j++)
	h[j] = T(1.0)/(T(1.0) + math::exp(-h[j]));
      
      // NW += h.outerproduct(v)/T(NUMSAMPLES);
      T scaling = T(1.0)/T(NUMSAMPLES);
      addouterproduct(NW, scaling, h, v);
      
      na += v/T(NUMSAMPLES);
      nb += h/T(NUMSAMPLES);
    }

    ga = (pa - na);
    gb = (pb - nb);
    gW = (PW - NW);
  }

  math::vertex<T> grad;
  grad.resize(this->qsize());
  
  convertParametersToQ(gW, ga, gb, grad);
  
  return grad;
}

////////////////////////////////////////////////////////////

// load & saves RBM data from/to file
template <typename T>
bool BBRBM<T>::load(const std::string& filename) throw()
{
    whiteice::dataset<T> file;

  if(file.load(filename) == false)
    return false;

  if(file.getNumberOfClusters() != 7)
    return false;

  std::vector<std::string> names;
  if(file.getClusterNames(names) == false) return false;

  bool found = false;
  for(auto& n : names)
    if(n == "whiteice::BBRBM file"){
      found = true;
      break;
    }

  if(found == false)
    return false; // unknown filetype
				  
  // tries to load data..
  std::vector< math::vertex<T> > data;

  if(file.getData(0, data) == false) return false;
  a = data[0];
  if(file.getData(1, data) == false) return false;
  b = data[0];
  if(file.getData(2, data) == false) return false;
  W.resize(b.size(), a.size());
  if(W.load_from_vertex(data[0]) == false)
    return false;
  if(file.getData(3, data) == false) return false;
  h = data[0];
  if(file.getData(4, data) == false) return false;
  v = data[0];
  if(file.getData(5, data) == false) return false;
  if(data.size() > 0)
    Usamples = data;
  else
    Usamples.clear();

  // some sanity checks..
  if(a.size() != W.ysize()) return false;
  if(b.size() != W.xsize()) return false;
  if(a.size() != v.size())  return false;
  if(b.size() != h.size())  return false;
  
  return true;
}

template <typename T>
bool BBRBM<T>::save(const std::string& filename) const throw()
{
  whiteice::dataset<T> file;

  file.createCluster("a", a.size());
  if(file.add(0, a) == false) return false;
  file.createCluster("b", b.size());
  if(file.add(1, b) == false) return false;
  file.createCluster("W", W.size());

  math::vertex<T> vecW(W.xsize()*W.ysize());
  W.save_to_vertex(vecW);
  
  if(file.add(2, vecW) == false) return false;

  if(file.createCluster("h", h.size()) == false) return false;
  file.add(3, h);
  if(file.createCluster("v", v.size()) == false) return false;
  file.add(4, v);

  if(Usamples.size() > 0){
    file.createCluster("Usamples", Usamples[0].size());
    if(file.add(5, Usamples) == false) return false;
  }
  else{
    file.createCluster("Usamples", 0);
  }

  file.createCluster("whiteice::BBRBM file", 1);
  
  return file.save(filename);
}

template class BBRBM< float >;
template class BBRBM< double >;
template class BBRBM< math::blas_real<float> >;
template class BBRBM< math::blas_real<double> >;

} /* namespace whiteice */
