
#include "RBM.h"
#include "norms.h"

namespace whiteice
{
  
  // creates 1x1 network, used to load some useful network later
  template <typename T>
  RBM<T>::RBM()
  {
    v.resize(1);
    h.resize(1);
    W.resize(1,1);
    
    v[0]   = randomValue();
    h[0]   = randomValue();
    W(0,0) = randomValue();
  }
  
  template <typename T>
  RBM<T>::RBM(const RBM<T>& rbm)
  {
    this->v = rbm.v;
    this->h = rbm.h;
    this->W = rbm.W;
  }
  
  
  // creates 2-layer: V * H network
  template <typename T>
  RBM<T>::RBM(unsigned int visible, unsigned int hidden)
    throw(std::invalid_argument)
  {
    if(visible == 0 || hidden == 0)
      throw std::invalid_argument("invalid network architecture");
    
    // the last term is always constant: 1 (one)
    v.resize(visible + 1);
    h.resize(hidden + 1);
    W.resize(hidden + 1, visible + 1);
    
    for(unsigned int j=0;j<hidden;j++)
      for(unsigned int i=0;i<visible;i++)
	W(j,i) = randomValue();
  }
  
  template <typename T>
  RBM<T>::~RBM()
  {
    // nothing to do
  }
  
  template <typename T>  
  RBM<T>& RBM<T>::operator=(const RBM<T>& rbm)
  {
    this->v = rbm.v;
    this->h = rbm.h;
    this->W = rbm.W;
    
    return (*this);
  }
  
  template <typename T>
  math::vertex<T> RBM<T>::getVisible() const
  {
    math::vertex<T> t = v;
    t.resize(v.size() - 1);
    
    return t;
  }
  
  template <typename T>
  bool RBM<T>::setVisible(const math::vertex<T>& v)
  {
    if(this->v.size() != (v.size() + 1))
      return false;
    
    for(unsigned int j=0;j<v.size();j++)
      this->v[j] = v[j];
    
    this->v[v.size()] = T(1.0);
    
    return true;
  }
  
  template <typename T>
  math::vertex<T> RBM<T>::getHidden() const
  {
    math::vertex<T> t = h;
    t.resize(h.size() - 1);
    
    return t;
  }
  
  
  template <typename T>
  bool RBM<T>::setHidden(const math::vertex<T>& h)
  {
    if(this->h.size() != (h.size() + 1))
      return false;
    
    for(unsigned int j=0;j<h.size();j++)
      this->h[j] = h[j];
    
    this->h[h.size()] = T(1.0);
    
    return true;
  }
  
  
  template <typename T>
  T RBM<T>::learnWeights(const std::vector< math::vertex<T> >& samples)
  {
    const unsigned int L = 1; // CD-1 algorithm
    const T lambda = T(0.01);
    
    math::matrix<T> P(W);
    math::matrix<T> N(W);
    math::matrix<T> originalW = W;
    
    for(unsigned int i=0;i<samples.size();i++){
      const unsigned int index = rand() % samples.size();
      
      v = samples[index];
      
      {
	h = W*v;
	
	// 1. hidden units: calculates sigma(a_j)
	for(unsigned int j=0;j<(h.size()-1);j++){
	  T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	  T r = T(rand())/T(RAND_MAX);
	  
	  if(aj > r) h[j] = T(1.0); // discretization step
	  else       h[j] = T(0.0);
	}
	
	h[h.size()-1] = T(1.0);
	
	for(unsigned int y=0;y<h.size();y++)
	  for(unsigned int x=0;x<v.size();x++)
	    P(y,x) = h[y]*v[x];
	
	math::matrix<T> Wt = W.transpose();
	
	
	for(unsigned int l=0;l<L;l++){
	  v = Wt*h;
	  
	  // 1. visible units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(v.size()-1);j++){
	    T aj = T(1.0)/(T(1.0) + math::exp(-v[j]));
	    T r = T(rand())/T(RAND_MAX);
	    
	    if(aj > r) v[j] = T(1.0); // discretization step
	    else       v[j] = T(0.0);
	  }
	  
	  v[v.size()-1] = T(1.0);
	  
	  h = W*v;
	  
	  // 2. hidden units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(h.size()-1);j++){
	    T aj = T(1.0)/(T(1.0) + math::exp(-h[j]));
	    T r = T(rand())/T(RAND_MAX);
	    
	    if(aj > r) h[j] = T(1.0); // discretization step
	    else       h[j] = T(0.0);
	  }
	  
	  h[h.size()-1] = T(1.0);
	  
	}
	
	for(unsigned int y=0;y<h.size();y++)
	  for(unsigned int x=0;x<v.size();x++)
	    N(y,x) = h[y]*v[x];
      }
      
      // updates weights according to CD rule
      W = W * lambda*(P - N);
    }
    
    
    originalW -= W;
    
    return math::frobenius_norm(originalW);
  }
  
  
  
  template <typename T>
  T RBM<T>::randomValue()
  {
    T r = T(rand())/T(RAND_MAX);
    
    r = T(0.02)*r - T(0.01);
    
    return r;
  }
  
  
  
  
  template class RBM< float >;
  template class RBM< double >;  
  template class RBM< math::blas_real<float> >;
  template class RBM< math::blas_real<double> >;
  
};
