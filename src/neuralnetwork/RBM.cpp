
#include "RBM.h"

namespace whiteice
{
  
  // creates 1x1 network, used to load some useful network later
  template <typename T>
  RBM<T>::RBM()
  {
    v.resize(1);
    h.resize(1);
    W.resize(1,1);
    W(0,0) = T(0.0);
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
    
    v.resize(visible);
    h.resize(hidden);
    W.resize(hidden, visible);
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
    return v;
  }
  
  template <typename T>
  bool RBM<T>::setVisible(const math::vertex<T>& v)
  {
    if(this->v.size() != v.size())
      return false;
    
    this->v = v;
    return true;
  }
  
  template <typename T>
  math::vertex<T> RBM<T>::getHidden() const
  {
    return h;
  }

  template <typename T>
  bool RBM<T>::setHidden(const math::vertex<T>& h)
  {
    if(this->h.size() != h.size())
      return false;
    
    this->h = h;
    return true;
  }
  
  
  template <typename T>
  T RBM<T>::learnWeights(const std::vector< math::vertex<T> >& samples)
  {
    return T(1000.0f); // not yet implemented
  }
  
  
};
