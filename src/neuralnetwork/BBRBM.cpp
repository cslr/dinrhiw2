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
  
  a.resize(1);
  b.resize(1);
  W.resize(1,1);

  initializeWeights();
}

template <typename T>
BBRBM<T>::BBRBM(const BBRBM<T>& rbm)
{
  this->v = rbm.v;
  this->h = rbm.h;

  this->a = rbm.a;
  this->b = rbm.b;
  this->W = rbm.W;
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

  this->a = rbm.a;
  this->b = rbm.b;
  this->W = rbm.W;

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
  return false;
}

template <typename T>
bool BBRBM<T>::reconstructData(std::vector< math::vertex<T> >& samples, unsigned int iters)
{
  return false;
}

template <typename T>
bool BBRBM<T>::reconstructDataHidden(unsigned int iters)
{
  return false;
}

template <typename T>
void BBRBM<T>::getParameters(math::matrix<T>& W, math::vertex<T>& a, math::vertex<T>& b) const
{

}


template <typename T>
bool BBRBM<T>::initializeWeights() // initialize weights to small values
{
  return false;
}

// calculates single epoch for updating weights using CD-1 and
// returns reconstruction error
// EPOCHS control quality of the solution, 1 epoch goes through data once
// but higher number of EPOCHS mean data calculations can take longer (higher quality)
template <typename T>
T BBRBM<T>::learnWeights(const std::vector< math::vertex<T> >& samples,
			 const unsigned int EPOCHS, bool verbose)
{
  
  return T(INFINITY);
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
