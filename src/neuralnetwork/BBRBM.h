/*
 * BBRBM.h
 *
 *  Created on: 22.6.2015
 *      Author: Tomas
 */

#ifndef NEURALNETWORK_BBRBM_H_
#define NEURALNETWORK_BBRBM_H_

#include "vertex.h"
#include "matrix.h"


namespace whiteice {

/**
 * Standard Bernoulli-Bernoulli RBM
 */
template <typename T = math::blas_real<float> >
class BBRBM {
  public:
  
  BBRBM();
  BBRBM(const BBRBM<T>& rbm);
  
  // creates 2-layer: V * H network
  BBRBM(unsigned int visible, unsigned int hidden) throw(std::invalid_argument);
  
  virtual ~BBRBM();
  
  
  BBRBM<T>& operator=(const BBRBM<T>& rbm);
  
  bool resize(unsigned int visible, unsigned int hidden);
  
  ////////////////////////////////////////////////////////////
  
  void getVisible(math::vertex<T>& v) const;
  bool setVisible(const math::vertex<T>& v);
  
  void getHidden(math::vertex<T>& h) const;
  bool setHidden(const math::vertex<T>& h);
  
  
  bool reconstructData(unsigned int iters = 1);
  bool reconstructData(std::vector< math::vertex<T> >& samples, unsigned int iters = 1);
  bool reconstructDataHidden(unsigned int iters = 1);
  
  void getParameters(math::matrix<T>& W, math::vertex<T>& a, math::vertex<T>& b) const;
  
  bool initializeWeights(); // initialize weights to small values
  
  // calculates single epoch for updating weights using CD-1 and
  // returns reconstruction error
  // EPOCHS control quality of the solution, 1 epoch goes through data once
  // but higher number of EPOCHS mean data calculations can take longer (higher quality)
  T learnWeights(const std::vector< math::vertex<T> >& samples,
		 const unsigned int EPOCHS=1,
		 bool verbose = false, bool learnVariance = false);
  
  ////////////////////////////////////////////////////////////
  
  // load & saves RBM data from/to file
  
  bool load(const std::string& filename) throw();
  bool save(const std::string& filename) const throw();
  
 private:
  math::vertex<T> h, v;
  
  math::vertex<T> a, b;
  math::matrix<T> W;
  
  
};

 
 extern template class BBRBM< float >;
 extern template class BBRBM< double >;
 extern template class BBRBM< math::blas_real<float> >;
 extern template class BBRBM< math::blas_real<double> >;
 
} /* namespace whiteice */

#endif /* NEURALNETWORK_BBRBM_H_ */
