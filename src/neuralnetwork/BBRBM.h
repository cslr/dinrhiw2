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
#include "RNG.h"


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
  
  
  bool reconstructData(unsigned int iters = 2); // 2 to v->h->h
  bool reconstructData(std::vector< math::vertex<T> >& samples,
		       unsigned int iters = 1);
  bool reconstructDataHidden(unsigned int iters = 2);
  
  void getParameters(math::matrix<T>& W, math::vertex<T>& a, math::vertex<T>& b) const;
  
  bool initializeWeights(); // initialize weights to small values
  
  // calculates single epoch for updating weights using CD-1 and
  // returns reconstruction error
  T learnWeights(const std::vector< math::vertex<T> >& samples,
		 const unsigned int EPOCHS=1,
		 bool verbose = false);

  T reconstructionError(const std::vector< math::vertex<T> >& samples,
			unsigned int N, // number of samples to use from samples to estimate reconstruction error
			const math::vertex<T>& a,
			const math::vertex<T>& b,			
			const math::matrix<T>& W) const throw(); // weight matrix (parameters) to use

  T reconstructionError(const std::vector< math::vertex<T> >& samples,
			unsigned int N) const throw() // number of samples to use from samples to estimate reconstruction error
  
  { return reconstructionError(samples, N, this->a, this->b, this->W); }

  T reconstructionError(const math::vertex<T>& s,
			const math::vertex<T>& a,
			const math::vertex<T>& b,
			const math::matrix<T>& W) const throw(); // weight matrix (parameters) to use

  ////////////////////////////////////////////////////////////
  // U(q) functions used to maximize P(v|data, q) ~ exp(-U(q))
  //       rbm parameters q

  bool setUData(const std::vector< math::vertex<T> >& samples);

  unsigned int qsize() const throw(); // size of q vector q = [vec(W)]
  
  // converts q vector into parameters (W, a, b)
  bool convertParametersToQ(const math::matrix<T>& W, const math::vertex<T>& a, const math::vertex<T>& b,
			    math::vertex<T>& q) const;

  // converts q vector into parameters (W, a, b)
  bool convertQToParameters(const math::vertex<T>& q, math::matrix<T>& W, math::vertex<T>& a, math::vertex<T>& b) const;
  
  // sets (W) parameters according to q vector
  bool setParametersQ(const math::vertex<T>& q);
  bool getParametersQ(math::vertex<T>& q) const;

  // keeps parameters within sane levels (clips overly large parameters and NaNs)
  void safebox(math::vertex<T>& a, math::vertex<T>& b, math::matrix<T>& W) const;
  
  T U(const math::vertex<T>& q) const throw();
  math::vertex<T> Ugrad(const math::vertex<T>& q) throw();

  ////////////////////////////////////////////////////////////
  
  // load & saves RBM data from/to file
  
  bool load(const std::string& filename) throw();
  bool save(const std::string& filename) const throw();
  
 private:
  math::vertex<T> h, v;
  
  math::matrix<T> W;
  math::vertex<T> a;
  math::vertex<T> b;

  std::vector< math::vertex<T> > Usamples;

  RNG<T> rng;
};

 
 extern template class BBRBM< float >;
 extern template class BBRBM< double >;
 extern template class BBRBM< math::blas_real<float> >;
 extern template class BBRBM< math::blas_real<double> >;
 
} /* namespace whiteice */

#endif /* NEURALNETWORK_BBRBM_H_ */
