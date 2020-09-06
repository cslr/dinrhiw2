
#ifndef stackedRBM_pretraining_h
#define stackedRBM_pretraining_h

#include "DBN.h"
#include "GBRBM.h"
#include "BBRBM.h"
#include "nnetwork.h"
#include "dataset.h"

namespace whiteice
{
  // deep pretraining of nnetwork
  // (sigmoid nonlins + pureLinear ouput layer)
  //
  // dataset must contain both input and output datasets
  // (outputs are used to optimize final linear output layer)
  // 
  // NOTE: neural network pointer is replaced by other neural network pointer allocated with new.
  //
  // binary = true pure BBRBM network,
  // binary = false GBRBM network input layer + BBRBM hidden layers
  // verbose = 0 no output,
  // verbose = 1 stdout output,
  // verbose = 2 logging output
  template <typename T>
  bool deep_pretrain_nnetwork(whiteice::nnetwork<T>*& nn,
			      const whiteice::dataset<T>& data,
			      const bool binary,
			      const int verbose,
			      const bool* running = NULL);


  // deep pretraining of nnetwork
  // (sigmoid nonlins AND sigmoidal ouput layer)
  // 
  // dataset need only contain input cluster (no outputs)
  // (outputs are not optimized to any output)
  // 
  // binary = true pure BBRBM network,
  // binary = false GBRBM network input layer + BBRBM hidden layers
  // verbose = 0 no output,
  // verbose = 1 stdout output,
  // verbose = 2 logging output
  template <typename T>
  bool deep_pretrain_nnetwork_full_sigmoid(whiteice::nnetwork<T>*& nn,
					   const whiteice::dataset<T>& data,
					   const bool binary,
					   const int verbose,
					   const bool* running = NULL);


  
  extern template bool deep_pretrain_nnetwork< math::blas_real<float> >
    (whiteice::nnetwork< math::blas_real<float> >*& nn,
     const whiteice::dataset< math::blas_real<float> >& data,
     const bool binary,
     const int verbose,
     const bool* running);
  
  extern template bool deep_pretrain_nnetwork< math::blas_real<double> >
    (whiteice::nnetwork< math::blas_real<double> >*& nn,
     const whiteice::dataset< math::blas_real<double> >& data,
     const bool binary,
     const int verbose,
     const bool* running);



  extern template bool deep_pretrain_nnetwork_full_sigmoid< math::blas_real<float> >
    (whiteice::nnetwork< math::blas_real<float> >*& nn,
     const whiteice::dataset< math::blas_real<float> >& data,
     const bool binary,
     const int verbose,
     const bool* running);
  
  extern template bool deep_pretrain_nnetwork_full_sigmoid< math::blas_real<double> >
    (whiteice::nnetwork< math::blas_real<double> >*& nn,
     const whiteice::dataset< math::blas_real<double> >& data,
     const bool binary,
     const int verbose,
     const bool* running);
  
};


#endif

