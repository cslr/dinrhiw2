
#ifndef stackedRBM_pretraining_h
#define stackedRBM_pretraining_h

#include "DBN.h"
#include "GBRBM.h"
#include "BBRBM.h"
#include "nnetwork.h"
#include "dataset.h"

namespace whiteice
{
  // binary = true pure BBRBM network,
  // binary = false GBRBM network input layer
  // verbose = 0 no output,
  // verbose = 1 stdout output,
  // verbose = 2 logging output
  template <typename T>
  bool deep_pretrain_nnetwork(whiteice::nnetwork<T>*& nn,
			      const whiteice::dataset<T>& data,
			      const bool binary,
			      const int verbose,
			      const bool* running = NULL);


  extern template bool deep_pretrain_nnetwork<float>
    (whiteice::nnetwork<float>*& nn,
     const whiteice::dataset<float>& data,
     const bool binary,
     const int verbose,
     const bool* running);

  extern template bool deep_pretrain_nnetwork<double>
    (whiteice::nnetwork<double>*& nn,
     const whiteice::dataset<double>& data,
     const bool binary,
     const int verbose,
     const bool* running);

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
  
};


#endif

