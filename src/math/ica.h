/*
 * independent components analysis code
 */

#ifndef ica_h
#define ica_h

#include "vertex.h"
#include "matrix.h"

namespace whiteice
{
  namespace math
  {
    
    // solves independent components from the data and saves
    // dependacy removal matrix to W. Uses deflate method.
    template <typename T>
      bool ica(const matrix<T>& D, matrix<T>& W, bool verbose = false) throw();

    template <typename T>
      bool ica(const std::vector< math::vertex<T> >& data, matrix<T>& W, bool verbose = false) throw();


    
    // ICA TODO: reordering and recalculating of ICs which have been computed
    // after first non-covergent IC. This way all ICs which converge will be reliable.
    // non-converged ICs should be checked for gassianity and gaussian ones should
    // grouped and PCAed so that the gaussian subspace is also solved as well as possible
    
    extern template bool ica< blas_real<float> >
      (const matrix< blas_real<float> >& D, matrix< blas_real<float> >& W, bool verbose) throw();
    extern template bool ica< blas_real<double> >
      (const matrix< blas_real<double> >& D, matrix< blas_real<double> >& W, bool verbose) throw();
    extern template bool ica< float >
      (const matrix<float>& D, matrix<float>& W, bool verbose) throw();
    extern template bool ica< double >
      (const matrix<double>& D, matrix<double>& W, bool verbose) throw();

    
    extern template bool ica< blas_real<float> >
      (const std::vector< math::vertex< blas_real<float> > >& data, matrix< blas_real<float> >& W, bool verbose) throw();
    extern template bool ica< blas_real<double> >
      (const std::vector< math::vertex< blas_real<double> > >& data, matrix< blas_real<double> >& W, bool verbose) throw();
    extern template bool ica< float >
      (const std::vector< math::vertex<float> >& data, matrix<float>& W, bool verbose) throw();
    extern template bool ica< double >
      (const std::vector< math::vertex<double> >& data, matrix<double>& W, bool verbose) throw();    
    
  };
};


#endif
