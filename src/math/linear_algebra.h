/*
 * linear algebra algorithms
 */

#ifndef linear_algebra_h
#define linear_algebra_h

#include "dinrhiw_blas.h"
#include <vector>


namespace whiteice
{
  namespace math
  {
    
    template <typename T> class vertex;
    template <typename T> class matrix;
    
    
    /* calculates gram-schimdt orthonormalization for given
     * (partial) basis { B(i,:), B(i+1,:), ... B(j-1,:) }
     * basis vectors are rows of given matrix
     *
     * FIXME: gramschmidt orthonormalization using complex numbers probably has bugs
     */
    template <typename T>
      bool gramschmidt(matrix<T>& B, const unsigned int i, const unsigned int j);
    
    template <typename T>
      bool gramschmidt(std::vector< vertex<T> >& B, const unsigned int i, const unsigned int j);
    
    extern template bool gramschmidt< blas_real<float> >
      (matrix< blas_real<float> >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< blas_real<double> >
      (matrix< blas_real<double> >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< blas_complex<float> >
      (matrix< blas_complex<float> >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< blas_complex<double> >
      (matrix< blas_complex<double> >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt<float>
      (matrix<float>& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt<double>
      (matrix<double>& B, const unsigned int i, const unsigned int j);
    
    extern template bool gramschmidt< blas_real<float> >
      (std::vector< vertex< blas_real<float> > >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< blas_real<double> >
      (std::vector< vertex< blas_real<double> > >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< blas_complex<float> >
      (std::vector< vertex< blas_complex<float> > >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< blas_complex<double> >
      (std::vector< vertex< blas_complex<double> > >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt<float>
      (std::vector< vertex<float> >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt<double>
      (std::vector< vertex<double> >& B, const unsigned int i, const unsigned int j);
  }
}


#include "matrix.h"
#include "vertex.h"


#endif
