/*
 * linear algebra algorithms
 */

#ifndef linear_algebra_h
#define linear_algebra_h

#include "atlas.h"
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
     */
    template <typename T>
      bool gramschmidt(matrix<T>& B, const unsigned int i, const unsigned int j);
    
    template <typename T>
      bool gramschmidt(std::vector< vertex<T> >& B, const unsigned int i, const unsigned int j);
    
    extern template bool gramschmidt< atlas_real<float> >
      (matrix< atlas_real<float> >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< atlas_real<double> >
      (matrix< atlas_real<double> >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< atlas_complex<float> >
      (matrix< atlas_complex<float> >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< atlas_complex<double> >
      (matrix< atlas_complex<double> >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt<float>
      (matrix<float>& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt<double>
      (matrix<double>& B, const unsigned int i, const unsigned int j);
    
    extern template bool gramschmidt< atlas_real<float> >
      (std::vector< vertex< atlas_real<float> > >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< atlas_real<double> >
      (std::vector< vertex< atlas_real<double> > >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< atlas_complex<float> >
      (std::vector< vertex< atlas_complex<float> > >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt< atlas_complex<double> >
      (std::vector< vertex< atlas_complex<double> > >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt<float>
      (std::vector< vertex<float> >& B, const unsigned int i, const unsigned int j);
    extern template bool gramschmidt<double>
      (std::vector< vertex<double> >& B, const unsigned int i, const unsigned int j);
  }
}


#include "matrix.h"
#include "vertex.h"


#endif
