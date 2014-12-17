/*
 * implements Fast PCA algorithm as described in:
 *
 * "Fast principal component analysis using fixed-point algorithm."
 * Alok Sharma *, Kuldip K. Paliwal (2007)
 *
 * Tomas Ukkonen 2014
 *
 */

#ifndef __fastpca_h
#define __fastpca_h

#include "matrix.h"
#include "vertex.h"

#include <vector>


namespace whiteice
{
  namespace math
  {
    
    /*
     * Extracts first "dimensions" PCA vectors from data
     * PCA = X^t when Cxx = E{(x-m)(x-m)^t} = X*L*X^t
     */
    template <typename T>
      bool fastpca(const std::vector< vertex<T> >& data, 
		   const unsigned int dimensions,
		   math::matrix<T>& PCA);

    
    extern template bool fastpca<float>(const std::vector< vertex<float> >& data, 
					const unsigned int dimensions,
					math::matrix<float>& PCA);
    extern template bool fastpca<double>(const std::vector< vertex<double> >& data, 
					 const unsigned int dimensions,
					 math::matrix<double>& PCA);
    extern template bool fastpca< blas_real<float> >(const std::vector< vertex< blas_real<float> > >& data, 
						     const unsigned int dimensions,
						     math::matrix< blas_real<float> >& PCA);
    extern template bool fastpca< blas_real<double> >(const std::vector< vertex< blas_real<double> > >& data, 
						      const unsigned int dimensions,
						      math::matrix< blas_real<double> >& PCA);
	
    
    
    
  };
};


#endif

