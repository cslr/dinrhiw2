/*
 * implements Fast PCA algorithm as described in:
 *
 * "Fast principal component analysis using fixed-point algorithm."
 * Alok Sharma *, Kuldip K. Paliwal (2007)
 *
 * FIXME: This apparently only works with real valued data but
 *        extensions to complex valued shouldn't be too hard.
 *
 * FIXME: The algorithm currently computes Cxx matrix and computes
 *        better = Cxx*candidate_vector. For large dimensionality
 *        data Cxx cannot be computed and matrix-vector product
 *        should be apprximated. If we remove mean from the data
 *        we see that Cxx = (a*a^h + b*b^h + c*c^h ..)/N.
 *        It means we can select M vectors randomly from data
 *        and compute Cxx*c ~ (a*(a^h*candidate) + b*(b^h*candidate)..)/M.
 *
 *        It then follows that we don't have to compute Cxx at all but
 *        compute M dot produts of vectors instead.
 *
 *        Write improved code which works with HUGE dimensional vectors and 
 *        large number of datapoints (need only sample smaller numbers to
 *        make statistical estimate).
 *
 * TODO:  PCA vectors returned don't have unit variance
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
     * PCA = X^t when Cxx = E{(x-m)(x-m)^t} = X*D*X^t
     */
    template <typename T>
    bool fastpca(const std::vector< vertex<T> >& data, 
		 const unsigned int dimensions,
		 math::matrix<T>& PCA);
    
    /*
     * Extracts PCA vectors having top p% E (0,1] of the total
     * variance in data. (Something like 90% could be
     * good for preprocessing while keeping most of variation
     * in data.
     */
    template <typename T>
    bool fastpca_p(const std::vector <vertex<T> >& data,
		   const float percent_total_variance,
		   math::matrix<T>& PCA);
		 

    
    extern template bool fastpca< blas_real<float> >
    (const std::vector< vertex< blas_real<float> > >& data, 
     const unsigned int dimensions,
     math::matrix< blas_real<float> >& PCA);
    
    extern template bool fastpca< blas_real<double> >
    (const std::vector< vertex< blas_real<double> > >& data, 
     const unsigned int dimensions,
     math::matrix< blas_real<double> >& PCA);
    
    extern template bool fastpca_p< blas_real<float> >
    (const std::vector <vertex< blas_real<float> > >& data,
     const float percent_total_variance,
     math::matrix< blas_real<float> >& PCA);

    extern template bool fastpca_p< blas_real<double> >
    (const std::vector <vertex< blas_real<double> > >& data,
     const float percent_total_variance,
     math::matrix< blas_real<double> >& PCA);
    
  };
};


#endif

