/*
 * correlation code
 */

#ifndef math_correlation_h
#define math_correlation_h

#include <vector>
#include "dinrhiw_blas.h"
#include "dynamic_bitset.h"

namespace whiteice
{
  namespace math
  {
    
    template <typename T>
      class vertex;
    
    template <typename T>
      class matrix;

    // calculates autocorrelation matrix (no mean removal) from given data
    template <typename T>
      bool autocorrelation(matrix<T>& R, const std::vector< vertex<T> >& data);

    
    // calculates autocorrelation matrix from W matrix'es row vectors
    template <typename T>
      bool autocorrelation(matrix<T>& R, const matrix<T>& W);

    
    // calculates mean and covariance matrix (E[(x-mean)(x-mean)']) from given data
    template <typename T>
      bool mean_covariance_estimate(vertex<T>& m, matrix<T>& R,
				    const std::vector< vertex<T> >& data);
    
    // calculates mean and covariance matrix from given data with
    // missing data (some data entries in vectors are missing)
    // missing[i]:s n:th bit is one if entry is missing.
    // missing entries *must* be zero.
    template <typename T>
      bool mean_covariance_estimate(vertex<T>& m, matrix<T>& R, 
				    const std::vector< vertex<T> >& data,
				    const std::vector< whiteice::dynamic_bitset >& missing);

    
    // calculates crosscorrelation matrix Cyx as well as mean values E[x], E[y]
    template <typename T>
    bool mean_crosscorrelation_estimate(vertex<T>& mx, vertex<T>& my, matrix<T>& Cyx,
					const std::vector< vertex<T> >& xdata,
					const std::vector< vertex<T> >& ydata);


    // calculates PCA dimension reduction using symmetric eigenvalue decomposition
    template <typename T>
      bool pca(const std::vector< vertex<T> >& data, 
	       const unsigned int dimensions,
	       math::matrix<T>& PCA,
	       math::vertex<T>& m,
	       T& original_var, T& reduced_var,
	       bool regularizeIfNeeded = false);
    
  }
}

    
#include "blade_math.h"
    

namespace whiteice
{
  namespace math
  {

    extern template bool autocorrelation<float>(matrix<float>& R, const std::vector< vertex<float> >& data);
    extern template bool autocorrelation<double>(matrix<double>& R, const std::vector< vertex<double> >& data);
    
    extern template bool autocorrelation<blas_real<float> >(matrix<blas_real<float> >& R,
							     const std::vector< vertex<blas_real<float> > >& data);
    extern template bool autocorrelation<blas_real<double> >(matrix<blas_real<double> >& R,
							      const std::vector< vertex<blas_real<double> > >& data);
    extern template bool autocorrelation<blas_complex<float> >(matrix<blas_complex<float> >& R,
								const std::vector< vertex<blas_complex<float> > >& data);
    extern template bool autocorrelation<blas_complex<double> >(matrix<blas_complex<double> >& R,
								 const std::vector< vertex<blas_complex<double> > >& data);
    extern template bool autocorrelation<complex<float> >(matrix<complex<float> >& R,
							  const std::vector< vertex<complex<float> > >& data);
    extern template bool autocorrelation<complex<double> >(matrix<complex<double> >& R,
							   const std::vector< vertex<complex<double> > >& data);    
    extern template bool autocorrelation<int>(matrix<int>& R, const std::vector< vertex<int> >& data);
    
    extern template bool autocorrelation<float>(matrix<float>& R, const matrix<float>& W);
    extern template bool autocorrelation<double>(matrix<double>& R, const matrix<double>& W);
    
    extern template bool autocorrelation<blas_real<float> >(matrix<blas_real<float> >& R,
							     const matrix<blas_real<float> >& W);
    extern template bool autocorrelation<blas_real<double> >(matrix<blas_real<double> >& R,
							      const matrix<blas_real<double> >& W);
    extern template bool autocorrelation<blas_complex<float> >(matrix<blas_complex<float> >& R,
								const matrix<blas_complex<float> >& W);
    extern template bool autocorrelation<blas_complex<double> >(matrix<blas_complex<double> >& R,
								 const matrix<blas_complex<double> >& W);
    
    extern template bool mean_covariance_estimate< float >
      (vertex< float >& m, matrix< float >& R,
       const std::vector< vertex< float > >& data);

    extern template bool mean_covariance_estimate< double >
      (vertex< double >& m, matrix< double >& R,
       const std::vector< vertex< double > >& data);
    
    extern template bool mean_covariance_estimate< blas_real<float> >
      (vertex< blas_real<float> >& m, matrix< blas_real<float> >& R,
       const std::vector< vertex< blas_real<float> > >& data);

    extern template bool mean_covariance_estimate< blas_real<double> >
      (vertex< blas_real<double> >& m, matrix< blas_real<double> >& R,
       const std::vector< vertex< blas_real<double> > >& data);
    
    extern template bool mean_covariance_estimate< blas_complex<float> >
      (vertex< blas_complex<float> >& m, matrix< blas_complex<float> >& R,
       const std::vector< vertex< blas_complex<float> > >& data);
    
    extern template bool mean_covariance_estimate< blas_complex<double> > 
      (vertex< blas_complex<double> >& m, matrix< blas_complex<double> >& R,
       const std::vector< vertex< blas_complex<double> > >& data);
    
    
    extern template bool mean_covariance_estimate< float >
      (vertex< float >& m, matrix< float >& R,
       const std::vector< vertex< float > >& data,
       const std::vector< whiteice::dynamic_bitset >& missing);
    
    extern template bool mean_covariance_estimate< double >
      (vertex< double >& m, matrix< double >& R,
       const std::vector< vertex< double > >& data,
       const std::vector< whiteice::dynamic_bitset >& missing);
    
    extern template bool mean_covariance_estimate< blas_real<float> >
      (vertex< blas_real<float> >& m, matrix< blas_real<float> >& R,
       const std::vector< vertex< blas_real<float> > >& data,
       const std::vector< whiteice::dynamic_bitset >& missing);
    
    extern template bool mean_covariance_estimate< blas_real<double> >
      (vertex< blas_real<double> >& m, matrix< blas_real<double> >& R,
       const std::vector< vertex< blas_real<double> > >& data,
       const std::vector< whiteice::dynamic_bitset >& missing);
    
    extern template bool mean_covariance_estimate< blas_complex<float> >
      (vertex< blas_complex<float> >& m, matrix< blas_complex<float> >& R,
       const std::vector< vertex< blas_complex<float> > >& data,
       const std::vector< whiteice::dynamic_bitset >& missing);
    
    extern template bool mean_covariance_estimate< blas_complex<double> >
      (vertex< blas_complex<double> >& m, matrix< blas_complex<double> >& R,
       const std::vector< vertex< blas_complex<double> > >& data,
       const std::vector< whiteice::dynamic_bitset >& missing);

    extern template bool mean_crosscorrelation_estimate< float >
    (vertex< float >& mx, vertex< float >& my, matrix< float >& Cyx,
     const std::vector< vertex<float> >& xdata,
     const std::vector< vertex<float> >& ydata);

    extern template bool mean_crosscorrelation_estimate< double >
    (vertex< double >& mx, vertex< double >& my, matrix< double >& Cyx,
     const std::vector< vertex<double> >& xdata,
     const std::vector< vertex<double> >& ydata);
    
    extern template bool mean_crosscorrelation_estimate< blas_real<float> >
    (vertex< blas_real<float> >& mx, vertex< blas_real<float> >& my, matrix< blas_real<float> >& Cyx,
     const std::vector< vertex< blas_real<float> > >& xdata,
     const std::vector< vertex< blas_real<float> > >& ydata);

    extern template bool mean_crosscorrelation_estimate< blas_real<double> >
    (vertex< blas_real<double> >& mx, vertex< blas_real<double> >& my, matrix< blas_real<double> >& Cyx,
     const std::vector< vertex< blas_real<double> > >& xdata,
     const std::vector< vertex< blas_real<double> > >& ydata);

    extern template bool mean_crosscorrelation_estimate< blas_complex<float> >
    (vertex< blas_complex<float> >& mx, vertex< blas_complex<float> >& my, matrix< blas_complex<float> >& Cyx,
     const std::vector< vertex< blas_complex<float> > >& xdata,
     const std::vector< vertex< blas_complex<float> > >& ydata);

    extern template bool mean_crosscorrelation_estimate< blas_complex<double> >
    (vertex< blas_complex<double> >& mx, vertex< blas_complex<double> >& my, matrix< blas_complex<double> >& Cyx,
     const std::vector< vertex< blas_complex<double> > >& xdata,
     const std::vector< vertex< blas_complex<double> > >& ydata);


    extern template bool pca<float>
      (const std::vector< vertex<float> >& data, 
       const unsigned int dimensions,
       math::matrix<float>& PCA,
       math::vertex<float>& m,
       float& original_var, float& reduced_var,
       bool regularizeIfNeeded);

    extern template bool pca<double>
      (const std::vector< vertex<double> >& data, 
       const unsigned int dimensions,
       math::matrix<double>& PCA,
       math::vertex<double>& m,
       double& original_var, double& reduced_var,
       bool regularizeIfNeeded);

    extern template bool pca< blas_real<float> >
      (const std::vector< vertex< blas_real<float> > >& data, 
       const unsigned int dimensions,
       math::matrix< blas_real<float> >& PCA,
       math::vertex< blas_real<float> >& m,
       blas_real<float>& original_var, blas_real<float>& reduced_var,
       bool regularizeIfNeeded);

    extern template bool pca< blas_real<double> >
      (const std::vector< vertex< blas_real<double> > >& data, 
       const unsigned int dimensions,
       math::matrix< blas_real<double> >& PCA,
       math::vertex< blas_real<double> >& m,
       blas_real<double>& original_var, blas_real<double>& reduced_var,
       bool regularizeIfNeeded);
    
  }
}


#include "matrix.h"
#include "vertex.h"


#endif


