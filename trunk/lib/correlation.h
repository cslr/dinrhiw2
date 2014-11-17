/*
 * correlation code
 */

#ifndef math_correlation_h
#define math_correlation_h

#include <vector>
#include "atlas.h"
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
    
  }
}

    
#include "blade_math.h"
    

namespace whiteice
{
  namespace math
  {

    extern template bool autocorrelation<float>(matrix<float>& R, const std::vector< vertex<float> >& data);
    extern template bool autocorrelation<double>(matrix<double>& R, const std::vector< vertex<double> >& data);
    
    extern template bool autocorrelation<atlas_real<float> >(matrix<atlas_real<float> >& R,
							     const std::vector< vertex<atlas_real<float> > >& data);
    extern template bool autocorrelation<atlas_real<double> >(matrix<atlas_real<double> >& R,
							      const std::vector< vertex<atlas_real<double> > >& data);
    extern template bool autocorrelation<atlas_complex<float> >(matrix<atlas_complex<float> >& R,
								const std::vector< vertex<atlas_complex<float> > >& data);
    extern template bool autocorrelation<atlas_complex<double> >(matrix<atlas_complex<double> >& R,
								 const std::vector< vertex<atlas_complex<double> > >& data);
    extern template bool autocorrelation<complex<float> >(matrix<complex<float> >& R,
							  const std::vector< vertex<complex<float> > >& data);
    extern template bool autocorrelation<complex<double> >(matrix<complex<double> >& R,
							   const std::vector< vertex<complex<double> > >& data);    
    extern template bool autocorrelation<int>(matrix<int>& R, const std::vector< vertex<int> >& data);
    
    extern template bool autocorrelation<float>(matrix<float>& R, const matrix<float>& W);
    extern template bool autocorrelation<double>(matrix<double>& R, const matrix<double>& W);
    
    extern template bool autocorrelation<atlas_real<float> >(matrix<atlas_real<float> >& R,
							     const matrix<atlas_real<float> >& W);
    extern template bool autocorrelation<atlas_real<double> >(matrix<atlas_real<double> >& R,
							      const matrix<atlas_real<double> >& W);
    extern template bool autocorrelation<atlas_complex<float> >(matrix<atlas_complex<float> >& R,
								const matrix<atlas_complex<float> >& W);
    extern template bool autocorrelation<atlas_complex<double> >(matrix<atlas_complex<double> >& R,
								 const matrix<atlas_complex<double> >& W);
    
    extern template bool mean_covariance_estimate< atlas_real<float> >
      (vertex< atlas_real<float> >& m, matrix< atlas_real<float> >& R,
       const std::vector< vertex< atlas_real<float> > >& data);

    extern template bool mean_covariance_estimate< atlas_real<double> >
      (vertex< atlas_real<double> >& m, matrix< atlas_real<double> >& R,
       const std::vector< vertex< atlas_real<double> > >& data);
    
    extern template bool mean_covariance_estimate< atlas_complex<float> >
      (vertex< atlas_complex<float> >& m, matrix< atlas_complex<float> >& R,
       const std::vector< vertex< atlas_complex<float> > >& data);
    
    extern template bool mean_covariance_estimate< atlas_complex<double> > 
      (vertex< atlas_complex<double> >& m, matrix< atlas_complex<double> >& R,
       const std::vector< vertex< atlas_complex<double> > >& data);
    
    
    extern template bool mean_covariance_estimate< atlas_real<float> >
      (vertex< atlas_real<float> >& m, matrix< atlas_real<float> >& R,
       const std::vector< vertex< atlas_real<float> > >& data,
       const std::vector< whiteice::dynamic_bitset >& missing);
    
    extern template bool mean_covariance_estimate< atlas_real<double> >
      (vertex< atlas_real<double> >& m, matrix< atlas_real<double> >& R,
       const std::vector< vertex< atlas_real<double> > >& data,
       const std::vector< whiteice::dynamic_bitset >& missing);
    
    extern template bool mean_covariance_estimate< atlas_complex<float> >
      (vertex< atlas_complex<float> >& m, matrix< atlas_complex<float> >& R,
       const std::vector< vertex< atlas_complex<float> > >& data,
       const std::vector< whiteice::dynamic_bitset >& missing);
    
    extern template bool mean_covariance_estimate< atlas_complex<double> >
      (vertex< atlas_complex<double> >& m, matrix< atlas_complex<double> >& R,
       const std::vector< vertex< atlas_complex<double> > >& data,
       const std::vector< whiteice::dynamic_bitset >& missing);
    
  }
}


#include "matrix.h"
#include "vertex.h"


#endif


