/*
 * mathematical norms
 */

#ifndef math_norms_h
#define math_norms_h


#include "vertex.h"
#include "matrix.h"
#include "blade_math.h"
#include "dinrhiw_blas.h"


namespace whiteice
{
  namespace math
  {
    
    // calculates frobenius matrix norm
    // (sum of squared values)
    template <typename T>
      T frobenius_norm(const matrix<T>& A);
    
    // calculates inf matrix norm
    // (the biggest absolute value)
    template <typename T>
      T norm_inf(const matrix<T>& A);
    
    // calculates inf vector norm
    // (the biggest absolute value)
    template <typename T>
      T norm_inf(const vertex<T>& v);
    
    
    
    extern template float frobenius_norm<float>(const matrix<float>& A);
    extern template double frobenius_norm<double>(const matrix<double>& A);
    extern template complex<float> frobenius_norm<complex<float> >(const matrix<complex<float> >& A);
    extern template complex<double> frobenius_norm<complex<double> >(const matrix<complex<double> >& A);
    
    
    extern template blas_real<float> frobenius_norm<blas_real<float> >(const matrix<blas_real<float> >& A);
    extern template blas_real<double> frobenius_norm<blas_real<double> >(const matrix<blas_real<double> >& A);
    extern template blas_complex<float> frobenius_norm<blas_complex<float> >(const matrix<blas_complex<float> >& A);
    extern template blas_complex<double> frobenius_norm<blas_complex<double> >(const matrix<blas_complex<double> >& A);

    extern template superresolution<blas_real<float>, modular<unsigned int> > frobenius_norm< superresolution<blas_real<float>, modular<unsigned int> > >(const matrix< superresolution<blas_real<float>, modular<unsigned int> > >& A);

    extern template superresolution<blas_real<double>, modular<unsigned int> > frobenius_norm< superresolution<blas_real<double>, modular<unsigned int> > >(const matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A);
    
    extern template superresolution<blas_complex<float>, modular<unsigned int> > frobenius_norm< superresolution<blas_complex<float>, modular<unsigned int> > >(const matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& A);

    extern template superresolution<blas_complex<double>, modular<unsigned int> > frobenius_norm< superresolution<blas_complex<double>, modular<unsigned int> > >(const matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A);


    
    
        
    extern template float norm_inf<float>(const matrix<float>& A);
    extern template double norm_inf<double>(const matrix<double>& A);
    extern template complex<float> norm_inf<complex<float> >(const matrix<complex<float> >& A);
    extern template complex<double> norm_inf<complex<double> >(const matrix<complex<double> >& A);
    
    extern template blas_real<float> norm_inf<blas_real<float> >(const matrix<blas_real<float> >& A);
    extern template blas_real<double> norm_inf<blas_real<double> >(const matrix<blas_real<double> >& A);
    extern template blas_complex<float> norm_inf<blas_complex<float> >(const matrix<blas_complex<float> >& A);
    extern template blas_complex<double> norm_inf<blas_complex<double> >(const matrix<blas_complex<double> >& A);

    extern template superresolution<blas_real<float>, modular<unsigned int> > norm_inf< superresolution<blas_real<float>, modular<unsigned int> > >(const matrix< superresolution<blas_real<float>, modular<unsigned int> > >& A);

    extern template superresolution<blas_real<double>, modular<unsigned int> > norm_inf< superresolution<blas_real<double>, modular<unsigned int> > >(const matrix< superresolution<blas_real<double>, modular<unsigned int> > >& A);

    extern template superresolution<blas_complex<float>, modular<unsigned int> > norm_inf< superresolution<blas_complex<float>, modular<unsigned int> > >(const matrix< superresolution<blas_complex<float>, modular<unsigned int> > >& A);

    extern template superresolution<blas_complex<double>, modular<unsigned int> > norm_inf< superresolution<blas_complex<double>, modular<unsigned int> > >(const matrix< superresolution<blas_complex<double>, modular<unsigned int> > >& A);
    
    
    extern template float norm_inf<float>(const vertex<float>& A);
    extern template double norm_inf<double>(const vertex<double>& A);
    extern template complex<float> norm_inf<complex<float> >(const vertex<complex<float> >& A);
    extern template complex<double> norm_inf<complex<double> >(const vertex<complex<double> >& A);
    
    
    extern template blas_real<float> norm_inf<blas_real<float> >(const vertex<blas_real<float> >& A);
    extern template blas_real<double> norm_inf<blas_real<double> >(const vertex<blas_real<double> >& A);
    extern template blas_complex<float> norm_inf<blas_complex<float> >(const vertex<blas_complex<float> >& A);
    extern template blas_complex<double> norm_inf<blas_complex<double> >(const vertex<blas_complex<double> >& A);

    extern template superresolution<blas_real<float>, modular<unsigned int> > norm_inf< superresolution<blas_real<float>, modular<unsigned int> > >(const vertex< superresolution<blas_real<float>, modular<unsigned int> > >& A);

    extern template superresolution<blas_real<double>, modular<unsigned int> > norm_inf< superresolution<blas_real<double>, modular<unsigned int> > >(const vertex< superresolution<blas_real<double>, modular<unsigned int> > >& A);

    extern template superresolution<blas_complex<float>, modular<unsigned int> > norm_inf< superresolution<blas_complex<float>, modular<unsigned int> > >(const vertex< superresolution<blas_complex<float>, modular<unsigned int> > >& A);

    extern template superresolution<blas_complex<double>, modular<unsigned int> > norm_inf< superresolution<blas_complex<double>, modular<unsigned int> > >(const vertex< superresolution<blas_complex<double>, modular<unsigned int> > >& A);
    
  }
}


#endif
