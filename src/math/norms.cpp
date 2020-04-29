
#include "norms.h"
#include "vertex.h"
#include "matrix.h"


#ifndef math_norms_cpp
#define math_norms_cpp

namespace whiteice
{
  namespace math
  {
    
    
    // calculates frobenius matrix norm
    template <typename T>
    T frobenius_norm(const matrix<T>& A)
    {
      T sum = T(0.0);
      
      for(unsigned int j=0;j<A.ysize();j++){
	for(unsigned int i=0;i<A.xsize();i++){
	  sum += whiteice::math::conj(A(j,i)) * A(j,i);
	}
      }
      
      sum = whiteice::math::sqrt(sum);
      
      return sum;
    }
    
    
    // calculates inf matrix norm
    template <typename T>
    T norm_inf(const matrix<T>& A)
    {
      T c, max = T(0.0);
      
      if(A.ysize() <= 0 || A.xsize() <= 0)
	return max;
      
      max = whiteice::math::abs(A(0,0));
      
      for(unsigned int j=0;j<A.ysize();j++){
	for(unsigned int i=1;i<A.xsize();i++){
	  
	  c = whiteice::math::abs(A(j,i));
	  
	  if(whiteice::math::real(c) > whiteice::math::real(max))
	    max = c;
	}
      }
      
      return max;
    }
    
    
    // calculates inf vector norm
    template <typename T>
    T norm_inf(const vertex<T>& v)
    {
      T c, max = T(0.0);
      
      if(v.size() <= 0) return max;
      
      max = whiteice::math::abs(v[0]);
      
      for(unsigned int i=1;i<v.size();i++){
	
	c = whiteice::math::abs(v[i]);
	
	if(whiteice::math::real(c) > whiteice::math::real(max))
	  max = c;
      }
      
      return max;
    }
    
    
    
    // explicit template instantations
    
    template float frobenius_norm<float>(const matrix<float>& A);
    template double frobenius_norm<double>(const matrix<double>& A);
    template complex<float> frobenius_norm<complex<float> >(const matrix<complex<float> >& A);
    template complex<double> frobenius_norm<complex<double> >(const matrix<complex<double> >& A);
    
    //template int frobenius_norm<int>(const matrix<int>& A);
    //template char frobenius_norm<char>(const matrix<char>& A);
    //template unsigned int frobenius_norm<unsigned int>(const matrix<unsigned int>& A);
    //template unsigned char frobenius_norm<unsigned char>(const matrix<unsigned char>& A);
    
    template blas_real<float> frobenius_norm<blas_real<float> >(const matrix<blas_real<float> >& A);
    template blas_real<double> frobenius_norm<blas_real<double> >(const matrix<blas_real<double> >& A);
    template blas_complex<float> frobenius_norm<blas_complex<float> >(const matrix<blas_complex<float> >& A);
    template blas_complex<double> frobenius_norm<blas_complex<double> >(const matrix<blas_complex<double> >& A);
    
        
    template float norm_inf<float>(const matrix<float>& A);
    template double norm_inf<double>(const matrix<double>& A);
    template complex<float> norm_inf<complex<float> >(const matrix<complex<float> >& A);
    template complex<double> norm_inf<complex<double> >(const matrix<complex<double> >& A);
    
    //template int norm_inf<int>(const matrix<int>& A);
    //template char norm_inf<char>(const matrix<char>& A);
    //template unsigned int norm_inf<unsigned int>(const matrix<unsigned int>& A);
    //template unsigned char norm_inf<unsigned char>(const matrix<unsigned char>& A);
    
    template blas_real<float> norm_inf<blas_real<float> >(const matrix<blas_real<float> >& A);
    template blas_real<double> norm_inf<blas_real<double> >(const matrix<blas_real<double> >& A);
    template blas_complex<float> norm_inf<blas_complex<float> >(const matrix<blas_complex<float> >& A);
    template blas_complex<double> norm_inf<blas_complex<double> >(const matrix<blas_complex<double> >& A);
    
    
    template float norm_inf<float>(const vertex<float>& A);
    template double norm_inf<double>(const vertex<double>& A);
    template complex<float> norm_inf<complex<float> >(const vertex<complex<float> >& A);
    template complex<double> norm_inf<complex<double> >(const vertex<complex<double> >& A);
    
    //template int norm_inf<int>(const vertex<int>& A);
    //template char norm_inf<char>(const vertex<char>& A);
    //template unsigned int norm_inf<unsigned int>(const vertex<unsigned int>& A);
    //template unsigned char norm_inf<unsigned char>(const vertex<unsigned char>& A);
    
    template blas_real<float> norm_inf<blas_real<float> >(const vertex<blas_real<float> >& A);
    template blas_real<double> norm_inf<blas_real<double> >(const vertex<blas_real<double> >& A);
    template blas_complex<float> norm_inf<blas_complex<float> >(const vertex<blas_complex<float> >& A);
    template blas_complex<double> norm_inf<blas_complex<double> >(const vertex<blas_complex<double> >& A);    
  }
}



#endif
