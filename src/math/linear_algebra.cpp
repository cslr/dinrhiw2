
#include <vector>
#include <typeinfo>
#include "linear_algebra.h"

#include "matrix.h"
#include "vertex.h"
#include "dinrhiw_blas.h"
#include "Log.h"

// #define OPENBLAS 1

namespace whiteice
{
  namespace math
  {
    
    template bool gramschmidt< blas_real<float> >
      (matrix< blas_real<float> >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< blas_real<double> >
      (matrix< blas_real<double> >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< blas_complex<float> >
      (matrix< blas_complex<float> >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< blas_complex<double> >
      (matrix< blas_complex<double> >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt<float>
      (matrix<float>& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt<double>
      (matrix<double>& B, const unsigned int i, const unsigned int j);
    
    
    template bool gramschmidt< blas_real<float> >
      (std::vector< vertex< blas_real<float> > >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< blas_real<double> >
      (std::vector< vertex< blas_real<double> > >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< blas_complex<float> >
      (std::vector< vertex< blas_complex<float> > >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< blas_complex<double> >
      (std::vector< vertex< blas_complex<double> > >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt<float>
      (std::vector< vertex<float> >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt<double>
      (std::vector< vertex<double> >& B, const unsigned int i, const unsigned int j);
    
    
    
    
    // calculates gram-schimdt orthonormalization
    template <typename T>
    bool gramschmidt(matrix<T>& B,
		     const unsigned int i,
		     const unsigned int j)
    {
      if(i>=j || j>B.size())
	return false;
      
      vertex<T> z;
      
      // B = [NxM matrix]
      // const unsigned int N = B.ysize();
      const unsigned int M = B.xsize();
      z.resize(M);

#if CUBLAS

      if(typeid(T) == typeid(blas_real<float>)){
	
	for(unsigned int n=i;n<j;n++){
	  z.zero();
	  blas_real<float> w;
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*numCols], B.data[n*numCols], numRows);
	    
	    auto e = cublasSdot(cublas_handle, B.numCols,
				(const float*)&(B[k]), B.numRows,
				(const float*)&(B[n]), B.numRows,
				(float*)&w);

	    if(e != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("gramschmidt(): cublasSdot() failed.");
	      throw CUDAException("CUBLAS cublasSdot() call failed.");
	    }
	    
	    // z += scal(w, data[k*numCols], numRows)
	    
	    e = cublasSaxpy(cublas_handle, B.numCols,
			    (const float*)&w,
			    (const float*)&(B[k]), B.numRows,
			    (float*)z.data, 1);

	    if(e != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("gramschmidt(): cublasSaxpy() failed.");
	      throw CUDAException("CUBLAS cublasSaxpy() call failed.");
	    }
	    
	  }
	  
	  // B[n] -= z;
	  
	  T alpha = T(-1.0f);
	  
	  auto e = cublasSaxpy(cublas_handle, B.numCols,
			       (const float*)&alpha,
			       (const float*)z.data, 1,
			       (float*)&(B[n]), B.numRows);

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasSaxpy() failed.");
	    throw CUDAException("CUBLAS cublasSaxpy() call failed.");
	  }

	  T nrm2 = T(0.0f);
	  
	  e = cublasSnrm2(cublas_handle, B.numCols,
			  (const float*)&(B[n]), B.numRows, (float*)&nrm2);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasSnrm2() failed.");
	    throw CUDAException("CUBLAS cublasSnrm2() call failed.");
	  }
	  
	  if(nrm2 != 0.0f) w = T(1.0f) / whiteice::math::sqrt(whiteice::math::abs(nrm2));
	  
	  e = cublasSscal(cublas_handle, B.numCols,
			  (const float*)&w, (float*)&(B[n]), B.numRows);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasSscal() failed.");
	    throw CUDAException("CUBLAS cublasSscal() call failed.");
	  }
	  
	}

	gpu_sync();
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){

	for(unsigned int n=i;n<j;n++){
	  z.zero();
	  blas_complex<float> w;
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*numCols], B.data[n*numCols], numRows);
	    
	    auto e = cublasCdotc(cublas_handle, B.numCols,
				 (const cuComplex*)&(B[k]), B.numRows,
				 (const cuComplex*)&(B[n]), B.numRows,
				 (cuComplex*)&w);

	    if(e != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("gramschmidt(): cublasCdotc() failed.");
	      throw CUDAException("CUBLAS cublasCdotc() call failed.");
	    }
	    
	    // z += scal(w, data[k*numCols], numRows)
	    
	    e = cublasCaxpy(cublas_handle, B.numCols,
			    (const cuComplex*)&w,
			    (const cuComplex*)&(B[k]), B.numRows,
			    (cuComplex*)z.data, 1);

	    if(e != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("gramschmidt(): cublasCaxpy() failed.");
	      throw CUDAException("CUBLAS cublasCaxpy() call failed.");
	    }
	    
	  }
	  
	  // B[n] -= z;
	  
	  T alpha = T(-1.0f);
	  
	  auto e = cublasCaxpy(cublas_handle, B.numCols,
			       (const cuComplex*)&alpha,
			       (const cuComplex*)z.data, 1,
			       (cuComplex*)&(B[n]), B.numRows);

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasCaxpy() failed.");
	    throw CUDAException("CUBLAS cublasCaxpy() call failed.");
	  }

	  float nrm2 = 0.0f;
	  
	  e = cublasScnrm2(cublas_handle, B.numCols,
			   (const cuComplex*)&(B[n]), B.numRows, (float*)&nrm2);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasScnrm2() failed.");
	    throw CUDAException("CUBLAS cublasScnrm2() call failed.");
	  }
	  
	  if(nrm2 != 0.0f){
	    auto v = T(1.0f) / whiteice::math::sqrt(whiteice::math::abs(T(nrm2)));
	    whiteice::math::convert(w, v);
	  }

	  e = cublasCscal(cublas_handle, B.numCols,
			  (const cuComplex*)&w, (cuComplex*)&(B[n]), B.numRows);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasCscal() failed.");
	    throw CUDAException("CUBLAS cublasCscal() call failed.");
	  }
	  
	}

	gpu_sync();
		
      }
      else if(typeid(T) == typeid(blas_real<double>)){

	for(unsigned int n=i;n<j;n++){
	  z.zero();
	  blas_real<double> w;
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*numCols], B.data[n*numCols], numRows);
	    
	    auto e = cublasDdot(cublas_handle, B.numCols,
				(const double*)&(B[k]), B.numRows,
				(const double*)&(B[n]), B.numRows,
				(double*)&w);

	    if(e != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("gramschmidt(): cublasDdot() failed.");
	      throw CUDAException("CUBLAS cublasDdot() call failed.");
	    }
	    
	    // z += scal(w, data[k*numCols], numRows)
	    
	    e = cublasDaxpy(cublas_handle, B.numCols,
			    (const double*)&w,
			    (const double*)&(B[k]), B.numRows,
			    (double*)z.data, 1);

	    if(e != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("gramschmidt(): cublasDaxpy() failed.");
	      throw CUDAException("CUBLAS cublasDaxpy() call failed.");
	    }
	    
	  }
	  
	  // B[n] -= z;
	  
	  T alpha = T(-1.0f);
	  
	  auto e = cublasDaxpy(cublas_handle, B.numCols,
			       (const double*)&alpha,
			       (const double*)z.data, 1,
			       (double*)&(B[n]), B.numRows);

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasDaxpy() failed.");
	    throw CUDAException("CUBLAS cublasDaxpy() call failed.");
	  }

	  T nrm2 = T(0.0f);
	  
	  e = cublasDnrm2(cublas_handle, B.numCols,
			  (const double*)&(B[n]), B.numRows, (double*)&nrm2);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasDnrm2() failed.");
	    throw CUDAException("CUBLAS cublasDnrm2() call failed.");
	  }
	  
	  if(nrm2 != 0.0f) w = T(1.0f) / whiteice::math::sqrt(whiteice::math::abs(nrm2));

	  e = cublasDscal(cublas_handle, B.numCols,
			  (const double*)&w, (double*)&(B[n]), B.numRows);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasDscal() failed.");
	    throw CUDAException("CUBLAS cublasDscal() call failed.");
	  }
	  
	}

	gpu_sync();
	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	for(unsigned int n=i;n<j;n++){
	  z.zero();
	  blas_complex<double> w;
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*numCols], B.data[n*numCols], numRows);
	    
	    auto e = cublasZdotc(cublas_handle, B.numCols,
				 (const cuDoubleComplex*)&(B[k]), B.numRows,
				 (const cuDoubleComplex*)&(B[n]), B.numRows,
				 (cuDoubleComplex*)&w);

	    if(e != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("gramschmidt(): cublasDdotc() failed.");
	      throw CUDAException("CUBLAS cublasDdotc() call failed.");
	    }
	    
	    // z += scal(w, data[k*numCols], numRows)
	    
	    e = cublasZaxpy(cublas_handle, B.numCols,
			    (const cuDoubleComplex*)&w,
			    (const cuDoubleComplex*)&(B[k]), B.numRows,
			    (cuDoubleComplex*)z.data, 1);

	    if(e != CUBLAS_STATUS_SUCCESS){
	      whiteice::logging.error("gramschmidt(): cublasZaxpy() failed.");
	      throw CUDAException("CUBLAS cublasZaxpy() call failed.");
	    }
	    
	  }
	  
	  // B[n] -= z;
	  
	  T alpha = T(-1.0f);
	  
	  auto e = cublasZaxpy(cublas_handle, B.numCols,
			       (const cuDoubleComplex*)&alpha,
			       (const cuDoubleComplex*)z.data, 1,
			       (cuDoubleComplex*)&(B[n]), B.numRows);

	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasZaxpy() failed.");
	    throw CUDAException("CUBLAS cublasZaxpy() call failed.");
	  }

	  double nrm2 = 0.0f;
	  
	  e = cublasDznrm2(cublas_handle, B.numCols,
			   (const cuDoubleComplex*)&(B[n]), B.numRows, (double*)&nrm2);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasDznrm2() failed.");
	    throw CUDAException("CUBLAS cublasDznrm2() call failed.");
	  }
	  
	  if(nrm2 != 0.0f){
	    auto v = T(1.0f) / whiteice::math::sqrt(whiteice::math::abs(T(nrm2)));
	    whiteice::math::convert(w, v);
	  }

	  e = cublasZscal(cublas_handle, B.numCols,
			  (const cuDoubleComplex*)&w, (cuDoubleComplex*)&(B[n]), B.numRows);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("gramschmidt(): cublasZdscal() failed.");
	    throw CUDAException("CUBLAS cublasZdscal() call failed.");
	  }
	  
	}

	gpu_sync();
	
      }
      else{
	// only for real valued basis
	
	vertex<T> a, b;
	a.resize(M);
	b.resize(M);
	
	for(unsigned int n=i;n<j;n++){
	  z = T(0);
	  T r = T(0);
	  
	  for(unsigned int k=i;k<n;k++){
	    r = T(0);
	    for(unsigned int u=0;u<M;u++)
	      r += B(k,u) * B(n,u); // conj (?)
	    
	    for(unsigned int u=0;u<M;u++)
	      z[u] += r * B(k, u);
	  }
	  
	  for(unsigned int u=0;u<M;u++)
	    B(n,u) -= z[u];
	  
	  r = T(0);
	  for(unsigned int u=0;u<M;u++)
	    r += B(n,u)*B(n,u); // conj (?)
	  
	  r = whiteice::math::sqrt(r);
	  
	  for(unsigned int u=0;u<M;u++)
	    B(n,u) = B(n,u) / r;
	}
      }
      
#else
      if(typeid(T) == typeid(blas_real<float>)){
	
	for(unsigned int n=i;n<j;n++){
	  z = T(0);
	  blas_real<float> w;
	  
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*numCols], B.data[n*numCols], numRows);
	    w = cblas_sdot(B.numCols, 
			   (float*)&(B.data[k*B.numCols]), 1,
			   (float*)&(B.data[n*B.numCols]), 1);
	    
	    // z += scal(w, data[k*numCols], numRows)
	    cblas_saxpy(B.numCols, *((float*)(&w)),
			(float*)&(B.data[k*B.numCols]), 1,
			(float*)z.data, 1);
	  }
	  
	  // B[n] -= z;
	  cblas_saxpy(B.numCols, -1.0f,
		      (float*)z.data, 1,
		      (float*)&(B.data[n*B.numCols]), 1);
	  
	  // normalize(B.data[n*B.numCols], B.numRows)
	  w = cblas_snrm2(B.numCols, (float*)&(B.data[n*B.numCols]), 1);
	  
	  if(w != 0.0f) w = 1.0f / whiteice::math::sqrt(whiteice::math::abs(w));
	  
	  cblas_sscal(B.numCols, *((float*)(&w)), (float*)&(B.data[n*B.numCols]), 1);
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	for(unsigned int n=i;n<j;n++){
	  z = T(0);
	  blas_complex<float> w;
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*numCols], B.data[n*numCols], numRows);
#ifdef OPENBLAS
	    cblas_cdotc_sub(B.numCols, 
			    (float*)&(B.data[k*B.numCols]), 1,
			    (float*)&(B.data[n*B.numCols]), 1, (openblas_complex_float*)&w);
	    // (float*)&(B.data[n*B.numCols]), 1, (float*)&w);
#else
	    cblas_cdotc_sub(B.numCols, 
			    (float*)&(B.data[k*B.numCols]), 1,
			    // (float*)&(B.data[n*B.numCols]), 1, (openblas_complex_float*)&w);
			    (float*)&(B.data[n*B.numCols]), 1, (float*)&w);
#endif
	    
	    
	    // z += scal(w, B.data[k*numCols], numRows)
	    cblas_caxpy(B.numCols, (const float*)&w,
			(float*)&(B.data[k*B.numCols]), 1,
			(float*)z.data, 1);
	  }
	  
	  // B[n] -= z;
	  w = blas_complex<float>(-1.0f);
	  cblas_caxpy(B.numCols, (const float*)&w,
		      (float*)z.data, 1,
		      (float*)&(B.data[n*B.numCols]), 1);
	  
	  // normalize(B.data[n*numCols], numRows)
	  float nn = cblas_scnrm2(B.numCols, (float*)&(B.data[n*B.numCols]), 1);

	  nn = 1.0f/sqrtf(abs(nn));
	  
	  cblas_csscal(B.numCols, nn, (float*)&(B.data[n*B.numCols]), 1);
	}
	
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	for(unsigned int n=i;n<j;n++){
	  z = T(0);
	  blas_real<double> w;
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*B.numCols], B.data[n*B.numCols], B.numRows);
	    w = cblas_ddot(B.numCols, 
			   (double*)&(B.data[k*B.numCols]), 1,
			   (double*)&(B.data[n*B.numCols]), 1);
	    
	    // z += scal(w, B.data[k*numCols], numRows)
	    cblas_daxpy(B.numCols, *((double*)&w),
			(double*)&(B.data[k*B.numCols]), 1,
			(double*)z.data, 1);
	  }
	  
	  // B[n] -= z;
	  cblas_daxpy(B.numCols, -1.0,
		      (double*)z.data, 1,
		      (double*)&(B.data[n*B.numCols]), 1);
	  
	  // normalize(B.data[n*numCols], numRows)
	  w = cblas_dnrm2(B.numCols, (double*)&(B.data[n*B.numCols]), 1);

	  w = blas_real<double>(1.0)/whiteice::math::sqrt(whiteice::math::abs(w));
	  
	  cblas_dscal(B.numCols, *((double*)&w), 
		      (double*)&(B.data[n*B.numCols]), 1);
	}
	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	for(unsigned int n=i;n<j;n++){
	  z = T(0);
	  blas_complex<double> w;
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*numCols], B.data[n*numCols], numRows);
#ifdef OPENBLAS
	    cblas_zdotc_sub(B.numCols, 
	      (double*)&(B.data[k*B.numCols]), 1,
	      (double*)&(B.data[n*B.numCols]), 1, (openblas_complex_double*)&w);
	    // (double*)&(B.data[n*B.numCols]), 1, (double*)&w);
#else
	    cblas_zdotc_sub(B.numCols, 
	      (double*)&(B.data[k*B.numCols]), 1,
	      // (double*)&(B.data[n*B.numCols]), 1, (openblas_complex_double*)&w);
	      (double*)&(B.data[n*B.numCols]), 1, (double*)&w);
#endif

	    
	    // z += scal(w, B.data[k*numCols], numRows)
	    cblas_zaxpy(B.numCols, (const double*)&w,
			(double*)(&B.data[k*B.numCols]), 1,
			(double*)z.data, 1);
	  }
	  
	  // B[n] -= z;
	  w = blas_complex<double>(-1.0);
	  cblas_zaxpy(B.numCols, (const double*)&w,
		      (double*)z.data, 1,
		      (double*)&(B.data[n*B.numCols]), 1);
	  
	  // normalize(B.data[n*numCols], numRows)
	  double nn = cblas_dznrm2(B.numCols,
				   (double*)(&B.data[n*B.numCols]), 1);

	  nn = 1.0/whiteice::math::sqrt(whiteice::math::abs(nn));
	  
	  cblas_zdscal(B.numCols, nn,
		       (double*)(&B.data[n*B.numCols]), 1);
	}
	
      }
      else{
	// only for real valued basis
	
	vertex<T> a, b;
	a.resize(M);
	b.resize(M);
	
	for(unsigned int n=i;n<j;n++){
	  z = T(0);
	  T r = T(0);
	  
	  for(unsigned int k=i;k<n;k++){
	    r = T(0);
	    for(unsigned int u=0;u<M;u++)
	      r += B(k,u) * B(n,u); // conj (?)
	    
	    for(unsigned int u=0;u<M;u++)
	      z[u] += r * B(k, u);
	  }
	  
	  for(unsigned int u=0;u<M;u++)
	    B(n,u) -= z[u];
	  
	  r = T(0);
	  for(unsigned int u=0;u<M;u++)
	    r += B(n,u)*B(n,u); // conj (?)
	  
	  r = whiteice::math::sqrt(r);
	  
	  for(unsigned int u=0;u<M;u++)
	    B(n,u) = B(n,u) / r;
	}
      }
#endif
      
      return true;
    }
    
    
    // calculates gram-schimdt orthonormalization
    template <typename T>
    bool gramschmidt(std::vector< vertex<T> >& B,
		     const unsigned int i, const unsigned int j)
    {
      if(i>=j || j>B.size())
	return false;
      
      vertex<T> z;
      
      // B = [N->M vertex]
      const unsigned int M = B[0].size();
      z.resize(M);
      
      for(unsigned int n=i;n<j;n++){
	z.zero();
	T w = T(0.0);
	  
	for(unsigned int k=i;k<n;k++){
	  w = (B[k]*B[n])[0]; // inner product (conjugate other one)
	  z += w*B[k];
	}
	
	B[n] -= z;
	B[n].normalize();
      }
      
      return true;
    }
    
  }
}
