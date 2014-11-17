
#include <vector>
#include <typeinfo>
#include "linear_algebra.h"

#include "matrix.h"
#include "vertex.h"
#include "atlas.h"

// #define OPENBLAS 1

namespace whiteice
{
  namespace math
  {
    
    template bool gramschmidt< atlas_real<float> >
      (matrix< atlas_real<float> >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< atlas_real<double> >
      (matrix< atlas_real<double> >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< atlas_complex<float> >
      (matrix< atlas_complex<float> >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< atlas_complex<double> >
      (matrix< atlas_complex<double> >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt<float>
      (matrix<float>& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt<double>
      (matrix<double>& B, const unsigned int i, const unsigned int j);
    
    
    template bool gramschmidt< atlas_real<float> >
      (std::vector< vertex< atlas_real<float> > >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< atlas_real<double> >
      (std::vector< vertex< atlas_real<double> > >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< atlas_complex<float> >
      (std::vector< vertex< atlas_complex<float> > >& B, const unsigned int i, const unsigned int j);
    template bool gramschmidt< atlas_complex<double> >
      (std::vector< vertex< atlas_complex<double> > >& B, const unsigned int i, const unsigned int j);
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
      

      if(typeid(T) == typeid(atlas_real<float>)){
	
	for(unsigned int n=i;n<j;n++){
	  z = T(0);
	  atlas_real<float> w;
	  
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*numCols], B.data[n*numCols], numRows);
	    w = cblas_sdot(B.numRows, 
			   (float*)&(B.data[k*B.numCols]), 1,
			   (float*)&(B.data[n*B.numCols]), 1);
	    
	    // z += scal(w, data[k*numCols], numRows)
	    cblas_saxpy(B.numRows, *((float*)(&w)),
			(float*)&(B.data[k*B.numCols]), 1,
			(float*)z.data, 1);
	  }
	  
	  // B[n] -= z;
	  cblas_saxpy(B.numRows, -1.0f,
		      (float*)z.data, 1,
		      (float*)&(B.data[n*B.numCols]), 1);
	  
	  // normalize(B.data[n*B.numCols], B.numRows)
	  w = cblas_snrm2(B.numRows, (float*)&(B.data[n*B.numCols]), 1);
	  
	  if(w != 0.0f) w = 1.0f / w;
	  
	  cblas_sscal(B.numRows, *((float*)(&w)), (float*)&(B.data[n*B.numCols]), 1);
	}
	
      }
      else if(typeid(T) == typeid(atlas_complex<float>)){
	
	for(unsigned int n=i;n<j;n++){
	  z = T(0);
	  atlas_complex<float> w;
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*numCols], B.data[n*numCols], numRows);
#ifdef OPENBLAS
	    cblas_cdotc_sub(B.numRows, 
			    (float*)&(B.data[k*B.numCols]), 1,
			    (float*)&(B.data[n*B.numCols]), 1, (openblas_complex_float*)&w);
	    // (float*)&(B.data[n*B.numCols]), 1, (float*)&w);
#else
	    cblas_cdotc_sub(B.numRows, 
			    (float*)&(B.data[k*B.numCols]), 1,
			    // (float*)&(B.data[n*B.numCols]), 1, (openblas_complex_float*)&w);
			    (float*)&(B.data[n*B.numCols]), 1, (float*)&w);
#endif
	    
	    
	    // z += scal(w, B.data[k*numCols], numRows)
	    cblas_caxpy(B.numRows, (const float*)&w,
			(float*)&(B.data[k*B.numCols]), 1,
			(float*)z.data, 1);
	  }
	  
	  // B[n] -= z;
	  w = atlas_complex<float>(-1.0f);
	  cblas_caxpy(B.numRows, (const float*)&w,
		      (float*)z.data, 1,
		      (float*)&(B.data[n*B.numCols]), 1);
	  
	  // normalize(B.data[n*numCols], numRows)
	  float nn = cblas_scnrm2(B.numRows, (float*)&(B.data[n*B.numCols]), 1);
	  
	  cblas_csscal(B.numRows, nn, (float*)&(B.data[n*B.numCols]), 1);
	}
	
      }
      else if(typeid(T) == typeid(atlas_real<double>)){
	
	for(unsigned int n=i;n<j;n++){
	  z = T(0);
	  atlas_real<double> w;
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*B.numCols], B.data[n*B.numCols], B.numRows);
	    w = cblas_ddot(B.numRows, 
			   (double*)&(B.data[k*B.numCols]), 1,
			   (double*)&(B.data[n*B.numCols]), 1);
	    
	    // z += scal(w, B.data[k*numCols], numRows)
	    cblas_daxpy(B.numRows, *((double*)&w),
			(double*)&(B.data[k*B.numCols]), 1,
			(double*)z.data, 1);
	  }
	  
	  // B[n] -= z;
	  cblas_daxpy(B.numRows, -1.0,
		      (double*)z.data, 1,
		      (double*)&(B.data[n*B.numCols]), 1);
	  
	  // normalize(B.data[n*numCols], numRows)
	  w = cblas_dnrm2(B.numRows, (double*)&(B.data[n*B.numCols]), 1);
	  
	  cblas_dscal(B.numRows, *((double*)&w), 
		      (double*)&(B.data[n*B.numCols]), 1);
	}
	
      }
      else if(typeid(T) == typeid(atlas_complex<double>)){
	
	for(unsigned int n=i;n<j;n++){
	  z = T(0);
	  atlas_complex<double> w;
	  
	  for(unsigned int k=i;k<n;k++){
	    // w = dot(B.data[k*numCols], B.data[n*numCols], numRows);
#ifdef OPENBLAS
	    cblas_zdotc_sub(B.numRows, 
	      (double*)&(B.data[k*B.numCols]), 1,
	      (double*)&(B.data[n*B.numCols]), 1, (openblas_complex_double*)&w);
	    // (double*)&(B.data[n*B.numCols]), 1, (double*)&w);
#else
	    cblas_zdotc_sub(B.numRows, 
	      (double*)&(B.data[k*B.numCols]), 1,
	      // (double*)&(B.data[n*B.numCols]), 1, (openblas_complex_double*)&w);
	      (double*)&(B.data[n*B.numCols]), 1, (double*)&w);
#endif

	    
	    // z += scal(w, B.data[k*numCols], numRows)
	    cblas_zaxpy(B.numRows, (const double*)&w,
			(double*)(&B.data[k*B.numCols]), 1,
			(double*)z.data, 1);
	  }
	  
	  // B[n] -= z;
	  w = atlas_complex<double>(-1.0);
	  cblas_zaxpy(B.numRows, (const double*)&w,
		      (double*)z.data, 1,
		      (double*)&(B.data[n*B.numCols]), 1);
	  
	  // normalize(B.data[n*numCols], numRows)
	  double nn = cblas_dznrm2(B.numRows,
				   (double*)(&B.data[n*B.numCols]), 1);
	  
	  cblas_zdscal(B.numRows, nn,
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
