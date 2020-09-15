
#include "matrix.h"
#include "vertex.h"
#include "Log.h"

#include "outerproduct.h"

namespace whiteice
{
  namespace math
  {

    // calculates A = A + scalar*a*b^T
    template <typename T>
      bool addouterproduct(matrix<T>& A,
			   const T& scalar,
			   const vertex<T>& a,
			   const vertex<T>& b)
    {
      if(A.numRows != a.dataSize) return false;
      if(A.numCols != b.dataSize) return false;

#ifdef CUBLAS

      if(typeid(T) == typeid(blas_real<float>)){
	float s = 0.0;
	convert(s, scalar);
	
	auto e = cublasSger(cublas_handle,
			    A.numRows, A.numCols,
			    (const float*)&s, (const float*)a.data, 1,
			    (float*)b.data, 1,
			    (float*)A.data, A.numRows);
	gpu_sync();

	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("addouterprodut(): cublasSger() failed.");
	  throw CUDAException("CUBLAS cublasSger() call failed.");
	}		    
	
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double s = 0.0;
	convert(s, scalar);
	
	auto e = cublasDger(cublas_handle,
			    A.numRows, A.numCols,
			    (const double*)&s, (const double*)a.data, 1,
			    (double*)b.data, 1,
			    (double*)A.data, A.numRows);
	gpu_sync();
	
	if(e != CUBLAS_STATUS_SUCCESS){
	  whiteice::logging.error("addouterprodut(): cublasDger() failed.");
	  throw CUDAException("CUBLAS cublasDger() call failed.");
	}
	
      }
      else{
	// COLUMN MAJOR matrix in cuBLAS
	for(unsigned int r=0;r<A.numRows;r++){
	  for(unsigned int c=0;c<A.numCols;c++){
	    A[r + c*A.numRows] += scalar*a[r]*b[c];
	  }
	}
      }

#else
      if(typeid(T) == typeid(blas_real<float>)){
	float s = 0.0;
	convert(s, scalar);
	
	cblas_sger(CblasRowMajor, A.numRows, A.numCols,
		   s, (float*)a.data, 1, (float*)b.data, 1,
		   (float*)A.data, A.numCols);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double s = 0.0;
	convert(s, scalar);
	cblas_dger(CblasRowMajor, A.numRows, A.numCols,
		   s, (double*)a.data, 1, (double*)b.data, 1,
		   (double*)A.data, A.numCols);
      }
      else{
	for(unsigned int r=0;r<A.numRows;r++){
	  for(unsigned int c=0;c<A.numCols;c++){
	    A[r*A.numCols + c] += scalar*a[r]*b[c];
	  }
	}
      }
#endif

      return true;
    }


    template bool addouterproduct<float>(matrix<float>& A,
					 const float& scalar,
					 const vertex<float>& a,
					 const vertex<float>& b);

    template bool addouterproduct<double>(matrix<double>& A,
					  const double& scalar,
					  const vertex<double>& a,
					  const vertex<double>& b);
    
    template bool addouterproduct< blas_real<float> >(matrix< blas_real<float> >& A,
						      const blas_real<float>& scalar,
						      const vertex< blas_real<float> >& a,
						      const vertex< blas_real<float> >& b);
    
    template bool addouterproduct< blas_real<double> >(matrix< blas_real<double> >& A,
						       const blas_real<double>& scalar,
						       const vertex< blas_real<double> >& a,
						       const vertex< blas_real<double> >& b);

  }
}
