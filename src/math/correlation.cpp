
#ifndef correlation_cpp
#define correlation_cpp

#include "correlation.h"

#include "matrix.h"
#include "vertex.h"
#include "eig.h"

#include <typeinfo>
#include <string.h>

namespace whiteice
{
  namespace math
  {
    
    // calculates autocorrelation matrix from the given data
    template <typename T>
    bool autocorrelation(matrix<T>& R, const std::vector< vertex<T> >& data)
    {
      if(data.size() <= 0)
	return false;
      
      typename std::vector< vertex<T> >::const_iterator i = data.begin();
      const unsigned int N = data[0].size();
      
      R.resize(N, N);
      R.zero();
      
      T s = T(1.0f) / T(data.size());
      
      
      if(typeid(T) == typeid(blas_real<float>)){	
	
	while(i != data.end()){
	  cblas_sspr(CblasRowMajor, CblasUpper, N, 
		     *((float*)&s), (float*)(i->data), 1, (float*)R.data);
	  i++;
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<=(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memcpy(&(R.data[r*N + r]),
		 &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		 sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_scopy(N - i - 1, 
		      (float*)&(R.data[i*N + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*N + i]), N);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	while(i != data.end()){
	  cblas_chpr(CblasRowMajor, CblasUpper, N,
		     *((float*)&s), (float*)(i->data), 1, (float*)R.data);
	  i++;
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<=(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memcpy(&(R.data[r*N + r]),
		 &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		 sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_ccopy(N - i - 1, 
		      (float*)&(R.data[i*N + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*N + i]), N);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	while(i != data.end()){
	  cblas_dspr(CblasRowMajor, CblasUpper, N, 
		     *((double*)&s), (double*)(i->data), 1, (double*)(R.data));
	  i++;
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memcpy(&(R.data[r*N + r]),
		 &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		 sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_dcopy(N - i - 1, 
		      (double*)&(R.data[i*N + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*N + i]), N);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	while(i != data.end()){
	  cblas_zhpr(CblasRowMajor, CblasUpper, N,
		     *((double*)&s), (double*)(i->data), 1, (double*)R.data);
	  i++;
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memcpy(&(R.data[r*N + r]),
		 &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		 sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_zcopy(N - i - 1, 
		      (double*)&(R.data[i*N + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*N + i]), N);
      }
      else{ // generic code
	
	while(i != data.end()){
	  
	  R += (*i).outerproduct((*i));
	  i++;
	}
	
	R *= s;
	
      }      
      
      
      return true;
    }
    
    
    
    // calculates autocorrelation matrix from W matrix'es row vectors
    template <typename T>
    bool autocorrelation(matrix<T>& R, const matrix<T>& W)
    {
      // autocorrelation is R= s * W^t * W, where s is 
      // one divided by number of rows in W
      
      if(W.ysize() <= 0 || W.xsize() <= 0)
	return false;
      
      const unsigned int N = W.ysize(); // meaning is swapped from cblas_ssyrk() parameters
      const unsigned int K = W.xsize(); // (N is number of vectors here and K is dimension)
      
      R.resize(K, K);
      R.zero(); // not necessary..
      
      T s = T(1.0) / T((double)N);
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, K, N,
		    *((float*)&s), (float*)W.data, W.xsize(),
		    0.0f, (float*)R.data, R.xsize());

	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(K-1);i++)
	  cblas_scopy(K - i - 1, 
		      (float*)&(R.data[i*K + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*K + i]), K);

      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cherk(CblasRowMajor, CblasUpper, CblasTrans, K, N,
		    *((float*)&s), (float*)W.data, W.xsize(),
		    0.0f, (float*)R.data, R.xsize());
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(K-1);i++)
	  cblas_ccopy(K - i - 1, 
		      (float*)&(R.data[i*K + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*K + i]), K);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, K, N,
		    *((double*)&s), (double*)W.data, W.xsize(),
		    0.0, (double*)R.data, R.xsize());
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(K-1);i++)
	  cblas_dcopy(K - i - 1, 
		      (double*)&(R.data[i*K + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*K + i]), K);

      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zherk(CblasRowMajor, CblasUpper, CblasTrans, K, N,
		    *((double*)&s), (double*)W.data, W.xsize(),
		    0.0, (double*)R.data, R.xsize());
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(K-1);i++)
	  cblas_zcopy(K - i - 1, 
		      (double*)&(R.data[i*K + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*K + i]), K);
      }
      else{ // generic code
	
	matrix<T> V = W;
	V.transpose();
	
	R = V * W;
	
	R *= s;
	
      }      
      
      return true;
    }
    
    
    
    
    template <typename T>
    bool mean_covariance_estimate(vertex<T>& m, matrix<T>& R,
				  const std::vector< vertex<T> >& data)
    {
      if(data.size() <= 0)
	return false;
      
      typename std::vector< vertex<T> >::const_iterator i = data.begin();
      const unsigned int N = data[0].size();
      
      R.resize(N, N);
      R.zero();
      
      m.resize(N);
      m.zero();
      
      T s = T(1.0f) / T(data.size());

      
      if(typeid(T) == typeid(blas_real<float>)){	
	
	while(i != data.end()){
	  cblas_sspr(CblasRowMajor, CblasUpper, N, 
		     *((float*)&s), (float*)(i->data), 1, (float*)R.data);
	  m += *i;
	  
	  i++;
	}

	// converts packed symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<=(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memcpy(&(R.data[r*N + r]),
		 &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		 sizeof(T) * (i+1));
	}

	for(unsigned int i=0;i<(N-1);i++)
	  cblas_scopy(N - i - 1, 
		      (float*)&(R.data[i*N + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*N + i]), N);

	m *= s;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	while(i != data.end()){
	  cblas_chpr(CblasRowMajor, CblasUpper, N,
		     *((float*)&s), (float*)(i->data), 1, (float*)R.data);
	  m += *i;
	  
	  i++;
	}
	
	// converts packed symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<=(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memcpy(&(R.data[r*N + r]),
		 &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		 sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_ccopy(N - i - 1, 
		      (float*)&(R.data[i*N + 1 + i]), 1,
		      (float*)&(R.data[(i+1)*N + i]), N);
	
	m *= s;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	while(i != data.end()){
	  cblas_dspr(CblasRowMajor, CblasUpper, N, 
		     *((double*)&s), (double*)(i->data), 1, (double*)(R.data));
	  m += *i;
	  
	  i++;
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<=(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memcpy(&(R.data[r*N + r]),
		 &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		 sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_dcopy(N - i - 1, 
		      (double*)&(R.data[i*N + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*N + i]), N);
	
	m *= s;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	while(i != data.end()){
	  cblas_zhpr(CblasRowMajor, CblasUpper, N,
		     *((double*)&s), (double*)(i->data), 1, (double*)R.data);
	  m += *i;
	  
	  i++;
	}
	
	// converts symmetric matrix results into regular matrix data format
	for(unsigned int i=0;i<=(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memcpy(&(R.data[r*N + r]),
		 &(R.data[((N+1)*N)/2 - ((i+1)*(i+2))/2]),
		 sizeof(T) * (i+1));
	}
	
	// copy data from upper triangular part to lower
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_zcopy(N - i - 1, 
		      (double*)&(R.data[i*N + 1 + i]), 1,
		      (double*)&(R.data[(i+1)*N + i]), N);
	
	m *= s;
      }
      else{ // generic code
	
	while(i != data.end()){
	  
	  R += (*i).outerproduct((*i));
	  m += *i;
	  
	  i++;
	}
	
	R *= s;
	m *= s;
      }

      
      // removes mean from R = E[xx^t]

      for(unsigned int j=0;j<N;j++){
	for(unsigned int i=0;i<=j;i++){
	  T tmp = m[j]*m[i];
	  R(j,i) -= tmp;
	  if(j!=i)
	    R(i,j) -= tmp;
	}
      }

      
      return true;
    }
    
    
    
    
    template <typename T>
    bool mean_covariance_estimate(vertex<T>& m, matrix<T>& R, 
				  const std::vector< vertex<T> >& data,
				  const std::vector< whiteice::dynamic_bitset >& missing)
    {
      if(data.size() <= 0)
	return false;
      
      if(data.size() != missing.size())
	return false;
      
      const unsigned int D = data[0].size();
      
      // because some entries are missing calculating
      // E[x] and E[(x - m)(x - m)^t] must happen per dimension or correlation pair
      // basis and because some values are missing Ni or Nij
      // (samples used in estimate SUM(x_i), SUM(x_i*x_j)
      // 
      // however, if/when non-available entries are zeros they contribute nothing
      // to unnormalized sum estimate calculated normally. So normal methods
      // and 
      
      if(!autocorrelation(R, data))
	return false;

      // Ni = number of entries available
      matrix<T> RN(D,D);
      vertex<T> N(D);
      
      m.resize(D);
      m.zero();
      
      for(unsigned int i=0;i<missing.size();i++){
	for(unsigned int j=0;j<D;j++){
	  if(!missing[i][j])
	    N[j] += T(1.0);
	  
	  for(unsigned int k=0;k<D;k++)
	    if((!missing[i][j]) && (!missing[i][k]))
	      RN(j,k) += T(1.0);
	}
	
	m += data[i];
      }

      
      
      
      for(unsigned int i=0;i<D;i++)
	m[i] /= N[i];
      
      // rescales autocorrelation matrix
      for(unsigned int i=0;i<D;i++)
	for(unsigned int j=0;j<D;j++)
	  R(i,j) *= T(data.size())/RN(i,j);
      
      
      return true;
    }


    // calculates crosscorrelation matrix Cyx as well as mean values E[x], E[y]
    template <typename T>
    bool mean_crosscorrelation_estimate(vertex<T>& mx, vertex<T>& my, matrix<T>& Cyx,
					const std::vector< vertex<T> >& xdata,
					const std::vector< vertex<T> >& ydata)
    {
      // not optimized

      if(xdata.size() != ydata.size()) return false;
      if(xdata.size() == 0) return false;

      // calculates Ryx and mean and then Cyx = Ryx - my*mx^t

      mx.resize(xdata[0].size());
      my.resize(ydata[0].size());
      Cyx.resize(ydata[0].size(), xdata[0].size());
      mx.zero();
      my.zero();
      Cyx.zero();

      for(unsigned int i=0;i<xdata.size();i++){
	mx += xdata[i];
	my += ydata[i];

	auto hx = xdata[i];
	hx.hermite();
	for(unsigned int y=0;y<ydata[0].size();y++)
	  for(unsigned int x=0;x<xdata[0].size();x++)
	    Cyx(y,x) += ydata[i][y]*hx[x];
      }

      mx /= T(xdata.size());
      my /= T(xdata.size());
      Cyx /= T(xdata.size());

      auto hmx = mx;
      hmx.hermite();

      for(unsigned int y=0;y<ydata[0].size();y++)
	for(unsigned int x=0;x<xdata[0].size();x++)
	  Cyx(y,x) -= my[y]*hmx[x];

      return true;
    }
    
    
    // calculates PCA dimension reduction using symmetric eigenvalue decomposition
    template <typename T>
    bool pca(const std::vector< vertex<T> >& data, 
	     const unsigned int dimensions,
	     math::matrix<T>& PCA,
	     math::vertex<T>& m,
	     T& original_var, T& reduced_var,
	     bool regularizeIfNeeded)
    {
      matrix<T> C;

      if(mean_covariance_estimate(m, C, data) == false)
	return false;

      
      matrix<T> X;

      
      if(regularizeIfNeeded){
	// not optimized
	
	matrix<T> CC(C);
	T regularizer = T(0.0f);
	for(unsigned int j=0;j<CC.ysize();j++)
	  for(unsigned int i=0;i<CC.xsize();i++)
	    regularizer += abs(CC(j,i));
	regularizer /= (CC.ysize()*CC.xsize());
	regularizer *= T(0.001);
	if(regularizer <= T(0.0f)) regularizer = T(0.0001f);
	
	unsigned int counter = 0;
	const unsigned int CLIMIT = 20;
	
	
	while(symmetric_eig(CC, X, true) == false){
	  CC = C;
	  
	  for(unsigned int i=0;i<CC.ysize();i++)
	    CC(i,i) += regularizer;
	  
	  regularizer *= T(2.0f);
	  counter++;
	  
	  if(counter >= CLIMIT)
	    return false;
	}

	C = CC;
      }
      else{
	if(symmetric_eig(C, X, true) == false)
	  return false;
      }

      // C = X * D * X^t

      // we want to keep only the top variance vectors

      original_var = T(0.0f);
      reduced_var  = T(0.0f);

      for(unsigned int i=0;i<X.xsize();i++){
	if(i<dimensions) reduced_var += C(i,i);
	original_var += C(i,i);
      }

      if(X.submatrix(PCA, 0, 0, dimensions, X.xsize()) == false)
	return false;

      PCA.transpose();

      // TODO: tests that PCA matrix works (for debugging)

      return true;
    }
    
    
    
    // explicit template instantations
    
    template bool autocorrelation<float>(matrix<float>& R, const std::vector< vertex<float> >& data);
    template bool autocorrelation<double>(matrix<double>& R, const std::vector< vertex<double> >& data);
    
    template bool autocorrelation<blas_real<float> >(matrix<blas_real<float> >& R,
						      const std::vector< vertex<blas_real<float> > >& data);
    template bool autocorrelation<blas_real<double> >(matrix<blas_real<double> >& R,
						       const std::vector< vertex<blas_real<double> > >& data);
    template bool autocorrelation<blas_complex<float> >(matrix<blas_complex<float> >& R,
							 const std::vector< vertex<blas_complex<float> > >& data);
    template bool autocorrelation<blas_complex<double> >(matrix<blas_complex<double> >& R,
							  const std::vector< vertex<blas_complex<double> > >& data);
    template bool autocorrelation<complex<float> >(matrix<complex<float> >& R,
						   const std::vector< vertex<complex<float> > >& data);
    template bool autocorrelation<complex<double> >(matrix<complex<double> >& R,
						    const std::vector< vertex<complex<double> > >& data);    
    // template bool autocorrelation<int>(matrix<int>& R, const std::vector< vertex<int> >& data);
    
    template bool autocorrelation<float>(matrix<float>& R, const matrix<float>& W);
    template bool autocorrelation<double>(matrix<double>& R, const matrix<double>& W);
    
    template bool autocorrelation<blas_real<float> >(matrix<blas_real<float> >& R,
						      const matrix<blas_real<float> >& W);
    template bool autocorrelation<blas_real<double> >(matrix<blas_real<double> >& R,
						       const matrix<blas_real<double> >& W);
    template bool autocorrelation<blas_complex<float> >(matrix<blas_complex<float> >& R,
							 const matrix<blas_complex<float> >& W);
    template bool autocorrelation<blas_complex<double> >(matrix<blas_complex<double> >& R,
							  const matrix<blas_complex<double> >& W);
    
    template bool mean_covariance_estimate< float >
    (vertex< float >& m, matrix< float >& R,
     const std::vector< vertex< float > >& data);

    template bool mean_covariance_estimate< double >
    (vertex< double >& m, matrix< double >& R,
     const std::vector< vertex< double > >& data);
    
    template bool mean_covariance_estimate< blas_real<float> >
    (vertex< blas_real<float> >& m, matrix< blas_real<float> >& R,
     const std::vector< vertex< blas_real<float> > >& data);
    
    template bool mean_covariance_estimate< blas_real<double> >
    (vertex< blas_real<double> >& m, matrix< blas_real<double> >& R,
     const std::vector< vertex< blas_real<double> > >& data);
    
    template bool mean_covariance_estimate< blas_complex<float> >
    (vertex< blas_complex<float> >& m, matrix< blas_complex<float> >& R,
     const std::vector< vertex< blas_complex<float> > >& data);
    
    template bool mean_covariance_estimate< blas_complex<double> > 
    (vertex< blas_complex<double> >& m, matrix< blas_complex<double> >& R,
     const std::vector< vertex< blas_complex<double> > >& data);
    
    
    
    template bool mean_covariance_estimate< float >
    (vertex< float >& m, matrix< float >& R,
     const std::vector< vertex< float > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    
    template bool mean_covariance_estimate< double >
    (vertex< double >& m, matrix< double >& R,
     const std::vector< vertex< double > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    
    template bool mean_covariance_estimate< blas_real<float> >
    (vertex< blas_real<float> >& m, matrix< blas_real<float> >& R,
     const std::vector< vertex< blas_real<float> > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);

    template bool mean_covariance_estimate< blas_real<double> >
    (vertex< blas_real<double> >& m, matrix< blas_real<double> >& R,
     const std::vector< vertex< blas_real<double> > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    
    template bool mean_covariance_estimate< blas_complex<float> >
    (vertex< blas_complex<float> >& m, matrix< blas_complex<float> >& R,
     const std::vector< vertex< blas_complex<float> > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);
    
    template bool mean_covariance_estimate< blas_complex<double> >
    (vertex< blas_complex<double> >& m, matrix< blas_complex<double> >& R,
     const std::vector< vertex< blas_complex<double> > >& data,
     const std::vector< whiteice::dynamic_bitset >& missing);

    
    template bool mean_crosscorrelation_estimate< float >
    (vertex< float >& mx, vertex< float >& my, matrix< float >& Cyx,
     const std::vector< vertex<float> >& xdata,
     const std::vector< vertex<float> >& ydata);

    template bool mean_crosscorrelation_estimate< double >
    (vertex< double >& mx, vertex< double >& my, matrix< double >& Cyx,
     const std::vector< vertex<double> >& xdata,
     const std::vector< vertex<double> >& ydata);

    
    template bool mean_crosscorrelation_estimate< blas_real<float> >
    (vertex< blas_real<float> >& mx, vertex< blas_real<float> >& my, matrix< blas_real<float> >& Cyx,
     const std::vector< vertex< blas_real<float> > >& xdata,
     const std::vector< vertex< blas_real<float> > >& ydata);

    template bool mean_crosscorrelation_estimate< blas_real<double> >
    (vertex< blas_real<double> >& mx, vertex< blas_real<double> >& my, matrix< blas_real<double> >& Cyx,
     const std::vector< vertex< blas_real<double> > >& xdata,
     const std::vector< vertex< blas_real<double> > >& ydata);

    template bool mean_crosscorrelation_estimate< blas_complex<float> >
    (vertex< blas_complex<float> >& mx, vertex< blas_complex<float> >& my, matrix< blas_complex<float> >& Cyx,
     const std::vector< vertex< blas_complex<float> > >& xdata,
     const std::vector< vertex< blas_complex<float> > >& ydata);

    template bool mean_crosscorrelation_estimate< blas_complex<double> >
    (vertex< blas_complex<double> >& mx, vertex< blas_complex<double> >& my, matrix< blas_complex<double> >& Cyx,
     const std::vector< vertex< blas_complex<double> > >& xdata,
     const std::vector< vertex< blas_complex<double> > >& ydata);

    
    
    template bool pca<float>
      (const std::vector< vertex<float> >& data, 
       const unsigned int dimensions,
       math::matrix<float>& PCA,
       math::vertex<float>& m,
       float& original_var, float& reduced_var,
       bool regularizeIfNeeded);

    template bool pca<double>
      (const std::vector< vertex<double> >& data, 
       const unsigned int dimensions,
       math::matrix<double>& PCA,
       math::vertex<double>& m,
       double& original_var, double& reduced_var,
       bool regularizeIfNeeded);

    template bool pca< blas_real<float> >
      (const std::vector< vertex< blas_real<float> > >& data, 
       const unsigned int dimensions,
       math::matrix< blas_real<float> >& PCA,
       math::vertex< blas_real<float> >& m,
       blas_real<float>& original_var, blas_real<float>& reduced_var,
       bool regularizeIfNeeded);

    template bool pca< blas_real<double> >
      (const std::vector< vertex< blas_real<double> > >& data, 
       const unsigned int dimensions,
       math::matrix< blas_real<double> >& PCA,
       math::vertex< blas_real<double> >& m,
       blas_real<double>& original_var, blas_real<double>& reduced_var,
       bool regularizeIfNeeded);
    
  };
};

#endif
