
#ifndef matrix_cpp
#define matrix_cpp

#include "matrix.h"
#include "gcd.h"

#include <list>
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <typeinfo>

#include <string.h>
#include <stdlib.h>

using namespace std;


namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    matrix<T>::matrix(const unsigned int ysize,
		      const unsigned int xsize)
    {
      numRows = 0;
      numCols = 0;
      data = 0;

#ifdef BLAS_MEMALIGN
      // NOTE: electric fence don't know about posix_memalign()
      posix_memalign((void**)&data,
		     (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		     ysize*xsize*sizeof(T));
#else
      data = (T*)malloc(ysize*xsize*sizeof(T));
#endif
      
      if(!data) throw std::bad_alloc();
      
      memset(data, 0, ysize*xsize*sizeof(T));
      
      numRows = ysize;
      numCols = xsize;
      
      compressor = 0;
    }
    
    
    template <typename T>
    matrix<T>::matrix(const matrix<T>& M)
    {
      data = 0;

#ifdef BLAS_MEMALIGN
      // electric fence don't know about posix_memalign()
      posix_memalign((void**)&data,
		     (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		     M.numRows*M.numCols*sizeof(T));
#else
      data = (T*)malloc(M.numRows*M.numCols*sizeof(T));
#endif
      
      if(!data) throw std::bad_alloc();
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	cblas_scopy(M.numRows*M.numCols,
		    (const float*)M.data, 1,
		    (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cblas_ccopy(M.numRows*M.numCols,
		    (const float*)M.data, 1,
		    (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dcopy(M.numRows*M.numCols,
		    (const double*)M.data, 1,
		    (double*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cblas_zcopy(M.numRows*M.numCols,
		    (const double*)M.data, 1,
		    (double*)data, 1);
      }
      else{ // generic memcpy
	memcpy(data, M.data, M.numRows*M.numCols*sizeof(T));
      }
      
      
      numRows = M.numRows;
      numCols = M.numCols;
      
      compressor = M.compressor;
    }
    
#if 0    
    template <typename T>
    matrix<T>::matrix(matrix<T>&& t)
    {
      this->data = t.data;
      this->numRows = t.numRows;
      this->numCols = t.numCols;
      this->compressor = t.compressor;
      
      t.data = nullptr;
      t.compressor = nullptr;
    }
#endif
    
    template <typename T>
    matrix<T>::matrix(const vertex<T>& diagonal)
    {
      data = 0;

#ifdef BLAS_MEMALIGN
      // electric fence don't know about posix_memalign()
      posix_memalign((void**)&data,
		     (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		     diagonal.size()*diagonal.size()*sizeof(T));
#else
      data = (T*)malloc(diagonal.size()*diagonal.size()*sizeof(T));
#endif
      
      if(!data) throw std::bad_alloc();
      
      memset(data, 0, diagonal.size()*diagonal.size());
      numCols = diagonal.size();
      numRows = diagonal.size();
      
      unsigned int index = 0;
      for(unsigned int i=0;i<diagonal.size();i++){
	data[index] = diagonal[i];
	index += numCols + 1;
      }
      
      compressor = 0;
    }
    
    
    template <typename T>
    matrix<T>::~matrix(){
      if(data) free(data);
      if(compressor) delete compressor;
    }

    /**********************************************************************/
    
    
    template <typename T>
    matrix<T> matrix<T>::operator+(const matrix<T>& M) const
      throw(illegal_operation)
    {
      if(M.numCols != numCols ||
	 M.numRows != numRows)
	throw illegal_operation("'+' operator: matrix size mismatch");
      
      matrix<T> R(*this);
      
      for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	R.data[i] += M.data[i];
      
      return R;
    }
    
    
    template <typename T>
    matrix<T> matrix<T>::operator-(const matrix<T>& M) const
      throw(illegal_operation)
    {
      if(M.numCols != numCols || M.numRows != numRows)
	throw illegal_operation("'-' operator: matrix size mismatch");
      
      matrix<T> R(*this);
      
      for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	R.data[i] -= M.data[i];
      
      return R;
    }
    
    
    template <typename T>
    matrix<T> matrix<T>::operator*(const matrix<T>& M) const
      throw(illegal_operation)
    {	
      if(numCols != M.numRows)
	throw illegal_operation("'*' operator: matrix size mismatch");
      
      // uses cblas_Xgemm optimized matrix multiplication
      
      matrix<T> R(numRows, M.numCols);
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    1.0f, (float*)data, numCols, (float*)M.data, M.numCols,
		    0.0f, (float*)R.data, R.numCols);
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> a, b;
	a = 1.0f; b = 0.0f;
	
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    (float*)(&a), (float*)data, numCols, (float*)M.data, M.numCols,
		    (float*)(&b), (float*)R.data, R.numCols);
	return R;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    1.0, (double*)data, numCols, (double*)M.data, M.numCols,
		    0.0, (double*)R.data, R.numCols);
	return R;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> a, b;
	a = 1.0; b = 0.0;
	
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    (double*)(&a), (double*)data, numCols, (double*)M.data, M.numCols,
		    (double*)(&b), (double*)R.data, R.numCols);
	return R;
      }
      else{ // generic matrix multiplication
	
	// zero matrix	
	
	unsigned int rindex = 0;
	for(unsigned int j=0;j<numRows;j++){
	  for(unsigned int i=0;i<M.numCols;i++,rindex++){
	    for(unsigned int k=0;k<M.numRows;k++){
	      R[rindex] += 
		data[j*numCols + k] * M[k*M.numCols + i];
	    }
	  }
	}
      	
	
	return R;
      }      
    }
    
    
    
    template <typename T>
    matrix<T> matrix<T>::operator/(const matrix<T>& M) const
      throw(illegal_operation)
    {
      matrix<T> R(*this);
      R /= M;
      return R;
    }
    
    
    template <typename T>
    matrix<T> matrix<T>::operator!() const throw(illegal_operation){    
      throw illegal_operation("'!'-operator");
    }
    
    
    template <typename T>
    matrix<T> matrix<T>::operator-() const
      throw(illegal_operation)
    {
      matrix<T> M(ysize(), xsize());
      
      for(unsigned int i=0;i<numRows*numCols;i++)
	M.data[i] = -data[i];
      
      return M;
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator+=(const matrix<T>& M)
      throw(illegal_operation)
    {
      if(M.numCols != numCols || M.numRows != numRows)
	throw illegal_operation("'+=' operator: matrix size mismatch");
      
      for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	data[i] += M.data[i];
      
      return (*this);
    }
  
    
    template <typename T>
    matrix<T>& matrix<T>::operator-=(const matrix<T>& M)
      throw(illegal_operation)
    {
      if(M.numCols != numCols || M.numRows != numRows)
	throw illegal_operation("'-=' operator: matrix size mismatch");

      for(unsigned int i=0;i<M.numCols*M.numRows;i++)
	data[i] -= M.data[i];
      
      return (*this);
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator*=(const matrix<T>& M)
      throw(illegal_operation)
    {
      if(numCols != M.numRows)
	throw illegal_operation("'*=' operator: matrix size mismatch");
      
      // uses cblas_Xgemm optimized matrix multiplication
      
      matrix<T> R(numRows, M.numCols);
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    1.0f, (float*)data, numCols,
		    (float*)M.data, M.numCols,
		    0.0f, (float*)R.data, R.numCols);
	
	// if(!resize_x(R.numCols)) throw bad_alloc();
	// memcpy(data, R.data, numRows*numCols*sizeof(T));
	
	this->numCols = M.numCols;
	free(data);
	data = R.data;
	R.data = nullptr;
	
	return (*this);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> a, b;
	a = 1.0f; b = 0.0f;
	
	cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    (float*)(&a), (float*)data, numCols, (float*)M.data, M.numCols,
		    (float*)(&b), (float*)R.data, R.numCols);
	
	if(!resize_x(R.numCols)) throw bad_alloc();
	memcpy(data, R.data, numRows*numCols*sizeof(T));
	
	return (*this);	
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    1.0, (double*)data, numCols, (double*)M.data, M.numCols,
		    0.0, (double*)R.data, R.numCols);
	
	
	if(!resize_x(R.numCols)) throw bad_alloc();
	memcpy(data, R.data, numRows*numCols*sizeof(T));
	
	return (*this);	
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> a, b;
	a = 1.0; b = 0.0;	
	
	cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		    numRows, M.numCols, M.numRows,
		    (double*)(&a), (double*)data, numCols, (double*)M.data, M.numCols,
		    (double*)(&b), (double*)R.data, R.numCols);
	
	if(!resize_x(R.numCols)) throw bad_alloc();
	memcpy(data, R.data, numRows*numCols*sizeof(T));
	
	return (*this);	
      }
      else{ // generic matrix multiplication
	
	// zero matrix	
	
	unsigned int rindex = 0;
	for(unsigned int j=0;j<numRows;j++){
	  for(unsigned int i=0;i<M.numCols;i++,rindex++){
	    for(unsigned int k=0;k<M.numRows;k++){
	      R[rindex] += 
		data[j*numCols + k] * M[k*M.numCols + i];
	    }
	  }
	}
      	
	
	if(!resize_x(R.numCols)) throw bad_alloc();
	memcpy(data, R.data, numRows*numCols*sizeof(T));
	
	return (*this);
      }
      
    }
    
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator/=(const matrix<T>& m) throw(illegal_operation)
    {
      // C BLAS OPTIMIZE ("/" operator, too)
      matrix<T> n(m);

      if(!n.inv())
	throw illegal_operation("Cannot 'divide' with singular matrix");
      
      (*this) *= n;
      
      return (*this);
    }
    

    template <typename T>
    matrix<T>& matrix<T>::operator=(const matrix<T>& M) throw(illegal_operation)
    {
      if(this != &M){ // no self-assignment
	if(M.numCols != numCols) resize_x(M.numCols);
	if(M.numRows != numRows) resize_y(M.numRows);
	
	memcpy(data, M.data, sizeof(T)*numCols*numRows);
      }
      
      return (*this);
    }
    
#if 0    
    template <typename T>
    matrix<T>& matrix<T>::operator=(matrix<T>&& t) throw(illegal_operation)
    {
      if(this == &t) return *this; // self-assignment
      
      // printf("matrix&& operator=\n"); fflush(stdout);
      
      if(this->data) free(this->data);
      if(this->compressor) delete (this->compressor);
      
      this->data = std::move(t.data);
      this->numRows = std::move(t.numRows);
      this->numCols = std::move(t.numCols);
      this->compressor = std::move(t.compressor);
      
      t.data = nullptr;
      t.compressor = nullptr;
      
      return *this;
    }
#endif
    
    
    
    template <typename T>
    bool matrix<T>::operator==(const matrix<T>& M) const
      throw(uncomparable)
    {
      if(M.numCols != numCols || M.numRows != numRows)
	return false; // throw illegal_operation("'==' operator: matrix size mismatch");
      
      if(typeid(T) == typeid(blas_real<float>)    ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>)   ||
	 typeid(T) == typeid(blas_complex<double>)){
	
	return (memcmp(data, M.data, numCols*numRows*sizeof(T)) == 0);
      }
      else{
	for(unsigned int i=0;i<numCols*numRows;i++)
	  if(data[i] != M.data[i]) return false;
      }
      
      return true;
    }
    
    
    template <typename T>
    bool matrix<T>::operator!=(const matrix<T>& M) const
      throw(uncomparable)
    {
      if(M.numCols != numCols || M.numRows != numRows)
	return true; // throw illegal_operation("'!=' operator: matrix size mismatch");
      
      if(numCols*numRows <= 0) return false;
      
      if(typeid(T) == typeid(blas_real<float>)    ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>)   ||
	 typeid(T) == typeid(blas_complex<double>)){
	
	return (memcmp(data, M.data, numCols*numRows*sizeof(T)) != 0);
      }
      else{
	for(unsigned int i=0;i<numCols*numRows;i++)
	  if(data[i] == M.data[i]) return false;
      }
      
      return true;
    }
    
    
    template <typename T>
    bool matrix<T>::operator>=(const matrix<T>& M) const
      throw(uncomparable)
    {
      if(M.numCols != 1 || numCols != 1 || M.numRows != 1 || numRows != 1)
	throw illegal_operation("matrix '>=': not a 1x1 matrix ");
      
      return (data[0] < M.data[0]);
    }
    
    
    template <typename T>
    bool matrix<T>::operator<=(const matrix<T>& M) const
      throw(uncomparable)
    {
      if(M.numCols != 1 || numCols != 1 || M.numRows != 1 || numRows != 1)
	throw illegal_operation("matrix '<=': not a 1x1 matrix");
      
      return (data[0] <= M.data[0]);
    }
    
    
    template <typename T>
    bool matrix<T>::operator< (const matrix<T>& M) const
      throw(uncomparable)
    {
      if(M.numCols != 1 || numCols != 1 || M.numRows != 1 || numRows != 1)
	throw illegal_operation("matrix  '<': not a 1x1 matrix");
      
      return (data[0] < M.data[0]);
    }
    
  
    template <typename T>
    bool matrix<T>::operator> (const matrix<T>& M) const
      throw(uncomparable)
    {
      if(M.numCols != 1 || numCols != 1 || M.numRows != 1 || numRows != 1)
	throw illegal_operation("matrix  '>': not a 1x1 matrix");
      
      return (data[0] > M.data[0]);
    }

    
    /***************************************************/
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator=(const T& s) throw(illegal_operation)
    {
      const unsigned int LIMIT = numRows*numCols;
      
      for(unsigned int i=0;i<LIMIT;i++)
	data[i] = s;
      
      return (*this);
    }
    
    
    
    template <typename T>
    matrix<T>  matrix<T>::operator* (const T& s) const throw()
    {      
      matrix<T> M(numRows,numCols);
      const unsigned int MSIZE = numRows*numCols;
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(MSIZE, *((float*)&s), (float*)data, 1, (float*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_caxpy(MSIZE, (const float*)&s, (float*)data, 1, (float*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(MSIZE, *((double*)&s), (double*)data, 1, (double*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zaxpy(MSIZE, (const double*)&s, (double*)data, 1, (double*)M.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<MSIZE;i++)
	  M.data[i] = data[i]*s;
      }
      
      return M;
    }
    
    
    template <typename T>
    matrix<T> operator*(const T& s, const matrix<T>& N)
      throw(std::invalid_argument)
    {
      matrix<T> M(N.numRows, N.numCols);
      const unsigned int MSIZE = N.numRows*N.numCols;
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(MSIZE, *((float*)&s), (float*)N.data, 1, (float*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_caxpy(MSIZE, (const float*)&s, (float*)N.data, 1, (float*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(MSIZE, *((double*)&s), (double*)N.data, 1, (double*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zaxpy(MSIZE, (const double*)&s, (double*)N.data, 1, (double*)M.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<MSIZE;i++)
	  M.data[i] = N.data[i]*s;
      }
      
      
      return M;
    }
    
    
    template <typename T>
    matrix<T>  matrix<T>::operator/ (const T& s) const throw(std::invalid_argument)
    {
      matrix<T> M(numRows, numCols);
      const unsigned int MSIZE = numRows*numCols;
      T ss = T(1.0)/s;
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(MSIZE, *((float*)&ss), (float*)data, 1, (float*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_caxpy(MSIZE, (const float*)&ss, (float*)data, 1, (float*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(MSIZE, *((double*)&ss), (double*)data, 1, (double*)M.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zaxpy(MSIZE, (const double*)&ss, (double*)data, 1, (double*)M.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<MSIZE;i++)
	  M.data[i] = data[i]*ss;
      }      
      
      return M;
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator*=(const T& s) throw()
    {
      const unsigned int MSIZE = numRows*numCols;
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(MSIZE, *((float*)&s), (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cscal(MSIZE, (const float*)&s, (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(MSIZE, *((double*)&s), (double*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zscal(MSIZE, (const double*)&s, (double*)data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<MSIZE;i++)
	  data[i] *= s;
      }
      
      return (*this);
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::operator/=(const T& s) throw(std::invalid_argument)
    {
      const unsigned int MSIZE = numRows*numCols;
      T ss = T(1.0)/s;
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(MSIZE, *((float*)&ss), (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cscal(MSIZE, (const float*)&ss, (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(MSIZE, *((double*)&ss), (double*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zscal(MSIZE, (const double*)&ss, (double*)data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<MSIZE;i++)
	  data[i] *= ss;
      }      
      
      return (*this);      
    }
    
    
    /***************************************************/
    
    template <typename T>
    vertex<T> matrix<T>::operator*(const vertex<T>& v) const
      throw(std::invalid_argument)
    {
      if(v.size() == 0)
	throw std::invalid_argument("multiply: incompatible vertex/matrix sizes");
      if(numCols != v.size())
	throw std::invalid_argument("multiply: incompatible vertex/matrix sizes");
      
      vertex<T> r(numRows);
      
      // BLAS level 2
      // uses optimized cblas_Xgemv() functions
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sgemv(CblasRowMajor, CblasNoTrans,
		    numRows, numCols,
		    1.0f, (float*)data, numCols, (float*)v.data, 1,
		    0.0f, (float*)r.data, 1);
	
	return r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> a, b;
	a = 1.0f; b = 0.0f;
	
	cblas_cgemv(CblasRowMajor, CblasNoTrans,
		    numRows, numCols,
		    (float*)(&a), (float*)data, numCols, (float*)v.data, 1,
		    (float*)(&b), (float*)r.data, 1);
	
	return r;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dgemv(CblasRowMajor, CblasNoTrans,
		    numRows, numCols,
		    1.0, (double*)data, numCols, (double*)v.data, 1,
		    0.0, (double*)r.data, 1);
	
	return r;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> a, b;
	a = 1.0f; b = 0.0f;	
	
	cblas_zgemv(CblasRowMajor, CblasNoTrans,
		    numRows, numCols,
		    (double*)(&a), (double*)data, numCols, (double*)v.data, 1,
		    (double*)(&b), (double*)r.data, 1);
	
	return r;
      }
      else{ // generic matrix * vertex code
	
	unsigned int k = 0;
	for(unsigned int j=0;j<r.size();j++){
	  for(unsigned int i=0;i<numCols;i++,k++)
	    r[j] += data[k]*v.data[i];
	}
	
	return r;
      }
    }
    
    
    
    /***************************************************/

    // crossproduct matrix M(z): M(z) * y = z x y
    template <typename T>
    matrix<T>& matrix<T>::crossproduct(const vertex<T>& v) throw(std::domain_error)
    {
      if(v.size() != 3)
	throw std::out_of_range("crossproduct() requires 3 dimensions");
      
      if(numRows != 3 || numCols != 3)
	if(!resize(3,3)) throw std::bad_alloc();
      
      
      (*this)(0,0) = T(0);
      (*this)(0,1) = T(-v[2]);
      (*this)(0,2) = T( v[1]);
      (*this)(1,0) = T( v[2]);
      (*this)(1,1) = T(0);
      (*this)(1,2) = T(-v[0]);
      (*this)(2,0) = T(-v[1]);
      (*this)(2,1) = T( v[0]);
      (*this)(2,2) = T(0);
  
      return *this;
    }
    
    
    // euclidean rotation XYZ matrix
    template <typename T>
    matrix<T>& matrix<T>::rotation(const T& xr, const T& yr, const T& zr) throw()
    {
      if( (xsize() != 3 && ysize() != 3) ||
	  (xsize() != 4 && ysize() != 4) ){
	if(!resize_y(4)) throw std::bad_alloc();
	if(!resize_x(4)) throw std::bad_alloc();
      }
      
      T a(cos(xr));
      T b(sin(xr));
      T c(cos(yr));
      T d(sin(yr));
      T e(cos(zr));
      T f(sin(zr));
      
      
      (*this)(0,0) = T( c*e);
      (*this)(0,1) = T(-c*f);
      (*this)(0,2) = T( d);
      (*this)(1,0) = T( b*d*e + a*f);
      (*this)(1,1) = T(-b*d*f + a*e);
      (*this)(1,2) = T(-b*c);
      (*this)(2,0) = T(-a*d*e + b*f);
      (*this)(2,1) = T( a*d*f + b*e);
      (*this)(2,2) = T( a*c);
      
      if(ysize() == 4){
	(*this)(0,3) = T(0);
	(*this)(1,3) = T(0);
	(*this)(2,3) = T(0);
	(*this)(3,0) = T(0);
	(*this)(3,1) = T(0);
	(*this)(3,2) = T(0);
	(*this)(3,3) = T(1);      
      }
      
      return *this;
    }
    
    
    // 4x4 translation matrix
    template <typename T>
    matrix<T>& matrix<T>::translation(const T& dx, const T& dy, const T& dz) throw()
    {
      if(ysize() != 4){
	if(!resize_y(4)) throw std::bad_alloc();
	if(!resize_x(4)) throw std::bad_alloc();
      }
      
      for(unsigned int j=0;j<3;j++)
	for(unsigned int i=0;i<4;i++){
	  if(j == i) (*this)(j,i) = T(1);
	  else (*this)(j,i) = T(0);
	}
      
      (*this)(3,0) = T(dx);
      (*this)(3,1) = T(dy);
      (*this)(3,2) = T(dz);
      (*this)(3,3) = T(1);
      
      return (*this);
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::abs() throw()
    {
      const unsigned int N = numRows*numCols;
      
      for(unsigned int i=0;i<N;i++)
	data[i] = whiteice::math::abs(data[i]);
	  
      return (*this);
    }
    
    
  
    template <typename T>
    matrix<T>& matrix<T>::transpose() throw()
    {
      const matrix<T> A(*this);
      this->resize(A.xsize(), A.ysize());
      
      // divide matrix into 16*16 blocks => 2*16 = 32 cache lines are in use
      // (small enough for most platforms)
      
      const unsigned int xleft = A.xsize() & 0x0F;
      const unsigned int yleft = A.ysize() & 0x0F;
      const unsigned int ys = A.ysize() & 0xF0;
      const unsigned int xs = A.xsize() & 0xF0;
      
      // transposes block by block
      // (so we hopefully are all the time within cache limits)
      unsigned int y, x;
      
      for(y=0;y<ys;y+=16){
	
	for(x=0;x<xs;x+=16){
	  
	  for(unsigned int j=0;j<16;j++){
	    for(unsigned int i=0;i<16;i++){
	      
	      (*this)(x + i, y + j) = A(y + j, x + i);
	    }
	  }
	}
	
	if(xleft){
	  
	  for(unsigned int j=0;j<16;j++){
	    for(unsigned int i=0;i<xleft;i++){
	      
	      (*this)(x + i, y + j) = A(y + j, x + i);
	      
	    }
	  }
	}
      }
      
      
      if(yleft){
	
	for(x=0;x<xs;x+=16){
	  
	  for(unsigned int j=0;j<yleft;j++){
	    for(unsigned int i=0;i<16;i++){
	      
	      (*this)(x + i, y + j) = A(y + j, x + i);
	    }
	  }
	}
	
	if(xleft){
	  
	  for(unsigned int j=0;j<yleft;j++){
	    for(unsigned int i=0;i<xleft;i++){
	      
	      (*this)(x + i, y + j) = A(y + j, x + i);
	      
	    }
	  }
	}
      }
      
      
      return *this;
    }
    
    
    
    template <typename T>
    T matrix<T>::det() const throw(std::logic_error)
    {
      if(ysize() != xsize())
	throw std::logic_error("matrix::determinate() - non square matrix");
                  
      const unsigned int N = numRows;
      
      // TODO: calculate faster with better algorithms and
      // fast matrix multiplication
      
      // calculates determinant using
      // gauss-jordan elimination
      // (upper triangle matrix 
      // -> multiplication of diagonal is determinate)
      
      matrix<T> copy(*this); // copy of matrix
      T det = T(1);
      
      for(unsigned int i=0;i<N;i++)
      {
	if(copy(i,i) == T(0)) // resort
	{
	  // (tries to) finds non-zero entry
	  for(unsigned int j=i+1;j<N;j++)
	  {
	    
	    if(copy(j,i) != T(0)) // swaps values
	    {
	      for(unsigned int k=i;k<N;k++) // starts from i: values before it are zero
		std::swap<T>(copy(j,k), copy(i,k));
	      
	      det *= T(-1);
	    }
	  }
	}
	
	
	// diagonal entry is zero -> det = 0
	if(copy(i,i) == T(0)) return T(0);
	
	det *= copy(i,i);
	
	// sets a_ii element to 1
	// starts from i: values before it are zero
	for(unsigned int k=i+1;k<N;k++)
	  copy(i,k) /= copy(i,i);
	
	copy(i,i) = T(1);
	
	// eliminates lower rows
	for(unsigned int j=i+1;j<N;j++)
	{
	  // i:th element will become zero in a process
	  T l = copy(j,i);
	  
	  // stars from i: values before it are already zero
	  for(unsigned int k=i;k<numCols;k++){
	    copy(j,k) -= l*copy(i,k);
	  }
	}
	
	
      }
      
      return det;
    }
    
    
    
    template <typename T>
    bool  matrix<T>::inv() throw()
    {
      // simple and slow: gaussian elimination - works for small matrixes
      // big ones: start to use atlas (don't bother to reinvent wheel)
      
      if(ysize() != xsize())
	return false;
      
      
      const unsigned int N = numRows;
      
      // gauss-jordan elimination
      
      matrix<T> copy(*this);
      this->identity();
      
      for(unsigned int i=0;i<N;i++){
	if(copy(i,i) == T(0.0)){ // resort
	  // (tries to) finds non-zero entry
	  bool singular = true;
	  
	  for(unsigned int j=i+1;j<N;j++){
	    if(copy(j,i) != T(0.0)){ // swaps values
	      
	      for(unsigned int k=0;k<N;k++){
		std::swap<T>(copy(j,k), copy(i,k));
		std::swap<T>((*this)(j,k), (*this)(i,k));
	      }
	      
	      // could find a way to solve apparent
	      // singualirity problem.
	      singular = false;
	      break;
	    }
	  }
	  
	  if(singular)
	    return false;
	}
	
	
	// sets a_ii = 1
	{
	  T t = copy(i,i);
	  
	  for(unsigned int j=0;j<N;j++){
	    copy(i,j) /= t;
	    (*this)(i,j) /= t;
	  }
	}
	
	
	if(i >= 1){
	  // eliminates upper row columns to zero	  
	  for(unsigned int j=0;j<i;j++){
	    T k = copy(j,i);
	    
	    for(unsigned int r=0;r<N;r++){
	      copy(j,r) -= k * copy(i,r);
	      (*this)(j,r) -= k * (*this)(i,r);
	    }
	  }
	}
	
	
	if(i < N-1){
	  // eliminates lower row columns to zero
	  for(unsigned int j=i+1;j<N;j++){
	    T k = copy(j,i);
	    
	    for(unsigned int r=0;r<N;r++){
	      copy(j,r) -= k * copy(i,r);
	      (*this)(j,r) -= k * (*this)(i,r);
	    }
	  }
	}
	
      }
      
      
      return true;
    }
    
    
    template <typename T>
    T matrix<T>::trace() const throw(std::logic_error)
    {
      if(numCols != numRows)
	throw std::logic_error("matrix::trace() non square matrix");

      T tr = T(0);
      
      unsigned int j=0;
      for(unsigned int i=0;i<numRows;i++)
      {
	tr += data[j];
	j += numCols + 1;
      }
      
      return tr;
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::identity()
    {
      
      unsigned int index = 0;
      for(unsigned int j=0;j<numRows;j++){
	for(unsigned int i=0;i<numCols;i++, index++)
	{
	  if(i == j) data[index] = T(1);	  
	  else data[index] = T(0);
	}
      }
      
      return (*this);
    }
    
    
    template <typename T>
    matrix<T>& matrix<T>::zero()
    {
      if(typeid(T) == typeid(blas_real<float>) ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>) ||
	 typeid(T) == typeid(blas_complex<double>) ||
	 typeid(T) == typeid(float) ||
	 typeid(T) == typeid(double)){
	
	memset(data, 0, numRows*numCols*sizeof(T));
      }
      else{
	for(unsigned int index=0;index<numRows*numCols;index++)
	  data[index] = T(0.0);
      }
      
      return (*this);
    }
    
    
    template <typename T>
    unsigned int matrix<T>::xsize() const throw()
    {
      if(numRows <= 0) return 0;
      
      return numCols;
    }
    
    
    template <typename T>
    unsigned int matrix<T>::size() const throw()
    {
      return numRows*numCols;
    }
    
    template <typename T>
    unsigned int matrix<T>::ysize() const throw()
    {
      return numRows;
    }
    
    
    template <typename T>
    bool matrix<T>::resize(unsigned int y, unsigned int x) throw()
    {
      if(!resize_x(x)) return false;
      if(!resize_y(y)) return false;      
      
      return true;
    }
    
    
    template <typename T>
    bool matrix<T>::resize_x(unsigned int d) throw()
    {      
      if(d == numCols){
	return true;
      }
      else if(d == 0){
	free(data);
	data = 0;
	
	numRows = 0;
	numCols = 0; 
	data = 0;	
	
	return true;
      }
      else if(d < numCols){
	
	// moves rows backwards/shortens rows
	for(unsigned i=1;i<numRows;i++)
	  memmove(&(data[i*d]), &(data[i*numCols]), d*sizeof(T));
	
	// (tries to) resize data
	T* new_area = 0;
	
	new_area = (T*)realloc(data, sizeof(T)*d*numRows);
	if(new_area) data = new_area;
	else return false;
	
	numCols = d;
	
	return true;
      }
      else if(d > numCols){
	T* new_area = 0;
	
	new_area = (T*)realloc(data, sizeof(T)*d*numRows);
	if(!new_area) return false;
	
	data = new_area;
	
	// moves rows to correct memory locations and zeroes the
	// rest of the data
	for(int i=numRows-1;i>0;i--){
	  memmove(&(data[i*d]), &(data[i*numCols]), numCols*sizeof(T));
	  memset(&(data[i*d + numCols]), 0, (d - numCols)*sizeof(T));
	}
	
	memset(&(data[numCols]), 0, (d - numCols)*sizeof(T));
	
	numCols = d;
	return true;
      }
      
      return true;
    }
    
    
    template <typename T>
    bool matrix<T>::resize_y(unsigned int d) throw()
    {
      T* new_area = 0;
      if(d == numRows){
	return true;
      }
      else if(d == 0){	
	free(data);
	
	numRows = 0;
	numCols = 0; 
	data = 0;
	
	return true;
      }
      
      new_area = (T*)realloc(data, sizeof(T)*numCols*d);
      if(!new_area) return false;
      
      data = new_area;
      
      if(numRows < d)
	memset(&(data[numCols*numRows]), 0, (d - numRows)*numCols*sizeof(T));

      numRows = d;
      
      return true;
    }
    
    
    
    template <typename T>
    T matrix<T>::rownorm(unsigned int y, unsigned int x1, unsigned int x2) const
      throw(std::out_of_range)
    {
      if(x2 < x1 || x1 >= numCols || y >= numRows)
	throw std::out_of_range("rownorm(): bad indeces to matrix");
      
      if(x2 >= numCols) x2 = numCols - 1;
      
      if(typeid(T) == typeid(blas_real<float>)){
	return T( cblas_snrm2(x2 - x1 + 1, (const float*)&(data[y*numCols + x1]), 1) );
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	return T( cblas_scnrm2(x2 - x1 + 1, (const float*)&(data[y*numCols + x1]), 1) );
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	return T( cblas_dnrm2(x2 - x1 + 1, (const double*)&(data[y*numCols + x1]), 1) );
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	return T( cblas_dznrm2(x2 - x1 + 1, (const double*)&(data[y*numCols + x1]), 1) );
      }
      else{ // generic length calculation
	T len = T(0);
	
	for(unsigned int i=x1;i<=x2;i++)
	  len += data[y*numCols + i]*data[y*numCols + i];
	
	return whiteice::math::sqrt(len);
      }
    }
    
    
    template <typename T>
    T matrix<T>::colnorm(unsigned int x, unsigned int y1, unsigned int y2)
      const throw(std::out_of_range)
    {
      if(y2 < y1 || y1 >= numRows || x >= numCols)
	throw std::out_of_range("colnorm(): bad indeces to matrix");
      
      if(y2 > numRows) y2 = numRows - 1;
      
      if(typeid(T) == typeid(blas_real<float>)){
	return T( cblas_snrm2(y2 - y1 + 1, (const float*)&(data[y1*numCols + x]), numCols) );
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	return T( cblas_scnrm2(y2 - y1 + 1, (const float*)&(data[y1*numCols + x]), numCols) );
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	return T( cblas_dnrm2(y2 - y1 + 1, (const double*)&(data[y1*numCols + x]), numCols) );
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	return T( cblas_dznrm2(y2 - y1 + 1, (const double*)&(data[y1*numCols + x]), numCols) );
      }
      else{ // generic length calculation
	T len = T(0);
	
	for(unsigned int i=y1;i<=y2;i++)
	  len += data[x + i*numCols] * data[x + i*numCols];
	
	return whiteice::math::sqrt(len);
      }
    }
    
    
    
    // copies row data to a given vector, M(y,x1:x2) -> v
    template <typename T>
    void matrix<T>::rowcopyto(vertex<T>& v, unsigned int y, unsigned int x1, unsigned int x2)
      const throw(std::out_of_range)
    {
      if(x2 < x1 || x1 >= numCols || y >= numRows)
	throw std::out_of_range("rowcopyto(): bad indeces to matrix");
      
      if(x2 >= numCols) x2 = numCols - 1;
      
      if(v.resize(x2 - x1 + 1) != x2 - x1 + 1)
	throw std::bad_alloc();
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	cblas_scopy(x2 - x1 + 1, (const float*)&(data[y*numCols + x1]), 1, (float*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cblas_ccopy(x2 - x1 + 1, (const float*)&(data[y*numCols + x1]), 1, (float*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dcopy(x2 - x1 + 1, (const double*)&(data[y*numCols + x1]), 1, (double*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cblas_zcopy(x2 - x1 + 1, (const double*)&(data[y*numCols + x1]), 1, (double*)v.data, 1);
      }
      else{ // generic vector copy
	for(unsigned int i=x1;i<=x2;i++)
	  v[i - x1] = data[y*numCols + x1];
      }
    }
    
    
    // copies column data to a given vector, M(y1:y2,x) -> v
    template <typename T>
    void matrix<T>::colcopyto(vertex<T>& v, unsigned int x, unsigned int y1, unsigned int y2)
      const throw(std::out_of_range)
    {
      if(y2 < y1 || y1 >= numRows || x >= numCols)
	throw std::out_of_range("colnorm(): bad indeces to matrix");
      
      if(y2 >= numRows) y2 = numRows - 1;
      
      if(v.resize(y2 - y1 + 1) != y2 - y1 + 1)
	throw std::bad_alloc();
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	cblas_scopy(y2 - y1 + 1, (const float*)&(data[y1*numCols + x]), numCols, (float*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cblas_ccopy(y2 - y1 + 1, (const float*)&(data[y1*numCols + x]), numCols, (float*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dcopy(y2 - y1 + 1, (const double*)&(data[y1*numCols + x]), numCols, (double*)v.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cblas_zcopy(y2 - y1 + 1, (const double*)&(data[y1*numCols + x]), numCols, (double*)v.data, 1);
      }
      else{ // generic copy
	for(unsigned int i=y1;i<=y2;i++)
	  v[i - y1] = data[x + i*numCols];
      }
    }
    
    
    
    template <typename T>
    void matrix<T>::rowcopyfrom(const vertex<T>& v, unsigned int y, unsigned int x1, unsigned int x2)
      throw(std::out_of_range)
    {
      if(x2 >= numCols) x2 = numCols - 1;
      if(x2 < x1 || x1 >= numCols || y >= numRows || v.size() != x2 - x1 + 1)
	throw std::out_of_range("rowcopyfrom(): bad indeces to matrix");
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	cblas_scopy(x2 - x1 + 1, (const float*)v.data, 1, (float*)&(data[y*numCols + x1]), 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cblas_ccopy(x2 - x1 + 1, (const float*)v.data, 1, (float*)&(data[y*numCols + x1]), 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dcopy(x2 - x1 + 1, (const double*)v.data, 1, (double*)&(data[y*numCols + x1]), 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cblas_zcopy(x2 - x1 + 1, (const double*)v.data, 1, (double*)&(data[y*numCols + x1]), 1);
      }
      else{ // generic vector copy
	for(unsigned int i=x1;i<=x2;i++)
	  data[y*numCols + i] = v[i - x1];
      }
    }
    
    
    template <typename T>
    void matrix<T>::colcopyfrom(const vertex<T>& v, unsigned int x, unsigned int y1, unsigned int y2)
      throw(std::out_of_range)
    {
      if(y2 >= numRows) y2 = numRows - 1;
      if(y2 < y1 || y1 >= numRows || x >= numCols || v.size() != y2 - y1 + 1)
	throw std::out_of_range("colnorm(): bad indeces to matrix");      
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	cblas_scopy(y2 - y1 + 1, (const float*)v.data, 1, (float*)&(data[y1*numCols + x]), numCols);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	cblas_ccopy(y2 - y1 + 1, (const float*)v.data, 1, (float*)&(data[y1*numCols + x]), numCols);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dcopy(y2 - y1 + 1, (const double*)v.data, 1, (double*)&(data[y1*numCols + x]), numCols);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	cblas_zcopy(y2 - y1 + 1, (const double*)v.data, 1, (double*)&(data[y1*numCols + x]), numCols);
      }
      else{ // generic length calculation
	for(unsigned int i=y1;i<=y2;i++)
	  data[x + i*numCols] = v[i - y1];
      }
    }
    
    
    // creates submatrix M from matrix ([x0,x0+xs-1],[y0:y0+ys-1])
    template <typename T>
    bool matrix<T>::submatrix(matrix<T>& M,
			      unsigned int x0, unsigned int y0,
			      unsigned int xs, unsigned int ys) const
    {
      if(x0+xs > numCols || y0+ys > numRows)
	return false;
      
      M.resize(ys, xs);
      T* from = this->data + x0 + y0*numCols;
      T* to   = M.data;      
      
      for(unsigned int j=0;j<ys;j++){
	memcpy(to, from, xs*sizeof(T));
	from += numCols;
	to   += xs;
      }
      
      return true;
    }
    
    
    // writes submatrix M to matrix area ([x0+M.xsize()-1],[y0+M.ysize()-1])
    template <typename T>
    bool matrix<T>::write_submatrix(const matrix<T>& M,
				    unsigned int x0, unsigned int y0)
    {
      const unsigned int ys = M.ysize();
      const unsigned int xs = M.xsize();
      
      if(x0+xs > numCols || y0+ys > numRows)
	return false;
      
      T* from = M.data;
      T* to   = this->data + x0 + y0*numCols;
      
      for(unsigned int j=0;j<ys;j++){
	memcpy(to, from, xs*sizeof(T));
	from += xs;
	to   += numCols;
      }
      
      return true;
    }
    
    
    
    // writes and reads matrix data to/from vertex
    template <typename T>
    bool matrix<T>::save_to_vertex(vertex<T>& out,
				   unsigned int x0) const
    {
      out.resize(x0 + numCols*numRows);
      memcpy(&(out.data[x0]), this->data, numCols*numRows*sizeof(T));
      
      return true;
    }
    
    
    template <typename T>
    bool matrix<T>::load_from_vertex(const vertex<T>& in,
				     unsigned int x0)
    {
      if(in.size() < numCols*numRows + x0)
	return false;
      
      memcpy(this->data, &(in.data[x0]), numCols*numRows*sizeof(T));
      
      return true;
    }
    
    
    template <typename T>
    void matrix<T>::normalize() throw()
    {
      // normalizes each row to have unit length
      // normalization of each column: transpose + normalize  + transpose is slow
      // TODO: write code (and test it) which normalizes each column to have unit length
      // (main difference , incX is numCols and length is numRows etc.
      
      if(typeid(T) == typeid(blas_real<float>)){
	float f;
	
	for(unsigned int j=0;j<numRows;j++){
	  f = cblas_snrm2(numCols, (float*)&(data[j*numCols]), 1);
	  
	  if(f == 0.0f) return;
	  else f = 1.0f/f;
	  
	  cblas_sscal(numCols,  f, (float*)&(data[j*numCols]), 1);
	}
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	float f;
	
	for(unsigned int j=0;j<numRows;j++){
	  f = cblas_scnrm2(numCols, (float*)&(data[j*numCols]), 1);
	  
	  if(f == 0.0f) return;
	  else f = 1.0f/f;
	  
	  cblas_csscal(numCols,  f, (float*)&(data[j*numCols]), 1);
	}	
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double f;
	
	for(unsigned int j=0;j<numRows;j++){
	  f = cblas_dnrm2(numCols, (double*)&(data[j*numCols]), 1);
	  
	  if(f == 0.0) return;
	  else f = 1.0/f;
	  
	  cblas_dscal(numCols,  f, (double*)&(data[j*numCols]), 1);
	}
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	double f;
	
	for(unsigned int j=0;j<numRows;j++){
	  f = cblas_dznrm2(numCols, (double*)&(data[j*numCols]), 1);
	  
	  if(f == 0.0) return;
	  else f = 1.0/f;
	  
	  cblas_zdscal(numCols,  f, (double*)&(data[j*numCols]), 1);
	}
      }
      else{ // generic normalization of rows
	
	for(unsigned int j=0;j<numRows;j++){
	  T len = T(0.0);
	  
	  for(unsigned int i=0;i<numCols;i++){
	    len += (data[i+j*numRows])*(data[i+j*numRows]);
	  }
	  
	  len = whiteice::math::sqrt(len);
	  
	  if(len != T(0.0)){
	    for(unsigned int i=0;i<numCols;i++){
	      data[i+j*numRows] /= len;
	    }
	  }
	}
      }
      
    }
    
    
    template <typename T>
    bool matrix<T>::comparable() throw()
    {
      return false;
    }
    
    
    
    ////////////////////////////////////////////////////////////
    // matrix data compression
    // note: compressor destroys possible memory
    // aligmentations
    
    
    template <typename T>
    bool matrix<T>::compress() throw()
    {
      if(compressor != 0) return false; // already compressed
      
      compressor = new MemoryCompressor();
      
      compressor->setMemory(data, sizeof(T)*numRows*numCols);
      // let compressor allocate the memory
      
      if(compressor->compress()){ // compression ok.
	free(data); data = 0; // free's memory
	compressor->setMemory(data, 0);
	return true;
      }
      else{
	if(compressor->getTarget() != 0)
	  free(compressor->getTarget());
	
	delete compressor;
	compressor = 0;
	return false;
      }
    }
    
    
    template <typename T>
    bool matrix<T>::decompress() throw()
    {
      if(compressor == 0) return false; // not compressed
      
      if(compressor->decompress()){ // decompression ok.
	data = (T*)( compressor->getMemory() );
	
	free(compressor->getTarget());
	
	delete compressor;
	compressor = 0;
	
	return true;
      }
      else{
	return false;
      }
    }
    
    
    template <typename T>
    bool matrix<T>::iscompressed() const throw()
    {
      return (compressor != 0);
    }
    
    
    template <typename T>
    float matrix<T>::ratio() const throw()
    {
      if(compressor == 0) return 1.0f;
      return ( ((float)compressor->getTargetSize()) / ((float)(numRows*numCols*sizeof(T))) );
    }
    
    
    ////////////////////////////////////////////////////////////
    
    /***************************************************/
    
    
    template <typename T>
    std::ostream& operator<<(std::ostream& ios, const matrix<T>& M)
    {
      ios << "[";
    
      for(unsigned int j=0;j<M.ysize();j++){
	for(unsigned int i=0;i<M.xsize();i++)
	{
	  ios << " " << M(j,i);
	}
	
	ios << "; ";
      }
    
      ios << "]";
    
      return ios;
    }
    
    
    
    // explicit template instantations
    
    template class matrix<float>;    
    template class matrix<double>;
    template class matrix<complex<float> >;
    template class matrix<complex<double> >;
    
    template class matrix<int>;
    template class matrix<char>;
    template class matrix<unsigned int>;
    template class matrix<unsigned char>;
        
    template class matrix< blas_real<float> >;
    template class matrix< blas_real<double> >;
    template class matrix< blas_complex<float> >;
    template class matrix< blas_complex<double> >;
    
    
    template matrix<float> operator*<float>(const float&, const matrix<float>&) throw(std::invalid_argument);
    template matrix<double> operator*<double>(const double&, const matrix<double>&) throw(std::invalid_argument);
    template matrix<complex<float> > operator*<complex<float> >(const complex<float>&, const matrix<complex<float> >&)
      throw(std::invalid_argument);    
    template matrix<complex<double> > operator*<complex<double> >(const complex<double>&, const matrix<complex<double> >&)
      throw(std::invalid_argument);
    
    template matrix<int> operator*<int>(const int&, const matrix<int>&) throw(std::invalid_argument);
    template matrix<char> operator*<char>(const char&, const matrix<char>&) throw(std::invalid_argument);
    template matrix<unsigned int> operator*<unsigned int>(const unsigned int&, const matrix<unsigned int>&)
      throw(std::invalid_argument);
    template matrix<unsigned char> operator*<unsigned char>(const unsigned char&, const matrix<unsigned char>&)
      throw(std::invalid_argument);
    
    
    template matrix<blas_real<float> > operator*<blas_real<float> >
      (const blas_real<float>&, const matrix<blas_real<float> >&) throw(std::invalid_argument);
       
    template matrix<blas_real<double> > operator*<blas_real<double> >
      (const blas_real<double>&, const matrix<blas_real<double> >&) throw(std::invalid_argument);
    
    
    template matrix<blas_complex<float> > operator*<blas_complex<float> >
      (const blas_complex<float>&, const matrix<blas_complex<float> >&) throw(std::invalid_argument);
    template matrix<blas_complex<double> > operator*<blas_complex<double> >
      (const blas_complex<double>&, const matrix<blas_complex<double> >&) throw(std::invalid_argument);
        
    template std::ostream& operator<< <float>(std::ostream& ios, const matrix<float>& M);
    template std::ostream& operator<< <double>(std::ostream& ios, const matrix<double>& M);
    template std::ostream& operator<< <complex<float> >(std::ostream& ios, const matrix<complex<float> >& M);
    template std::ostream& operator<< <complex<double> >(std::ostream& ios, const matrix<complex<double> >& M);
    template std::ostream& operator<< <int>(std::ostream& ios, const matrix<int>& M);
    template std::ostream& operator<< <char>(std::ostream& ios, const matrix<char>& M);
    template std::ostream& operator<< <unsigned int>(std::ostream& ios, const matrix<unsigned int>& M);
    template std::ostream& operator<< <unsigned char>(std::ostream& ios, const matrix<unsigned char>& M);
    template std::ostream& operator<< <blas_real<float> >(std::ostream& ios, const matrix<blas_real<float> >& M);
    template std::ostream& operator<< <blas_real<double> >(std::ostream& ios, const matrix<blas_real<double> >& M);
    template std::ostream& operator<< <blas_complex<float> >(std::ostream& ios, const matrix<blas_complex<float> >& M);
    template std::ostream& operator<< <blas_complex<double> >(std::ostream& ios, const matrix<blas_complex<double> >& M);
    
    
  };
};
  
  
#endif
  
