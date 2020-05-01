

#ifndef vertex_cpp
#define vertex_cpp

#include "vertex.h"
#include "dinrhiw_blas.h"
#include "ownexception.h"
#include "gcd.h"
#include "number.h"

#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <exception>
#include <typeinfo>
#include <vector>
#include <new>

#include <stdlib.h>
#include <string.h>

// #define OPENBLAS 0

namespace whiteice
{
  namespace math
  {
    
    template <typename T>
    vertex<T>::vertex()
    {
      this->compressor = nullptr;
      this->dataSize = 0;      
      this->data = nullptr;
      
      this->data = (T*)malloc(sizeof(T));
      if(this->data == nullptr) throw std::bad_alloc();
      
      memset(this->data, 0, sizeof(T));
      this->dataSize = 1;      
    }
    
    
    // vertex ctor, i is dimension of vector
    template <typename T>
    vertex<T>::vertex(unsigned int i)
    {
      this->compressor = nullptr;
      this->dataSize = 0;
      this->data = nullptr;
      
      if(i > 0){
#ifdef BLAS_MEMALIGN
	// electric fence don't know about posix_memalign()
	posix_memalign((void**)&(this->data),
		       (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		       i*sizeof(T));
#else
	this->data = (T*)malloc(i*sizeof(T));
#endif
	
	if(this->data == 0)
	  throw std::bad_alloc();
	
	
	memset(this->data, 0, i*sizeof(T));
      }
      
      
      this->dataSize = i;
    }
    
    
    // vertex ctor - makes copy of v
    template <typename T>
    vertex<T>::vertex(const vertex<T>& v)
    {
      this->compressor = 0;
      this->dataSize = 0;
      this->data = 0;
      
      if(v.compressor != 0)
	throw illegal_operation("vertex ctor: to be copied vertex is compressed");
      
      if(v.data){
#ifdef BLAS_MEMALIGN
	// electric fence don't know about posix_memalign()
	posix_memalign((void**)&(this->data),
		       (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		       v.dataSize*sizeof(T));
#else
	this->data = (T*)malloc(v.dataSize*sizeof(T));
#endif
	
	if(this->data == 0)
	  throw std::bad_alloc();
	
	
	if(typeid(T) == typeid(blas_real<float>)){
	  cblas_scopy(v.dataSize,
		      (const float*)v.data, 1,
		      (float*)(this->data), 1);
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
	  cblas_ccopy(v.dataSize,
		      (const float*)v.data, 1,
		      (float*)(this->data), 1);
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  cblas_dcopy(v.dataSize,
		      (const double*)v.data, 1,
		      (double*)(this->data), 1);
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
	  cblas_zcopy(v.dataSize,
		      (const double*)v.data, 1,
		      (double*)(this->data), 1);
	}
	else{ // generic memcpy
	  memcpy(this->data, v.data, v.dataSize*sizeof(T));
	}	  	
      }
      
      this->dataSize = v.dataSize;
    }
    
    
    // makes direct copy of temporal value
#if 0    
    template <typename T>
    vertex<T>::vertex(vertex<T>&& t)
    {
      this->data = t.data;
      this->dataSize = t.dataSize;
      this->compressor = t.compressor;
      
      t.data = nullptr;
      t.compressor = nullptr;
    }
#endif
    
    
    // vertex ctor - makes copy of v
    template <typename T>
    vertex<T>::vertex(const std::vector<T>& v)
    {
      this->compressor = 0;
      this->dataSize = 0;
      this->data = 0;
      
      if(v.size() > 0){
#ifdef BLAS_MEMALIGN
	posix_memalign((void**)&(this->data),
		       (8/whiteice::gcd<unsigned int>(8,sizeof(void*)))*sizeof(void*),
		       v.size()*sizeof(T));
#else
	this->data = (T*)malloc(v.size()*sizeof(T));
#endif
	
	if(this->data == 0)
	  throw std::bad_alloc();
	
	
	for(unsigned int i=0;i<v.size();i++)
	  (this->data)[i] = v[i];
      }
      
      
      this->dataSize = v.size();
    }
    
    
    // vertex dtor
    template <typename T>
    vertex<T>::~vertex()
    {
      if(this->compressor) delete (this->compressor);
      if(this->data) free(this->data);      
    }
    
    /***************************************************/
    
    // returns vertex dimension/size
    template <typename T>
    unsigned int vertex<T>::size() const { return dataSize; }
    
    
    // sets vertex dimension/size, fills new dimensios with zero
    template <typename T>
    unsigned int vertex<T>::resize(unsigned int d) 
    {
      if(d == 0){
	free(data);
	data = 0;
	dataSize = 0;
      }
      else{
	T* new_area = 0;
	
	if(data != 0){
	  new_area = (T*)realloc(data, sizeof(T)*d);
	}
	else{
	  new_area = (T*)malloc(sizeof(T)*d);
	}
	  
	if(new_area == 0)
	  return dataSize; // mem. alloc failure
	
	data = new_area;
    
	// fills new memory area with zeros
	if(dataSize < d)
	  for(unsigned int s = dataSize;s<d;s++)
	    data[s] = T(0.0);
	
	dataSize = d;
      }
      
      return dataSize;
    }

    
    // returns length of vertex
    template <typename T>
    T vertex<T>::norm() const 
    {
      T len; // cblas_Xnrm2 optimizated functions
      
      if(typeid(T) == typeid(blas_real<float>)){
	len = (T)cblas_snrm2(dataSize, (float*)data, 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	len = (T)cblas_scnrm2(dataSize, (float*)data, 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	len = (T)cblas_dnrm2(dataSize, (double*)data, 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	len = (T)cblas_dznrm2(dataSize, (double*)data, 1);
	
	return len;
      }
      else{ // generic length calculation
	len = T(0);
	
	for(unsigned int i=0;i<dataSize;i++)
	  len += data[i]*data[i];
	
	len = (T)whiteice::math::sqrt(len);
	return len;
      }
    }
    
    
    // calculates partial norm for vertex(i:j)
    template <typename T>
    T vertex<T>::norm(unsigned int i, unsigned int j) const 
    {
      T len = T(0.0f); // cblas_Xnrm2 optimizated functions
      
      if(i >= j || i > dataSize || j > dataSize)
	return len;
      
      
      if(typeid(T) == typeid(blas_real<float>)){
	len = (T)cblas_snrm2(j - i,(float*)(&(data[i])), 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	len = (T)cblas_scnrm2(j - i,(float*)(&(data[i])), 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	len = (T)cblas_dnrm2(j - i,(double*)(&(data[i])), 1);
	
	return len;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	len = (T)cblas_dznrm2(j - i,(double*)(&(data[i])), 1);
	
	return len;
      }
      else{ // generic length calculation
	
	for(unsigned int i=0;i<dataSize;i++)
	  len += (T)(data[i]*data[i]);
	
	len = (T)sqrt(len);
	return len;
      }
    }
    
    
    // sets length to zero, zero length -> retuns false
    template <typename T>
    bool vertex<T>::normalize() 
    {
      // uses optimized cblas_Xscal() routines
      
      T len = norm();
      if(len == T(0.0f)) return false;
      len = T(1.0f) / len;
    
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(dataSize, *((float*)&len), (float*)data, 1);
	return true;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cscal(dataSize, (const float*)&len, (float*)data, 1);
	return true;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(dataSize, *((double*)&len), (double*)data, 1);
	return true;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zscal(dataSize, (const double*)&len, (double*)data, 1);
	return true;
      }    
      else{
	
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] *= len;
	
	return true;
      }
    }
    
    
    // vertex = 0;
    template <typename T>
    void vertex<T>::zero() 
    {
      if(dataSize <= 0) return;
      
      if(typeid(T) == typeid(blas_real<float>) ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>) ||
	 typeid(T) == typeid(blas_complex<double>) ||
	 typeid(T) == typeid(float) ||
	 typeid(T) == typeid(double))
      {
	// all bits = 0, is zero number representation
	memset(data, 0, dataSize*sizeof(T));
      }
      else{
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] = T(0.0f);
      }
    }


    template <typename T>
    void vertex<T>::hermite() 
    {
      if(dataSize <= 0) return;

      for(unsigned int i=0;i<dataSize;i++)
	data[i] = conj(data[i]);
    }
    
    
    // calculates sum of vertexes
    template <typename T>
    vertex<T> vertex<T>::operator+(const vertex<T>& v) const
      
    {
      if(v.dataSize != dataSize){
	printf("ERROR: illegal operation: vector operator+ failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);
	
	throw illegal_operation("vector op: vector dim. mismatch");
      }
      
      // copy of this vector
      vertex<T> r(*this);
      
      // no BLAS speedups (alpha = 1, alpha*x + y , faster than manual?)
      
      for(unsigned int i=0;i<v.dataSize;i++){
	r.data[i] += v.data[i];
      }
      
      return r;
    }

    
    
    // substracts two vertexes
    template <typename T>
    vertex<T> vertex<T>::operator-(const vertex<T>& v) const
      
    {
      if(v.dataSize != dataSize){
	printf("ERROR: illegal operation: vector operator- failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);

	throw illegal_operation("vector op: vector dim. mismatch");
      }
      
      // copy of this vector
      vertex<T> r(*this);
      
      // cblas_Xaxpy() (alpha = -1) -> r = x - y
      
      if(typeid(T) == typeid(blas_real<float>)){
	float alpha = -1;
	
	cblas_saxpy(dataSize, alpha, (float*)v.data, 1,
		    (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	T alpha; alpha = -1;
	
	cblas_caxpy(dataSize, (const float*)&alpha, (float*)v.data, 1,
		    (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	double alpha = -1;
	
	cblas_daxpy(dataSize, alpha, (double*)v.data, 1,
		    (double*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	T alpha; alpha = -1;
	
	cblas_zaxpy(dataSize, (const double*)&alpha, (double*)v.data, 1,
		    (double*)r.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<v.dataSize;i++){
	  r.data[i] -= v.data[i];
	}
      }
      
      return r;
    }
    
    
    // calculates innerproduct - returns 1-dimension vertex
    // if input either of the vertexes is dim=1 vertex this
    // is scalar product
    template <typename T>
    vertex<T> vertex<T>::operator*(const vertex<T>& v) const
      
    {
      if(dataSize != v.dataSize && (dataSize != 1 && v.dataSize != 1)){
	printf("ERROR: illegal operation: vector operator* failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);
	throw illegal_operation("vector op: vector dim. mismatch");
      }
      
      // uses BLAS
      
      if(dataSize != 1 && v.dataSize != 1){
	
	vertex<T> r(1);
	r = T(0);
	
	if(typeid(T) == typeid(blas_real<float>)){
	  *((T*)&(r.data[0])) = T(cblas_sdot(dataSize, (float*)data, 1,
					     (float*)v.data, 1));
	  return r;
	}
	else if(typeid(T) == typeid(blas_complex<float>)){
#ifdef OPENBLAS
	  cblas_cdotc_sub(dataSize, (float*)data, sizeof(T),
			  //  (float*)v.data, 1, (openblas_complex_float*)&(r.data[0]));
			  (float*)v.data, 1, (openblas_complex_float*)&(r.data[0]));
#else
	  cblas_cdotc_sub(dataSize, (float*)data, sizeof(T),
			  //  (float*)v.data, 1, (openblas_complex_float*)&(r.data[0]));
			  (float*)v.data, 1, (float*)&(r.data[0]));	  
#endif
	  return r;
	}
	else if(typeid(T) == typeid(blas_real<double>)){
	  *((T*)&(r.data[0])) = T(cblas_ddot(dataSize, (double*)data, 1,
					     (double*)v.data, 1));
	  return r;
	}
	else if(typeid(T) == typeid(blas_complex<double>)){
#ifdef OPENBLAS
	  cblas_zdotc_sub(dataSize, (double*)data, 1,
			  // (double*)v.data, 1, (openblas_complex_double*)&(r.data[0]));
			  (double*)v.data, 1, (openblas_complex_double*)&(r.data[0]));
#else
	  cblas_zdotc_sub(dataSize, (double*)data, 1,
			  // (double*)v.data, 1, (openblas_complex_double*)&(r.data[0]));
			  (double*)v.data, 1, (double*)&(r.data[0]));
#endif
	  return r;
	}
	else{ // "normal implementation"
	  for(unsigned int i=0;i<v.dataSize;i++)
	    *((T*)&(r.data[0])) += data[i]*v.data[i];
	  
	  return r;
	}
	
      }
      else{ // scalar product
	
	if(dataSize == 1){
	  vertex<T> r = v * data[0];
	  return r;
	}
	else{ // v.dataSize == 1
	  vertex<T> r = (*this) * v.data[0];
	  return r;
	}
      }
      
    }
    
    // no divide operation
    template <typename T>
    vertex<T> vertex<T>::operator/(const vertex<T>& v) const {
      throw illegal_operation("vertex(): '/'-operator not available");
    }
    
    // no "!" operation
    template <typename T>
    vertex<T> vertex<T>::operator!() const {
      throw illegal_operation("vertex(): '!'-operation not available");
    }
    
    // changes vertex sign
    template <typename T>
    vertex<T> vertex<T>::operator-() const
      
    {
      vertex<T> r(*this);
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(dataSize, -1.0f, (float*)r.data, 1);
	return r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_csscal(dataSize, -1.0f, (float*)r.data, 1);
	return r;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(dataSize, -1.0, (double*)r.data, 1);
	return r;      
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zdscal(dataSize, -1.0, (double*)r.data, 1);
	return r;
      }    
      else{
	for(unsigned int i=0;i<dataSize;i++)
	  r.data[i] = -data[i];
	
	return r;
      }
    }
    
    // calculates cross product
    template <typename T>
    vertex<T> vertex<T>::operator^(const vertex<T>& v) const
      
    {
      if(v.dataSize != 3 || this->dataSize != 3)      
	throw illegal_operation("crossproduct: vector dimension != 3");
      
      // *NO* CBLAS
      
      vertex<T> r(3);
      
      r[0] = data[1]*v.data[2] - data[2]*v.data[1];
      r[1] = data[2]*v.data[0] - data[0]*v.data[2];
      r[2] = data[0]*v.data[1] - data[1]*v.data[0];
      
      return r;    
    }
    
    /***************************************************/
    
    // adds vertexes
    template <typename T>
    vertex<T>& vertex<T>::operator+=(const vertex<T>& v)
      
    {
      if(v.dataSize != dataSize){
	printf("ERROR: illegal operation: vector operator+= failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);

	throw illegal_operation("vector op: vector dim. mismatch");
      }
      
      if(typeid(T) == typeid(blas_real<float>)){
	float alpha = +1.0;
	
	cblas_saxpy(dataSize, alpha, (float*)v.data, 1, (float*)(this->data), 1);
	
      }
      else{
	// *NO* CBLAS
	
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] += v.data[i];
      }
      
      
      return *this;
    }
    
    // subtracts vertexes
    template <typename T>
    vertex<T>& vertex<T>::operator-=(const vertex<T>& v)
      
    {
      if(dataSize != dataSize){
	printf("ERROR: illegal operation: vector operator-= failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);
	
	throw illegal_operation("vector op: vector dim. mismatch");
      }
      
      if(typeid(T) == typeid(blas_real<float>)){
	float alpha = -1.0;
	
	cblas_saxpy(dataSize, alpha, (float*)v.data, 1, (float*)(this->data), 1);
	
      }
      else{
	// *NO* CBLAS
	
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] -= v.data[i];
      }
      
      return *this;
    }
    
    // calculates inner product
    template <typename T>
    vertex<T>& vertex<T>::operator*=(const vertex<T>& v)
      
    {
      if(v.dataSize != dataSize){
	printf("ERROR: illegal operation: vector operator*= failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);
	throw illegal_operation("vector op: vector dim. mismatch");
      }
      
      // *NO* CBLAS
      
      vertex<T> r(1);
      r[0] = T(0);
      
      for(unsigned int i=0;i<v.dataSize;i++)
	r[0] += data[i]*v.data[i];
      
      *this = r;
      
      return *this;
    }
    
    // dividing not available
    template <typename T>
    vertex<T>& vertex<T>::operator/=(const vertex<T>& v)
      {
      throw illegal_operation("vertex(): '/='-operator not available");
    }
    
    // assigns given vertex value to this vertex
    template <typename T>
    vertex<T>& vertex<T>::operator=(const vertex<T>& v)
      
    {
      if(v.compressor != 0 || this->compressor != 0)
	throw illegal_operation("vertex '='-operator: compressed vertex data");
      
      if(this != &v){ // no self-assignment
	if(v.dataSize != this->dataSize)
	  if(this->resize(v.dataSize) != v.dataSize)
	    throw illegal_operation("vertex '='-operator: out of memory");
	
	memcpy(this->data, v.data, sizeof(T)*v.dataSize);
      }
      
      
      return *this;
    }

#if 0    
    template <typename T>
    vertex<T>& vertex<T>::operator=(vertex<T>&& t) 
    {
      if(this == &t) return *this; // self-assignment
      
      // printf("vertex&& operator=\n"); fflush(stdout);
      
      if(this->data) free(data);
      if(this->compressor) delete compressor;
      
      this->data = std::move(t.data);
      this->dataSize = std::move(t.dataSize);
      this->compressor = std::move(t.compressor);
      
      t.data = nullptr;
      t.compressor = nullptr;
      
      return *this;
    }
#endif
    
    /***************************************************/

    // compares two vertexes for equality
    template <typename T>
    bool vertex<T>::operator==(const vertex<T>& v) const
      
    {
      if(v.dataSize != dataSize)
	return false; // throw uncomparable("vertex compare: dimension mismatch");
      
      if(typeid(T) == typeid(blas_real<float>)    ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>)   ||
	 typeid(T) == typeid(blas_complex<double>)){
	
	return (memcmp(v.data, data, dataSize*sizeof(T)) == 0);
      }
      else{
	
	for(unsigned int i=0;i<v.dataSize;i++)
	  if(data[i] != v.data[i]) return false;
      }
      
      
      return true;
    }
    
    // compares two vertexes for non-equality
    template <typename T>
    bool vertex<T>::operator!=(const vertex<T>& v) const
      
    {
      if(v.dataSize != dataSize)
	return true; // throw uncomparable("vertex compare: dimension mismatch");
      
      
      if(typeid(T) == typeid(blas_real<float>)    ||
	 typeid(T) == typeid(blas_complex<float>) ||
	 typeid(T) == typeid(blas_real<double>)   ||
	 typeid(T) == typeid(blas_complex<double>)){
	
	return (memcmp(v.data, data, dataSize*sizeof(T)) != 0);
      }
      else{
	for(unsigned int i=0;i<v.dataSize;i++)
	  if(data[i] != v.data[i]) return true;
      }
      
      return false;
    }
    
    
    // not defined
    template <typename T>
    bool vertex<T>::operator>=(const vertex<T>& v) const {
      if(dataSize != 1) throw uncomparable("vertex(): '>='-operator not defined");
      else return (data[0] >= v.data[0]);
    }
    
    // not defined
    template <typename T>
    bool vertex<T>::operator<=(const vertex<T>& v) const {
      if(dataSize != 1) throw uncomparable("vertex(): '<='-operator not defined");
      else return (data[0] <= v.data[0]);
    }
    
    // not defined
    template <typename T>
    bool vertex<T>::operator< (const vertex<T>& v) const {
      if(dataSize != 1) throw uncomparable("vertex(): '<'-operator not defined");
      else return (data[0] < v.data[0]);
    }

    // not defined
    template <typename T>
    bool vertex<T>::operator> (const vertex<T>& v) const {
      if(dataSize != 1) throw uncomparable("vertex(): '>'-operator not defined");
      else return (data[0] > v.data[0]);
    }
    

    // assigns quaternion to 4 dimension vertex
    template <typename T>
    vertex<T>& vertex<T>::operator=(const quaternion<T>& q)
      
    {
      if(dataSize != 4){
	printf("ERROR: illegal operation: vector operator= failed: vector dim is not 4 (%s:%d)\n",
	       __FILE__, __LINE__);

	throw std::domain_error("vertex '='-operator: cannot assign quaternion - dimension mismatch");
      }
      
      for(unsigned int i=0;i<4;i++)
	data[i] = q[i];
      
      return (*this);
    }

    
    // returns vertex with absolute value of each vertex element
    template <typename T>
    vertex<T>& vertex<T>::abs() 
    {
      for(unsigned int i=0;i<dataSize;i++)
	data[i] = whiteice::math::abs(data[i]);
      
      return (*this);
    }
    
    
    /***************************************************/
    // scalars
    
    
    /* sets all elements of vertex = given scalar */
    template <typename T>
    vertex<T>& vertex<T>::operator=(const T& s)
      
    {
      for(unsigned int i=0;i<dataSize;i++)
	data[i] = s;
      
      return *this;
    }
    
    
    
    // multiples vertex with scalar */
    template <typename T>
    vertex<T>  vertex<T>::operator*(const T& s) const 
    {
      vertex<T> r(dataSize);
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(dataSize, *((float*)&s), (float*)data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_caxpy(dataSize, (const float*)&s, (float*)data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(dataSize, *((double*)&s), (double*)data, 1, (double*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zaxpy(dataSize, (const double*)&s, (double*)data, 1, (double*)r.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<dataSize;i++)
	  r.data[i] = data[i]*s;
      }
      
      
      return r;
    }        
    
    
    // multiples vertex with scalar */
    template <typename T>
    vertex<T>  vertex<T>::operator/(const T& s) const 
    {
      vertex<T> r(dataSize);      
      T ss = T(1)/s;      
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(dataSize, *((float*)&ss), (float*)data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_caxpy(dataSize, (const float*)&ss, (float*)data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(dataSize, *((double*)&ss), (double*)data, 1, (double*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zaxpy(dataSize, (const double*)&ss, (double*)data, 1, (double*)r.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<dataSize;i++)
	  r.data[i] = data[i]*ss;
      }      
      
      return r;
    }
    
    
    
    // multiples vertex with scalar */
    template <typename T>
    vertex<T>& vertex<T>::operator*=(const T& s) 
    {
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(dataSize, *((float*)&s), (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cscal(dataSize, (const float*)&s, (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(dataSize, *((double*)&s), (double*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zscal(dataSize, (const double*)&s, (double*)data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] *= s;
      }
      
      return *this;
    }
    
    
    // multiples vertex with scalar */
    template <typename T>
    vertex<T>& vertex<T>::operator/=(const T& s) 
    {
      T ss = T(1.0)/s;
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_sscal(dataSize, *((float*)&ss), (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	
	cblas_cscal(dataSize, (const float*)&ss, (float*)data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_dscal(dataSize, *((double*)&ss), (double*)data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	
	cblas_zscal(dataSize, (const double*)&ss, (double*)data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<dataSize;i++)
	  data[i] *= ss;
      }
      
      return *this;
    }
    
    
    
    // scalar times vertex
    template <typename T>
    vertex<T> operator*(const T& s, const vertex<T>& v)
    {
      vertex<T> r(v.dataSize);
      
      if(typeid(T) == typeid(blas_real<float>)){
	
	cblas_saxpy(v.dataSize, *((float*)&s), (float*)v.data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> ss(*((blas_complex<float>*)&s));
	
	cblas_caxpy(v.dataSize, (const float*)&ss, (float*)v.data, 1, (float*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	
	cblas_daxpy(v.dataSize, *((double*)&s), (double*)v.data, 1, (double*)r.data, 1);
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> ss(*((blas_complex<double>*)&s));
	
	cblas_zaxpy(v.dataSize, (const double*)&ss, (double*)v.data, 1, (double*)r.data, 1);
      }
      else{ // "normal implementation"
	for(unsigned int i=0;i<v.dataSize;i++)
	  r.data[i] = v.data[i]*s;
      }
      
      return r;
    }
    
    
    // multiplies matrix from left
    template <typename T>
    vertex<T> vertex<T>::operator* (const matrix<T>& M) const
      
    {
      if(dataSize != M.numRows){
	printf("ERROR: illegal operation: vector/matrix operator* failed: dim %d != dim %dx%d (%s:%d)\n",
	       dataSize, M.numRows, M.numCols, __FILE__, __LINE__);
	
	throw std::invalid_argument("multiply: vertex/matrix dim. mismatch");
      }
      
      vertex<T> r(M.numCols);
      
      // BLAS level 2
      // uses optimized cblas_Xgemv() functions
      
      if(typeid(T) == typeid(blas_real<float>)){

	cblas_sgemv(CblasRowMajor, CblasTrans,
		    M.numRows, M.numCols,
		    1.0f, (float*)M.data, M.numCols, // 1,
		    (float*)data, 1,
		    0.0f, (float*)r.data, 1);
	
	return r;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	blas_complex<float> a, b;
	a = 1.0f; b = 0.0f;
	
	cblas_cgemv(CblasRowMajor, CblasTrans,
		    M.numRows, M.numCols,
		    (float*)(&a), (float*)M.data, M.numCols,
		    (float*)data, 1,
		    (float*)(&b), (float*)r.data, 1);
	
	return r;

      }
      else if(typeid(T) == typeid(blas_real<double>)){
	cblas_dgemv(CblasRowMajor, CblasTrans,
		    M.numRows, M.numCols,
		    1.0, (double*)M.data, M.numCols,
		    (double*)data, 1,
		    0.0, (double*)r.data, 1);
	
	return r;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	blas_complex<double> a, b;
	a = 1.0f; b = 0.0f;	
	
	cblas_zgemv(CblasRowMajor, CblasTrans,
		    M.numRows, M.numCols,
		    (double*)(&a), (double*)M.data, M.numCols,
		    (double*)data, 1,
		    (double*)(&b), (double*)r.data, 1);
	
	return r;
      }
      else{ // vertex * matrix calculation
	
	for(unsigned int j=0;j<M.numCols;j++){
	  for(unsigned int i=0;i<M.numRows;i++)
	    r.data[j] += data[i]*M(i,j);
	}
	
	return r;      
      }
      
      
    }
    
    
    /***************************************************/
    
    
    template <typename T>
    matrix<T> vertex<T>::outerproduct() const 
    {
      const unsigned int N = dataSize;
      T s = T(1.0f);
      
      if(typeid(T) == typeid(blas_real<float>)){
	matrix<T> M(N, N);
	
	cblas_sspr(CblasRowMajor, CblasUpper, N, 
		   *((float*)&s), (float*)data, 1, (float*)M.data);
	
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(M.data[r*N + r]), &(M.data[r*N - ((r - 1)*r)/2]), sizeof(T)*(i+1));
	}
	
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_scopy(N - i - 1, 
		      (float*)&(M.data[i*N + 1 + i]), 1,
		      (float*)&(M.data[(i+1)*N + i]), N);
	return M;
      }
      else if(typeid(T) == typeid(blas_complex<float>)){
	matrix<T> M(N, N);
	
	cblas_chpr(CblasRowMajor, CblasUpper, N,
		   *((float*)&s), (float*)data, 1, (float*)M.data);
	
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(M.data[r*N + r]), &(M.data[r*N - ((r - 1)*r)/2]), sizeof(T) * (i+1));
	}
	
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_ccopy(N - i - 1, 
		      (float*)&(M.data[i*N + 1 + i]), 1,
		      (float*)&(M.data[(i+1)*N + i]), N);
	return M;
      }
      else if(typeid(T) == typeid(blas_real<double>)){
	matrix<T> M(N, N);
	
	cblas_dspr(CblasRowMajor, CblasUpper, N, 
		   *((double*)&s), (double*)data, 1, (double*)(M.data));
	
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(M.data[r*N + r]), &(M.data[r*N - ((r - 1)*r)/2]), sizeof(T) * (i+1));	
	}
	
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_dcopy(N - i - 1,
		      (double*)&(M.data[i*N + 1 + i]), 1,
		      (double*)&(M.data[(i+1)*N + i]), N);
	return M;
      }
      else if(typeid(T) == typeid(blas_complex<double>)){
	matrix<T> M(N, N);
	
	cblas_zhpr(CblasRowMajor, CblasUpper, N,
		   *((double*)&s), (double*)data, 1, (double*)M.data);
	
	for(unsigned int i=0;i<(N-1);i++){ // (N-i):th row
	  unsigned int r = N - i - 1;
	  memmove(&(M.data[r*N + r]), &(M.data[r*N - ((r - 1)*r)/2]), sizeof(T) * (i+1));
	}
	
	for(unsigned int i=0;i<(N-1);i++)
	  cblas_zcopy(N - i - 1, 
		      (double*)&(M.data[i*N + 1 + i]), 1,
		      (double*)&(M.data[(i+1)*N + i]), N);
	return M;	  
      }
      else
	return outerproduct(*this, *this);
    }


    template <typename T>
    matrix<T> vertex<T>::outerproduct(const vertex<T>& v) const
      
    {
      return outerproduct(*this, v);
    }
    
    
    /* outer product of N length vertexes */
    template <typename T>
    matrix<T> vertex<T>::outerproduct(const vertex<T>& v0,
				      const vertex<T>& v1) const
      
    {
      matrix<T> m(v0.dataSize, v1.dataSize);
      
      for(unsigned int i=0;i<v0.dataSize;i++)
	for(unsigned int j=0;j<v1.dataSize;j++)
	  m(i,j) = v0.data[i]*v1.data[j];
      
      return m;
    }
    
    
    // element-wise multiplication of vector elements
    template <typename T>
    vertex<T>& vertex<T>::dotmulti(const vertex<T>& v) 
    {
      if(this->dataSize != v.dataSize){
	printf("ERROR: illegal operation: vector dotmulti() failed: dim %d != dim %d (%s:%d)\n",
	       dataSize, v.dataSize, __FILE__, __LINE__);
	
	throw illegal_operation("vector op: vector dim. mismatch");
      }
      
      // should be optimized
      
      for(unsigned int i=0;i<dataSize;i++)
	data[i] *= v.data[i];
      
      return (*this);
    }
    
    
    template <typename T>
    bool vertex<T>::subvertex(vertex<T>& v,
			      unsigned int x0,
			      unsigned int len) const 
    {
      if(x0+len > dataSize)
	return false;
      
      v.resize(len);
      
      memcpy(v.data, data + x0, len*sizeof(T));
      return true;
    }
    
    
    template <typename T>
    bool vertex<T>::write_subvertex(const vertex<T>& v, unsigned int x0) 
    {
      const unsigned int len = v.size();
      
      if(x0+len > dataSize)
	return false;
      
      memcpy(data + x0, v.data, len*sizeof(T));
      return true;
    }
    
    
#if 0    
    template <typename T> // iterators
    typename vertex<T>::iterator vertex<T>::begin() {
      return c.begin();
    }

    template <typename T>
    typename vertex<T>::iterator vertex<T>::end() {
      return c.end();
    }
    
    template <typename T> // iterators
    typename vertex<T>::const_iterator vertex<T>::begin() const {
      return c.begin();
    }
    
    template <typename T>
    typename vertex<T>::const_iterator vertex<T>::end() const {
      return c.end();
    }
#endif    
    
    template <typename T>
    bool vertex<T>::comparable() 
    {
      return false;
    }
    
    
    template <typename T>
    bool vertex<T>::saveAscii(const std::string& filename) const 
	{
    	FILE* fp = fopen(filename.c_str(), "wt");
    	if(fp == NULL || ferror(fp)) return false;

    	if(this->dataSize > 0){
    		double f = 0.0;
    		whiteice::math::convert(f, this->data[0]);
    		fprintf(fp, "%f", f);
    	}


    	for(unsigned int i=1;i<this->dataSize;i++){
    		double f = 0.0;
    		whiteice::math::convert(f, this->data[i]);
    		fprintf(fp, ",%f", f);
    	}

    	fprintf(fp, "\n");

    	if(ferror(fp)){
    		fclose(fp);
    		return false;
    	}

    	fclose(fp);
    	return true;
	}


    ////////////////////////////////////////////////////////////
    // matrix data compression
    // note: compressor destroys possible memory
    // aligmentations
    
    
    template <typename T>
    bool vertex<T>::compress() 
    {
      if(compressor != 0) return false; // already compressed
      
      compressor = new MemoryCompressor();
      
      compressor->setMemory(data, sizeof(T)*dataSize);
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
    bool vertex<T>::decompress() 
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
    bool vertex<T>::iscompressed() const 
    {
      return (compressor != 0);
    }
    

    template <typename T>
    float vertex<T>::ratio() const 
    {
      if(compressor == 0) return 1.0f;
      return compressor->ratio();
    }
    
    
    ////////////////////////////////////////////////////////////
    
    
    // copies vertex[start:(start+len-1)] = data[0:(len-1)]
    template <typename T>
    bool vertex<T>::importData(const T* data_,
			       unsigned int len,
			       unsigned int start) 
    {
      if(len == 0)
	len = dataSize - start;
      else if(len+start > dataSize)
	return false;
      if(start >= dataSize)
	return false;
      
      memcpy(this->data + start, data_, len*sizeof(T));
      
      return true;
    }
    
    
    // copies data[0:(len-1)] = vertex[start:(start+len-1)]
    template <typename T>
    bool vertex<T>::exportData(T* data_,
			       unsigned int len,
			       unsigned int start) const 
    {
      if(len == 0)
	len = dataSize - start;
      else if(len+start > dataSize)
	return false;
      if(start >= dataSize)
	return false;
      
      memcpy(data_, this->data + start, len*sizeof(T));
      
      return true;
    }


    template <typename T>
    void vertex<T>::toString(std::string& line) const 
    {
      if(this->size() == 0){ line = ""; return; }
      if(this->size() == 1){
	char buffer[20];
	double temp = 0.0;
	whiteice::math::convert(temp, (*this)[0]);
	snprintf(buffer, 20, "%f", temp);
	line = buffer;
	return;
      }

      line = "[";
      char buffer[20];
      double temp = 0.0;

      for(unsigned int i=0;i<this->size();i++){
	whiteice::math::convert(temp, (*this)[i]);
	snprintf(buffer, 20, " %f", temp);
	line += buffer;
      }

      line += "]";
      
      return;
    }
    
    ////////////////////////////////////////////////////////////
    
    /***************************************************/
    
    template <typename T>
    std::ostream& operator<<(std::ostream& ios,
			     const whiteice::math::vertex<T>& v)
    {
      if(v.size() == 1){
	ios << v[0];
      }
      else if(v.size() > 1){
	
	ios << "[";
	
	for(unsigned int i=0;i<v.size();i++){
	  ios << " " << v[i];
	}
	
	ios << " ]";
      }
    
      return ios;
    }
    
    
    // explicit template instantations
    
    template class vertex<float>;
    template class vertex<double>;
    template class vertex<complex<float> >;
    template class vertex<complex<double> >;
    
    //template class vertex<int>;
    //template class vertex<char>;
    //template class vertex<unsigned int>;
    //template class vertex<unsigned char>;
    
    template class vertex< blas_real<float> >;
    template class vertex< blas_real<double> >;
    template class vertex< blas_complex<float> >;
    template class vertex< blas_complex<double> >;
    
    template vertex<float> operator*<float>(const float& s, const vertex<float>& v);
    template vertex<double> operator*<double>(const double& s, const vertex<double>& v);
    
    template vertex<complex<float> > operator*<complex<float> >
      (const complex<float>& s, const vertex<complex<float> >& v);
    
    template vertex<complex<double> > operator*<complex<double> >
      (const complex<double>& s, const vertex<complex<double> >& v);
    
    //template vertex<int> operator*<int>(const int& s, const vertex<int>& v);
    //template vertex<char> operator*<char>(const char& s, const vertex<char>& v);
    //template vertex<unsigned int> operator*<unsigned int>(const unsigned int& s, const vertex<unsigned int>& v);
    //template vertex<unsigned char> operator*<unsigned char>(const unsigned char& s, const vertex<unsigned char>& v);
      
    template vertex<blas_real<float> > operator*<blas_real<float> >
      (const blas_real<float>& s, const vertex<blas_real<float> >& v);
									     
    template vertex<blas_real<double> > operator*<blas_real<double> >
      (const blas_real<double>& s, const vertex<blas_real<double> >& v);
    
    template vertex<blas_complex<float> > operator*<blas_complex<float> >
      (const blas_complex<float>& s, const vertex<blas_complex<float> >& v);
    
    template vertex<blas_complex<double> > operator*<blas_complex<double> >
      (const blas_complex<double>& s, const vertex<blas_complex<double> >& v);
       
    
    template std::ostream& operator<< <float>(std::ostream& ios, const vertex<float>&);
    template std::ostream& operator<< <double>(std::ostream& ios, const vertex<double>&);
    template std::ostream& operator<< <complex<float> >(std::ostream& ios, const vertex<complex<float> >&);
    template std::ostream& operator<< <complex<double> >(std::ostream& ios, const vertex<complex<double> >&);
    
    template std::ostream& operator<< <int>(std::ostream& ios, const vertex<int>&);
    template std::ostream& operator<< <char>(std::ostream& ios, const vertex<char>&);
    template std::ostream& operator<< <unsigned int>(std::ostream& ios, const vertex<unsigned int>&);
    template std::ostream& operator<< <unsigned char>(std::ostream& ios, const vertex<unsigned char>&);
    
    template std::ostream& operator<< <blas_real<float> >(std::ostream& ios, const vertex<blas_real<float> >&);
    template std::ostream& operator<< <blas_real<double> >(std::ostream& ios, const vertex<blas_real<double> >&);
    template std::ostream& operator<< <blas_complex<float> >(std::ostream& ios, const vertex<blas_complex<float> >&);
    template std::ostream& operator<< <blas_complex<double> >(std::ostream& ios, const vertex<blas_complex<double> >&);
    
  };
};




#endif
