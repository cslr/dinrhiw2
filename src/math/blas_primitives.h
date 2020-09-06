/*
 * atlas real and complex number
 * template data primitives
 */

#ifndef blas_primitives_h
#define blas_primitives_h

#include <stdio.h>
#include "ownexception.h"
#include <stdexcept>
#include <exception>
#include <complex>
#include <cassert>
#include <math.h>

namespace whiteice
{
  namespace math
  {
    
#ifdef __GNUG__
    // gnu c++ specific hack to keep sizeof(blas_real<T>) == sizeof(T)
    // (so there's no vtable, padding etc. to enlarge the 'struct'/class)
    // -> can malloc and use blas_xxx structures as pure types
    //    -> memcpy((T*)(blas_real<T>_pointer), xxx) etc.
    //    -> especially now ATLAS library can access data directly (as memory)

    template <typename T=float>
    inline void blas_safebox(const T& value){
#ifdef _GLIBCXX_DEBUG
      // in debugging mode we stop if data large (algorithms should work with smallish numbers)
      if(abs(value) > T(1000000.0f)){
	printf("BLAS VALUE TOO LARGE (larger than 10^6): %f\n", value);
	assert(0);
      }
#endif
    }
    
    template <typename T>
      struct blas_complex;

    
    template <typename T=float>
      struct blas_real
      {
	T c[1] __attribute__ ((packed));
	
	inline blas_real(){
	  c[0] = T(0.0f);
	  blas_safebox(c[0]);
	}
	
	inline blas_real(const T& t){
	  c[0] = t;
	  blas_safebox(c[0]);
	}
	
	//inline blas_real(const blas_real<T>& t){ c[0] = t.c[0]; }
	//inline blas_real(const blas_complex<T>& t){ c[0] = t.c[0]; }

	// work arounds stupid compiler..
	inline blas_real(const blas_complex<float>& t); // takes real part
	inline blas_real(const blas_complex<double>& t); // takes real part
	
	inline blas_real(const blas_real<float>& t){
	  c[0] = t.c[0];
	  blas_safebox(c[0]);
	}
	
	inline blas_real(const blas_real<double>& t){
	  c[0] = t.c[0];
	  blas_safebox(c[0]);
	}
	
	inline ~blas_real(){ }
	
	inline blas_real<T> operator++(int n){
	  if(n) c[0] += T(n);
	  else c[0]++;
	  
	  return (*this);
	}
	
	inline blas_real<T> operator+(const blas_real<T>& t) const 
	{ return blas_real<T>(this->c[0] + t.c[0]); }
	
	inline blas_real<T> operator-(const blas_real<T>& t) const 
	{ return blas_real<T>(this->c[0] - t.c[0]); }
	
	inline blas_real<T> operator*(const blas_real<T>& t) const 
	{ return blas_real<T>(this->c[0] * t.c[0]); }
	     
	inline blas_real<T> operator/(const blas_real<T>& t) const 
	{ return blas_real<T>(this->c[0] / t.c[0]); } // no division by zero checks

	inline blas_real<T> operator!() const  // complex conjugate
	{ return *this;}

	inline void conj(){ }   // complex conjugate (nothing to do with real valued data)
	  
	inline blas_real<T> operator-() const 
	{ return blas_real<T>(-this->c[0]); }
      
	inline blas_real<T>& operator+=(const blas_real<T>& t) 
	{
	  this->c[0] += t.c[0];
	  blas_safebox(c[0]);
	  return *this;
	}
	     
	inline blas_real<T>& operator-=(const blas_real<T>& t) 
	{
	  this->c[0] -= t.c[0];
	  blas_safebox(c[0]);
	  return *this;
	}
	
	inline blas_real<T>& operator*=(const blas_real<T>& t) 
	{
	  this->c[0] *= t.c[0];
	  blas_safebox(c[0]);
	  return *this;
	}
	     
	inline blas_real<T>& operator/=(const blas_real<T>& t) 
	{
	  this->c[0] /= t.c[0]; // no division by zero checks
	  blas_safebox(c[0]);
	  return *this;
	} 
	  
	inline blas_real<T>& operator=(const blas_real<T>& t) 
	{
	  this->c[0] = t.c[0];
	  blas_safebox(c[0]);
	  return *this;
	}
	
	inline bool operator==(const blas_real<T>& t) const 
	{ return (this->c[0] == t.c[0]); }
	     
	inline bool operator!=(const blas_real<T>& t) const 
	{ return (this->c[0] != t.c[0]); }
	     
	inline bool operator>=(const blas_real<T>& t) const 
	{ return (this->c[0] >= t.c[0]); }
	     
	inline bool operator<=(const blas_real<T>& t) const 
	{ return (this->c[0] <= t.c[0]); }
	     
	inline bool operator< (const blas_real<T>& t) const 
	{ return (this->c[0] < t.c[0]); }
	     
	inline bool operator> (const blas_real<T>& t) const 
	{ return (this->c[0] > t.c[0]); }
	
	inline bool operator==(const T& t) const 
	{ return (this->c[0] == t); }
	     
	inline bool operator!=(const T& t) const 
	{ return (this->c[0] != t); }
	     
	inline bool operator>=(const T& t) const 
	{ return (this->c[0] >= t); }
	     
	inline bool operator<=(const T& t) const 
	{ return (this->c[0] <= t); }
	     
	inline bool operator< (const T& t) const 
	{ return (this->c[0] < t); }
	     
	inline bool operator> (const T& t) const 
	{ return (this->c[0] > t); }
	
	// scalar operation
	inline blas_real<T>& operator= (const T& s) 
	{
	  this->c[0] = s;
	  blas_safebox(c[0]);
	  return *this;
	}

	inline blas_real<T> operator+=(const T& s) 
	{
	  this->c[0] += s;
	  blas_safebox(c[0]);
	  return *this;
	} 
	  
	inline blas_real<T>& operator-=(const T& s) 
	{
	  this->c[0] -= s;
	  blas_safebox(c[0]);
	  return *this;
	}
	     
	inline blas_real<T>  operator* (const T& s) const 
	{
	  blas_real<T> r;
	  r.c[0] = s * this->c[0];
	  return r;
	}
	     
	inline blas_real<T>  operator/ (const T& s) const 
	{ blas_real<T> r; r.c[0] =  this->c[0] / s; return r; } // no division by zero checks
	  
	inline blas_real<T>& operator*=(const T& s) 
	{
	  this->c[0] *= s;
	  blas_safebox(c[0]);
	  return *this;
	}
	     
	inline blas_real<T>& operator/=(const T& s) 
	{
	  this->c[0] /= s;
	  blas_safebox(c[0]);
	  return *this;
	}
	     
	inline blas_real<T> abs() const
	{ return blas_real<T>( T(fabs((double)c[0])) ); }
	
	inline T real() { return c[0]; }
	inline const T real() const { return c[0]; }

	inline T imag() { return T(0.0f); }
	inline const T imag() const { return T(0.0f); }

	inline T real(const T value)
	{
	  this->c[0] = value;
	  blas_safebox(c[0]);
	  return this->c[0];
	}
	
	inline T imag(const T value)
	{
	  blas_safebox(c[0]);
	  return T(0.0f); // has no imaginary value to set
	} 
	
	
	template <typename A>
	friend blas_real<A> operator*(const A& s, const blas_real<A>& r) ;
	
	template <typename A>
	friend blas_real<A> operator/(const A& s, const blas_real<A>& r) ;
	
	template <typename A>
	friend bool operator==(const A& t, const blas_real<A>& r) ;
	
	template <typename A>
	friend bool operator!=(const A& t, const blas_real<A>& r) ;
	
	template <typename A>
	friend bool operator>=(const A& t, const blas_real<A>& r) ;
	
	template <typename A>
	friend bool operator<=(const A& t, const blas_real<A>& r) ;
	
	template <typename A>
	friend bool operator< (const A& t, const blas_real<A>& r) ;
	
	template <typename A>
	friend bool operator> (const A& t, const blas_real<A>& r) ;
	
      } __attribute__ ((packed));
    
    
    
    template <typename T>
      inline blas_real<T> operator*(const T& s, const blas_real<T>& r) 
      {
	return blas_real<T>(r * s);
      }
    
    template <typename T>
      inline blas_real<T> operator/(const T& s, const blas_real<T>& r) 
      {
	return blas_real<T>(blas_real<T>(s) / r);
      }
    
    
    template <typename T>
      inline bool operator==(const T& t, const blas_real<T>& r) 
      { return (r.c[0] == t); }
    
    
    template <typename T>
      inline bool operator!=(const T& t, const blas_real<T>& r) 
      { return (r.c[0] != t); }
    
    
    template <typename T>
      inline bool operator>=(const T& t, const blas_real<T>& r) 
      { return (t >= r.c[0]); }
    
    
    template <typename T>
      inline bool operator<=(const T& t, const blas_real<T>& r) 
      { return (t <= r.c[0]); }
    
    
    template <typename T>
      inline bool operator< (const T& t, const blas_real<T>& r) 
      { return (t < r.c[0]); }
    
    
    template <typename T>
      inline bool operator> (const T& t, const blas_real<T>& r) 
      { return (t > r.c[0]); }
    
    
    
    
    
    
    
    template <typename T=float>
      struct blas_complex
      {
	T c[2] __attribute__ ((packed));
	
	
	inline blas_complex(){
	  c[0] = T(0.0f); c[1] = T(0.0f);
	  blas_safebox(c[0]); blas_safebox(c[1]);
	}
	
	inline blas_complex(const T& r){
	  c[0] = r; c[1] = T(0.0f);
	  blas_safebox(c[0]); blas_safebox(c[1]);
	}
	
	inline blas_complex(const T& r, const T& i){
	  c[0] = r; c[1] = i;
	  blas_safebox(c[0]); blas_safebox(c[1]);
	}
	
	//inline blas_complex(const blas_complex<T>& r){ c[0] = r.c[0]; c[1] = r.c[1]; }
	inline blas_complex(const std::complex<T>& z){
	  c[0] = std::real(z); c[1] = std::imag(z);
	  blas_safebox(c[0]); blas_safebox(c[1]);
	}
	
	// work arounds stupid compiler..
	inline blas_complex(const blas_real<float>& r){
	  c[0] = r.c[0]; c[1] = T(0.0f);
	  blas_safebox(c[0]); blas_safebox(c[1]);
	}
	
	inline blas_complex(const blas_real<double>& r){
	  c[0] = r.c[0]; c[1] = T(0.0f);
	  blas_safebox(c[0]); blas_safebox(c[1]);
	}
	
	inline blas_complex(const blas_complex<float>& r){
	  c[0] = r.c[0]; c[1] = r.c[1];
	  blas_safebox(c[0]); blas_safebox(c[1]);
	}
	
	inline blas_complex(const blas_complex<double>& r){
	  c[0] = r.c[0]; c[1] = r.c[1];
	  blas_safebox(c[0]); blas_safebox(c[1]);
	}
	
	inline ~blas_complex(){ }
	
	
	inline blas_complex<T> operator+(const blas_complex<T>& t) const 
	{ return blas_complex<T>(this->c[0] + t.c[0], this->c[1] + t.c[1]); }
	
	inline blas_complex<T> operator-(const blas_complex<T>& t) const 
	{ return blas_complex<T>(this->c[0] - t.c[0], this->c[1] - t.c[1]); }
	
	inline blas_complex<T> operator*(const blas_complex<T>& t) const 
	{ return blas_complex<T>(this->c[0] * t.c[0] - this->c[1]*t.c[1],
				  this->c[1] * t.c[0] + this->c[0]*t.c[1]); }
	
	// no division by zero checks
	inline blas_complex<T> operator/(const blas_complex<T>& t) const 
	{ blas_complex<T> r; r.c[0] = (c[0]*t.c[0] + c[1]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]);
	  r.c[1] = (c[1]*t.c[0] - c[0]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]); return r;}
	
	inline blas_complex<T> operator!() const   // complex conjugate
	{ return blas_complex<T>(this->c[0], -this->c[1]); }
	
	inline void conj()   // complex conjugate
	{
	  c[1] = -c[1];
	}
	
	inline blas_complex<T> operator-() const 
	{ return blas_complex<T>(-this->c[0], -this->c[1]); }
      
	inline blas_complex<T>& operator+=(const blas_complex<T>& t) 
	{
	  this->c[0] += t.c[0]; this->c[1] += t.c[1];
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	     
	inline blas_complex<T>& operator-=(const blas_complex<T>& t) 
	{
	  this->c[0] -= t.c[0]; this->c[1] -= t.c[1];
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	
	inline blas_complex<T>& operator*=(const blas_complex<T>& t) 
	{
	  T a = c[0] * t.c[0] - c[1]*t.c[1];
	  T b = c[1] * t.c[0] + c[0]*t.c[1];
	  this->c[0] = a; this->c[1] = b;
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	  
	// no division by zero checks
	inline blas_complex<T>& operator/=(const blas_complex<T>& t) 
	{
	  T a = (c[0]*t.c[0] + c[1]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]);
	  T b = (c[1]*t.c[0] - c[0]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]);
	  this->c[0] = a; this->c[1] = b;
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	  
	inline blas_complex<T>& operator=(const blas_complex<T>& t) 
	{
	  this->c[0] = t.c[0]; this->c[1] = t.c[1];
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	  
	inline blas_complex<T>& operator=(const blas_real<T>& t) 
	{
	  this->c[0] = t.c[0]; this->c[1] = T(0.0f);
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	
	inline bool operator==(const blas_complex<T>& t) const 
	{ return (this->c[0] == t.c[0] && this->c[1] == t.c[1]); }
	     
	inline bool operator!=(const blas_complex<T>& t) const 
	{ return (this->c[0] != t.c[0] && this->c[1] != t.c[1]); }
	     
	inline bool operator>=(const blas_complex<T>& t) const 
	{
	  const std::string error = "complex numbers cannot be compared";
	  printf("%s\n", error.c_str());
	  exit(-1);
	  throw uncomparable(error);
	}
	     
	inline bool operator<=(const blas_complex<T>& t) const 
	{
	  const std::string error = "complex numbers cannot be compared";
	  printf("%s\n", error.c_str());
	  exit(-1);
	  throw uncomparable(error);
	}
	     
	inline bool operator< (const blas_complex<T>& t) const 
	{
	  const std::string error = "complex numbers cannot be compared";
	  printf("%s\n", error.c_str());
	  exit(-1);
	  throw uncomparable(error);
	}
	     
	inline bool operator> (const blas_complex<T>& t) const 
	{
	  const std::string error = "complex numbers cannot be compared";
	  printf("%s\n", error.c_str());
	  exit(-1);
	  throw uncomparable(error);
	}
	
	inline bool operator==(const T& t) const 
	{ return (this->c[0] == t && this->c[1] == 0); }
	     
	inline bool operator!=(const T& t) const 
	{ return (this->c[0] != t && this->c[1] != 0); }
	     
	inline bool operator>=(const T& t) const 
	{
	  const std::string error = "complex numbers cannot be compared";
	  printf("%s\n", error.c_str());
	  exit(-1);
	  throw uncomparable(error);
	}
	     
	inline bool operator<=(const T& t) const 
	{
	  const std::string error = "complex numbers cannot be compared";
	  printf("%s\n", error.c_str());
	  exit(-1);
	  throw uncomparable(error);
	}
	     
	inline bool operator< (const T& t) const 
	{
	  const std::string error = "complex numbers cannot be compared";
	  printf("%s\n", error.c_str());
	  exit(-1);
	  throw uncomparable(error);
	}
	     
	inline bool operator> (const T& t) const 
	{
	  const std::string error = "complex numbers cannot be compared";
	  printf("%s\n", error.c_str());
	  exit(-1);
	  throw uncomparable(error);
	}
	
	// scalar operation
	inline blas_complex<T>& operator= (const T& s) 
	{
	  this->c[0] = s; this->c[1] = T(0.0f);
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	
	inline blas_real<T> operator+=(const T& s) 
	{
	  this->c[0] += s;
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	
	inline blas_real<T>& operator-=(const T& s) 
	{
	  this->c[0] -= s;
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	     
	inline blas_complex<T>  operator* (const T& s) const 
	{ blas_complex<T> r; r.c[0] = s * this->c[0]; r.c[1] = s * this->c[1]; return r; }
	
	// no division by zero checks
	inline blas_complex<T>  operator/ (const T& s) const 
	{ blas_complex<T> r; r.c[0] =  this->c[0] / s; r.c[1] = this->c[1] / s; return r; }
	  
	inline blas_complex<T>& operator*=(const T& s) 
	{
	  this->c[0] *= s; this->c[1] *= s;
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	     
	inline blas_complex<T>& operator/=(const T& s) 
	{
	  this->c[0] /= s; this->c[1] /= s;
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return *this;
	}
	     
	inline blas_real<T> abs() const
	{ blas_real<T> r; r.c[0] = T(sqrt((double)(c[0]*c[0] + c[1]*c[1]))); return r; }
	
	inline T real() { return c[0]; }
	inline const T real() const { return c[0]; }
	
	inline T imag() { return c[1]; }
	inline const T imag() const { return c[1]; }
	
	inline T real(const T value)
	{
	  this->c[0] = value;
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return this->c[0];
	}
	
	inline T imag(const T value)
	{
	  this->c[1] = value;
	  blas_safebox(c[0]); blas_safebox(c[1]);
	  return this->c[1];
	}
	
	template <typename A>
	friend blas_complex<A> operator*(const A& s, const blas_complex<A>& r) ;
	
	template <typename A>
	friend blas_complex<A> operator/(const A& s, const blas_complex<A>& r) ;
	
	template <typename A>
	friend bool operator==(const A& t, const blas_complex<A>& r) ;
	
	template <typename A>
	friend bool operator!=(const A& t, const blas_complex<A>& r) ;
	
	template <typename A>
	friend bool operator>=(const A& t, const blas_complex<A>& r) ;
	
	template <typename A>
	friend bool operator<=(const A& t, const blas_complex<A>& r) ;
	
	template <typename A>
	friend bool operator< (const A& t, const blas_complex<A>& r) ;
	
	template <typename A>
	friend bool operator> (const A& t, const blas_complex<A>& r) ;
	
      } __attribute__ ((packed));
    
    
    // works around stupid compiler where we cannot define function for blas_complex in blas_real
    // until blas_complex is defined
    template <typename T>
    inline blas_real<T>::blas_real(const blas_complex<float>& t){
      c[0] = t.c[0]; // takes real part
      blas_safebox(c[0]);
    } 
    
    template <typename T>
    inline blas_real<T>::blas_real(const blas_complex<double>& t){
      c[0] = t.c[0]; // takes real part
      blas_safebox(c[0]);
    } 
    


    template <typename T>
      inline blas_complex<T> operator*(const T& s, const blas_complex<T>& r) 
      {
	return blas_complex<T>(r * s);
      }
    
    
    template <typename T>
      inline blas_complex<T> operator/(const T& s, const blas_complex<T>& r) 
      {
	return blas_complex<T>(blas_complex<T>(s) / r);
      }
    
    
    template <typename T>
      inline bool operator==(const T& t, const blas_complex<T>& r) 
      { return (r.c[0] == t && r.c[1] == 0); }
    
    
    template <typename T>
      inline bool operator!=(const T& t, const blas_complex<T>& r) 
      { return (r.c[0] != t && r.c[1] != 0); }
    
    
    template <typename T>
      inline bool operator>=(const T& t, const blas_complex<T>& r) 
      {
	const std::string error = "complex numbers cannot be compared";
	printf("%s\n", error.c_str());
	exit(-1);
	throw uncomparable(error);
      }
    
    
    template <typename T>
      inline bool operator<=(const T& t, const blas_complex<T>& r) 
      {
	const std::string error = "complex numbers cannot be compared";
	printf("%s\n", error.c_str());
	exit(-1);
	throw uncomparable(error);
      }
    
    
    template <typename T>
      inline bool operator< (const T& t, const blas_complex<T>& r) 
      {
	const std::string error = "complex numbers cannot be compared";
	printf("%s\n", error.c_str());
	exit(-1);
	throw uncomparable(error);
      }
    
    
    template <typename T>
      inline bool operator> (const T& t, const blas_complex<T>& r) 
      {
	const std::string error = "complex numbers cannot be compared";
	printf("%s\n", error.c_str());
	exit(-1);
	throw uncomparable(error);
      }

    
    
    ////////////////////////////////////////////////////////////////////////////////
    // some basic math functions are in "blade_math.h"
    
    
    template <typename T>
      inline blas_real<T> sqrt(const blas_real<T>& a)
      {
	return blas_real<T>( T(::sqrt(a.c[0])) );
      }
    
    
    template <typename T>
      inline blas_complex<T> sqrt(const blas_complex<T>& a)
      {
	std::complex<T> t(a.c[0], a.c[1]);
	
	return blas_complex<T>( std::sqrt(t) );
      }
    
    
    

    
    //////////////////////////////////////////////////////////////////////
    // conversion functions

    // tries to convert blas_real of type S to blas_real of type T (B = A)
    template <typename T, typename S>
    inline bool convert(blas_real<T>& B, const blas_real<S>& A) 
    {
      try{ B.c[0] = static_cast<T>(A.c[0]); return true; }
      catch(std::exception& e){ return false; }
    }
    
    // tries to convert blas_complex of type S to blas_complex of type T (B = A)
    template <typename T, typename S>
    inline bool convert(blas_complex<T>& B, const blas_complex<S>& A) 
    {
      try{
	B.c[0] = static_cast<T>(A.c[0]);	
	B.c[1] = static_cast<T>(A.c[1]);
	return true;
      }
      catch(std::exception& e){ return false; }
    }
    
    // tries to convert blas_real of type S to blas_real of type T (B = A)
    template <typename T, typename S>
    inline bool convert(blas_real<T>& B, const blas_complex<S>& A) 
    {
      try{ B.c[0] = S(A.real()); return true; }
      catch(std::exception& e){ return false; }
    }
    
    
    // tries to convert blas_complex of type S to blas_complex of type T (B = A)
    template <typename T, typename S>
    inline bool convert(blas_complex<T>& B, const blas_real<S>& A) 
    {
      try{
	B.c[0] = T(A.c[0]);
	B.c[1] = T(0.0f);
	return true;
      }
      catch(std::exception& e){ return false; }
    }
    
    
    // tries to convert blas_real of type S to scalar of type T
    template <typename T, typename S>
    inline bool convert(T& B, const blas_real<S>& A) 
    {
      try{ B = T(A.c[0]); return true; }
      catch(std::exception& e){ return false; }
    }

#if 0
    // tries to convert scalar S to blas_real<T> type
    template <typename T, typename S>
    inline bool convert(blas_real<T>& B, const S& A)
    {
      try{ B.c[0] = T(A); return true; }
      catch(std::exception& e){ return false; }
    }
#endif
    

    
    

#else

#error "No packed BLAS primitives specified for this compiler."
    
#endif
    
    template <typename T>
      inline std::ostream& operator<<(std::ostream& ios,
				      const blas_real<T>& r){
      ios << r.c[0];
      return ios;
    }
    
  
    template <typename T>
      inline std::ostream& operator<<(std::ostream& ios,
				      const blas_complex<T>& r){
      ios << r.c[0] << " + ";
      ios << r.c[1] << "i";
      return ios;
    }
    
    
  };
};


#endif
