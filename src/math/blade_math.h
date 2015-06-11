// not tested,
// TODO: implement missing parts (this isn't fully done (afaik) and/or test
//


#include "dinrhiw_blas.h"
#include "number.h"
#include "ownexception.h"

#include "integer.h"
#include "gvertex.h"
#include "gmatrix.h"

#include "function.h"
#include "real.h"

// #include <cstdio>
#include <stddef.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

#include <complex>

#ifndef blade_math_h
#define blade_math_h

#ifndef M_PI
#define M_PI 3.14159265359
#endif


namespace whiteice
{
  
  namespace math
  {
    template <typename T> class vertex;
    template <typename T> class matrix;
    template <typename T> class quaternion;
    
    
    // inherits c++ complex class with comparision
    // operators (needed with whiteice::math::matrix
    // and whiteice::math::vertex )
    
    template <typename T>
      class complex : public std::complex<T>
      {
      public:
	// ctors
	
	complex(const T& realval = 0, const T& imagval = 0) :
	  std::complex<T>(realval, imagval){ }
	
	template <typename Other>
	  complex(const std::complex<Other>& right) :
	  std::complex<T>(right){ }
	
	
	// comparision operators
	
	bool operator>=(const std::complex<T>& v) const throw(uncomparable){
	  throw uncomparable("Cannot compare complex numbers."); }
	
	bool operator<=(const std::complex<T>& v) const throw(uncomparable){
	  throw uncomparable("Cannot compare complex numbers."); }
	
	bool operator< (const std::complex<T>& v) const throw(uncomparable){
	  throw uncomparable("Cannot compare complex numbers."); }
	
	bool operator> (const std::complex<T>& v) const throw(uncomparable){
	  throw uncomparable("Cannot compare complex numbers."); }
      };
    
    //////////////////////////////////////////////////
    
    // calculates 1d fast fourier transform
    // and its inverse

    template <unsigned int K, typename T>
      bool fft(vertex< whiteice::math::complex<T> >& v) throw();

    template <unsigned int K, typename T>
      bool ifft(vertex< whiteice::math::complex<T> >& v) throw();
    
    // TODO:
    // calculates 1d fast hartley transform and its inverse
    // this is only for real value data
    //
    //template <unsigned int K, typename T>
    //  bool fht(vertex<T, T>& v) throw();
    //
    //template <unsigned int K, typename T>
    //  bool ifht(vertex<T, T>& v) throw();
    
    
    //////////////////////////////////////////
    // convert() functions for float and doubles
    
    bool convert(float&  B, const float&  A) throw();
    bool convert(float&  B, const double& A) throw();
    bool convert(double& B, const float&  A) throw();
    bool convert(double& B, const double& A) throw();
    
    bool convert(float& B, const blas_real<float>& A) throw();
    bool convert(float& B, const blas_complex<float>& A) throw();
    bool convert(float& B, const blas_real<double>& A) throw();
    bool convert(float& B, const blas_complex<double>& A) throw();
    
    bool convert(double& B, const blas_real<float>& A) throw();
    bool convert(double& B, const blas_complex<float>& A) throw();
    bool convert(double& B, const blas_real<double>& A) throw();
    bool convert(double& B, const blas_complex<double>& A) throw();
    
    
    //////////////////////////////////////////////////////////////////////
    // square root
    
    double sqrt(double x) PURE_FUNCTION;
    float  sqrt(float x) PURE_FUNCTION;
    unsigned int sqrt(unsigned int x) PURE_FUNCTION;
    unsigned int sqrt(int x) PURE_FUNCTION;
    unsigned int sqrt(unsigned char x) PURE_FUNCTION;
    unsigned int sqrt(char x) PURE_FUNCTION;

    realnumber sqrt(const realnumber& r);
    
    
    template <typename T>
      T sqrt(const quaternion<T>& x) PURE_FUNCTION;
    
    template <typename T>
      T sqrt(const vertex<T>& v) PURE_FUNCTION;
    
    template <typename T>
      T sqrt(const matrix<T>& X) PURE_FUNCTION;
      
    template <typename T>
      T sqrt(const gvertex<T>& v) PURE_FUNCTION;
    
    template <typename T>
      T sqrt(const gmatrix<T>& X) PURE_FUNCTION;
    
    template <typename T>
      std::complex<T> sqrt(const std::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      whiteice::math::complex<T> sqrt(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    
    
    //////////////////////////////////////////////////////////////////////
   
    inline double sqrt(double x){ return std::sqrt(x); }
    inline float  sqrt(float x){ return std::sqrt(x); }
    
    inline unsigned int sqrt(unsigned int x){
      return ((unsigned int)std::sqrt((double)x));
    }
    
    inline unsigned int sqrt(int x){
      return ((int)std::sqrt((double)x));
    }

    inline unsigned int sqrt(unsigned char x){
      return ((unsigned char)std::sqrt((double)x));
    }
    
    inline unsigned int sqrt(char x){
      return ((char)std::sqrt((double)x));
    }
    
    
    inline realnumber sqrt(const realnumber& r){
      mpf_t result;
      mpf_init2(result, mpf_get_prec(r.data));
      mpf_sqrt(result, r.data);
      
      return realnumber(result);
    }
    
    
    template <typename T>
      T sqrt(const quaternion<T>& x)
      {
	quaternion<T> q = x; q.abs();
	return whiteice::math::sqrt(q[0]);
      }
    

    template <typename T>
      T sqrt(const vertex<T>& v)
      {
	if(v.size() == 1)
	  return whiteice::math::sqrt(v[0]);
	else{
	  std::cout << "sqrt vertex: bad params" << std::endl;
	  assert(0);
	}
      }
    
    
    template <typename T>
      T sqrt(const matrix<T>& X) 
      {
	if(X.xsize() == 1 && X.ysize() == 1)
	  return whiteice::math::sqrt(X[0][0]);
	else{
	  std::cout << "sqrt matrix: bad params" << std::endl;
	  assert(0);
	}
      }


    template <typename T>
      T sqrt(const gvertex<T>& v)
      {
	if(v.size() == 1)
	  return whiteice::math::sqrt(v[0]);
	else{
	  std::cout << "sqrt gvertex: bad params" << std::endl;
	  assert(0);
	}
      }
    
    template <typename T>
      T sqrt(const gmatrix<T>& X)
      {
	if(X.xsize() == 1 && X.ysize() == 1)
	  return whiteice::math::sqrt(X[0][0]);
	else{
	  std::cout << "sqrt gmatrix: bad params" << std::endl;
	  assert(0);
	}
      }
    
    
    template <typename T>
      std::complex<T> sqrt(const std::complex<T>& x)
      {
	return std::sqrt(x);
      }
    
    template <typename T>
      whiteice::math::complex<T> sqrt(const whiteice::math::complex<T>& x)
      {
	return whiteice::math::complex<T>(std::sqrt(x));
      }
    
    
    // sqrt() for atlas primitives is in "blas_primitives.h"
    
    
    
    //////////////////////////////////////////////////////////////////////
    // number rised to given power (x**y)
    
    double pow(double x, double y) PURE_FUNCTION;
    float  pow(float x,  float y) PURE_FUNCTION;
    unsigned int pow(unsigned int x, unsigned int y) PURE_FUNCTION;
    unsigned int pow(int x, int y) PURE_FUNCTION;
    unsigned int pow(unsigned char x, unsigned char y) PURE_FUNCTION;
    unsigned int pow(char x, char y) PURE_FUNCTION;
    
    // unlimited power ...
    realnumber pow(const realnumber& x, const realnumber& y);
    
    blas_real<float> pow(blas_real<float> x,
			  blas_real<float> y) PURE_FUNCTION;
    
    blas_real<double> pow(blas_real<double> x,
			   blas_real<double> y) PURE_FUNCTION;
    
    
    
    //////////////////////////////////////////////////////////////////////
    
    inline double pow(double x, double y){ return std::pow(x, y); }
    inline float  pow(float x,  float y){ return ::powf(x, y); }
    
    inline unsigned int pow(unsigned int x, unsigned int y){
      return ((unsigned int)std::pow((double)x, (double)y));
    }
    
    inline unsigned int pow(int x, int y){
      return ((int)std::pow((double)x, (double)y));
    }

    inline unsigned int pow(unsigned char x, unsigned char y){
      return ((unsigned char)std::pow((double)x, (double)y));
    }
    
    inline unsigned int pow(char x, char y){
      return ((char)std::pow((double)x, (double)y));
    }
    
    
    realnumber exp(const realnumber& x);
    realnumber log(const realnumber& x);
    
    // unlimited power is sure nice to have..
    // this doesn't work for x < 0 because
    // real logarithm (which this is using) isn't defined for negative numbers
    inline realnumber pow(const realnumber& x, const realnumber& y){
      return whiteice::math::exp(whiteice::math::log(x)*y);
    }
    
    
    inline blas_real<float> pow(blas_real<float> x,
				 blas_real<float> y){
      return blas_real<float>( ::powf(x.c[0], y.c[0]) );
    }
    
    
    inline blas_real<double> pow(blas_real<double> x,
				  blas_real<double> y){
      return blas_real<double>( std::pow(x.c[0], y.c[0]) );
    }
    
    
    
    //////////////////////////////////////////////////////////////////////
    // argument (angle) of (complex) number
    
    double arg(double x) PURE_FUNCTION;
    float arg(float x) PURE_FUNCTION;
    int arg(int x) PURE_FUNCTION;
    unsigned int arg(unsigned int x) PURE_FUNCTION;
    
    realnumber arg(const realnumber& x);
    
    template <typename T>
      std::complex<T> arg(const std::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      whiteice::math::complex<T> arg(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      blas_real<T> arg(const blas_real<T>& x) PURE_FUNCTION;
    
    template <typename T>
      blas_complex<T> arg(const blas_complex<T>& x) PURE_FUNCTION;
    
    //////////////////////////////////////////////////////////////////////
    
    //inline long double arg(long double x){ return 0.0; }
    inline double arg(double x){ return 0.0; }
    inline float arg(float x){ return 0.0f; }
    inline int arg(int x){ return 0; }
    inline unsigned int arg(unsigned int x){ return 0; }
    
    inline realnumber arg(const realnumber& x){
      return realnumber(0.0);
    }
    
    template <typename T>
      std::complex<T> arg(const std::complex<T>& x)
      {
	return std::arg(x);
      }
    
    template <typename T>
      whiteice::math::complex<T> arg(const whiteice::math::complex<T>& x)
      {
	return whiteice::math::complex<T>(std::arg(x));
      }
    
    
    template <typename T>
      blas_real<T> arg(const blas_real<T>& x){ return T(0.0); }
    
    template <typename T>
      blas_complex<T> arg(const blas_complex<T>& x)
      {
	return blas_complex<T>(std::arg(std::complex<T>(x.c[0], x.c[1])));
      }
    

    
    //////////////////////////////////////////////////////////////////////
    // nat. exponent of number
    
    inline double exp(double x) PURE_FUNCTION;
    inline float exp(float x) PURE_FUNCTION;
    inline int exp(int x) PURE_FUNCTION;
    inline unsigned int exp(unsigned int x) PURE_FUNCTION;
    
    realnumber exp(const realnumber& x);
    
    template <typename T>
      inline whiteice::math::complex<T> 
      exp(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      inline std::complex<T> 
      exp(const std::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      inline whiteice::math::blas_real<T> 
      exp(whiteice::math::blas_real<T> x) PURE_FUNCTION;
    
    template <typename T>
      inline whiteice::math::blas_complex<T> 
      exp(whiteice::math::blas_complex<T> x) PURE_FUNCTION;
    
    
    //////////////////////////////////////////////////////////////////////
    
    inline double exp(double x){ return ::exp(x); }
    inline float exp(float x){ return ::expf(x); }
    inline int exp(int x){ return ((int)::exp(x)); }
    inline unsigned int exp(unsigned int x){ return ((unsigned int)::exp(x)); }
    
    template <typename T>
      inline whiteice::math::complex<T> exp(const whiteice::math::complex<T>& x) {
        return whiteice::math::complex<T>(std::exp(x));
      }

    template <typename T>
      inline std::complex<T> exp(const std::complex<T>& x){
	return std::exp(x);
      }
    
    template <typename T>
      inline whiteice::math::blas_real<T> exp(whiteice::math::blas_real<T> x){
        return whiteice::math::blas_real<T>(whiteice::math::exp(x.c[0]));
      }
    
    template <typename T>
      inline whiteice::math::blas_complex<T> exp(whiteice::math::blas_complex<T> x){
        return whiteice::math::blas_complex<T>( whiteice::math::exp( std::complex<T>(x[0], x[1]) ) );
    }
    
    
    //////////////////////////////////////////////////////////////////////
    // nat. logarithm of number
    
    double log(double x) PURE_FUNCTION;
    float log(float x) PURE_FUNCTION;
    int log(int x) PURE_FUNCTION;
    unsigned int log(unsigned int x) PURE_FUNCTION;
    
    realnumber log(const realnumber& x);
    
    template <typename T>
      std::complex<T> 
      log(const std::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      whiteice::math::complex<T> 
      log(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      whiteice::math::blas_real<T> 
      log(whiteice::math::blas_real<T> x) PURE_FUNCTION;
    
    template <typename T>
      whiteice::math::blas_complex<T> 
      log(whiteice::math::blas_complex<T> x) PURE_FUNCTION;
    
    //////////////////////////////////////////////////////////////////////
    
    inline double log(double x){ return ::log(x); }
    inline float log(float x){ return logf(x); }
    inline int log(int x){ return ((int)log(((float)x))); }
    inline unsigned int log(unsigned int x){ return ((unsigned int)log(((float)x))); }    
    
    
    template <typename T>
      std::complex<T> log(const std::complex<T>& x)
      {
	return std::log(x);
      }
    
    
    template <typename T>
      whiteice::math::complex<T> log(const whiteice::math::complex<T>& x)
      {
	return whiteice::math::complex<T>(std::log(x));
      }    

    
    template <typename T>
      inline whiteice::math::blas_real<T> log(whiteice::math::blas_real<T> x)
      {
	return whiteice::math::blas_real<T>(whiteice::math::log(x.c[0]));
      }
    
    template <typename T>
      inline whiteice::math::blas_complex<T> log(whiteice::math::blas_complex<T> x)
      {
	return whiteice::math::blas_complex<T>( whiteice::math::log( std::complex<T>(x[0], x[1]) ) );
      }
    

    //////////////////////////////////////////////////////////////////////
    // log(gamma(x))
    
    double lgamma(double x, int* signp);
    float lgamma(float x, int* signp);
    whiteice::math::blas_real<float> lgamma(whiteice::math::blas_real<float> x, int* signp);
    whiteice::math::blas_real<double> lgamma(whiteice::math::blas_real<double> x, int* signp);
    
    
    //////////////////////////////////////////////////////////////////////
    
    inline double lgamma(double x, int* signp){
#ifdef WINOS
      // TODO: use configure script to figure out 
      // which gamma functions are available
      // (and what their parameters are)
      // I wonder how this can be such a mess:
      // Linux, MingW, (Free, others?)BSD are all different
      
      return lgamma(x, signp);
#else
      return lgamma_r(x, signp);
#endif
    }
    
    inline float lgamma(float x, int* signp){
#ifdef WINOS
      return lgamma(x, signp);
#else
      return lgammaf_r(x, signp);
#endif
    }
    
    inline whiteice::math::blas_real<float> lgamma(whiteice::math::blas_real<float> x, int* signp){
#ifdef WINOS
      return whiteice::math::blas_real<float>(lgamma(x.c[0], signp));
#else
      return whiteice::math::blas_real<float>(lgammaf_r(x.c[0], signp));
#endif
    }
    
    inline whiteice::math::blas_real<double> lgamma(whiteice::math::blas_real<double> x, int* signp){
#ifdef WINOS
      return whiteice::math::blas_real<double>(lgamma(x.c[0], signp));
#else
      return whiteice::math::blas_real<double>(lgamma_r(x.c[0], signp));
#endif
    }


    
    
    //////////////////////////////////////////////////////////////////////
    // calculates pi with given precision
    
    realnumber pi(unsigned long int prec);

    
    //////////////////////////////////////////////////////////////////////
    // sin(x)
    
    float sin(float x) PURE_FUNCTION;
    double sin(double x) PURE_FUNCTION;
    int sin(int x) PURE_FUNCTION;
    unsigned int sin(unsigned int x) PURE_FUNCTION;
    char sin(char x) PURE_FUNCTION;
    unsigned char sin(unsigned char x) PURE_FUNCTION;
    
    realnumber sin(const realnumber& x);
    
    blas_real<float> sin(const blas_real<float>& x) PURE_FUNCTION;
    blas_real<double> sin(const blas_real<double>& x) PURE_FUNCTION;
    blas_complex<float> sin(const blas_complex<float>& x) PURE_FUNCTION;
    blas_complex<double> sin(const blas_complex<double>& x) PURE_FUNCTION;
    
    
    //////////////////////////////////////////////////////////////////////
    
    inline float sin(float x){ return ::sinf(x); }
    inline double sin(double x){ return ::sin(x); }
    inline int sin(int x){ return (int)::sin((float)x); }
    inline unsigned int sin(unsigned int x){ return (unsigned int)::sin((float)x); }
    inline char sin(char x){ return (char)::sin((float)x); }
    inline unsigned char sin(unsigned char x){ return (unsigned char)::sin((float)x); }
    
    inline realnumber sin(const realnumber& x){
      assert(0);
      return realnumber(0.0);
    }
    
    inline blas_real<float> sin(const blas_real<float>& x){
      return blas_real<float>(::sinf(x.c[0]));
    }
    
    inline blas_real<double> sin(const blas_real<double>& x){
      return blas_real<double>(::sin(x.c[0]));
    }
    
    
    inline blas_complex<float> sin(const blas_complex<float>& x){
      return blas_complex<float>(std::sin<float>(std::complex<float>(x.c[0],x.c[1])));
    }
    
    inline blas_complex<double> sin(const blas_complex<double>& x){
      return blas_complex<double>(std::sin<double>(std::complex<double>(x.c[0],x.c[1])));
    }
    
    
    
    //////////////////////////////////////////////////////////////////////
    // cos
    
    float cos(float x) PURE_FUNCTION;
    double cos(double x) PURE_FUNCTION;
    int cos(int x) PURE_FUNCTION;
    unsigned int cos(unsigned int x) PURE_FUNCTION;
    char cos(char x) PURE_FUNCTION;
    unsigned char cos(unsigned char x) PURE_FUNCTION;
    
    realnumber cos(const realnumber& x);
    
    blas_real<float> cos(const blas_real<float>& x) PURE_FUNCTION;
    blas_real<double> cos(const blas_real<double>& x) PURE_FUNCTION;
    blas_complex<float> cos(const blas_complex<float>& x) PURE_FUNCTION;
    blas_complex<double> cos(const blas_complex<double>& x) PURE_FUNCTION;
    
    //////////////////////////////////////////////////////////////////////
    
    inline float cos(float x){ return ::cosf(x); }
    inline double cos(double x){ return ::cos(x); }
    inline int cos(int x){ return (int)::cos((float)x); }
    inline unsigned int cos(unsigned int x){ return (unsigned int)::cos((float)x); }
    inline char cos(char x){ return (char)::cos((float)x); }
    inline unsigned char cos(unsigned char x){ return (unsigned char)::cos((float)x); }
    
    inline realnumber cos(const realnumber& x){
      assert(0);
      return realnumber(0.0);
    }
    
    inline blas_real<float> cos(const blas_real<float>& x){
      return blas_real<float>(::cosf(x.c[0]));
    }
    
    inline blas_real<double> cos(const blas_real<double>& x){
      return blas_real<double>(::cos(x.c[0]));
    }
    
    inline blas_complex<float> cos(const blas_complex<float>& x){
      return blas_complex<float>(std::cos<float>(std::complex<float>(x.c[0],x.c[1])));
    }
    
    inline blas_complex<double> cos(const blas_complex<double>& x){
      return blas_complex<double>(std::cos<double>(std::complex<double>(x.c[0],x.c[1])));
    }
    
    //////////////////////////////////////////////////////////////////////
    // tanh
    
    float tanh(float x) PURE_FUNCTION;
    double tanh(double x) PURE_FUNCTION;
    int tanh(int x) PURE_FUNCTION;
    unsigned int tanh(unsigned int x) PURE_FUNCTION;
    char tanh(char x) PURE_FUNCTION;
    unsigned char tanh(unsigned char x) PURE_FUNCTION;
    
    realnumber tanh(const realnumber& x);
    
    blas_real<float> tanh(const blas_real<float>& x) PURE_FUNCTION;
    blas_real<double> tanh(const blas_real<double>& x) PURE_FUNCTION;
    blas_complex<float> tanh(const blas_complex<float>& x) PURE_FUNCTION;
    blas_complex<double> tanh(const blas_complex<double>& x) PURE_FUNCTION;
    
    //////////////////////////////////////////////////////////////////////
    
    inline float tanh(float x){ return ::tanhf(x); }
    inline double tanh(double x){ return ::tanh(x); }
    inline int tanh(int x){ return (int)::tanhf((float)x); }
    inline unsigned int tanh(unsigned int x){ return (unsigned int)::tanhf((float)x); }
    inline char tanh(char x){ return (char)::tanhf((float)x); }
    inline unsigned char tanh(unsigned char x){ return (unsigned char)::tanhf((float)x); }
    
    inline realnumber tanh(const realnumber& x){
      assert(0);
      return realnumber(0.0);
    }
    
    inline blas_real<float> tanh(const blas_real<float>& x){
      return blas_real<float>(::tanhf(x.c[0]));
    }
    
    inline blas_real<double> tanh(const blas_real<double>& x){
      return blas_real<double>(::tanh(x.c[0]));
    }
    
    inline blas_complex<float> tanh(const blas_complex<float>& x){
      return blas_complex<float>(std::tanh<float>(std::complex<float>(x.c[0],x.c[1])));
    }
    
    inline blas_complex<double> tanh(const blas_complex<double>& x){
      return blas_complex<double>(std::tanh<double>(std::complex<double>(x.c[0],x.c[1])));
    }

    //////////////////////////////////////////////////////////////////////
    // atanh
    
    float atanh(float x) PURE_FUNCTION;
    double atanh(double x) PURE_FUNCTION;
    int atanh(int x) PURE_FUNCTION;
    unsigned int atanh(unsigned int x) PURE_FUNCTION;
    char atanh(char x) PURE_FUNCTION;
    unsigned char atanh(unsigned char x) PURE_FUNCTION;
    
    realnumber atanh(const realnumber& x);
    
    blas_real<float> atanh(const blas_real<float>& x) PURE_FUNCTION;
    blas_real<double> atanh(const blas_real<double>& x) PURE_FUNCTION;
    blas_complex<float> atanh(const blas_complex<float>& x) PURE_FUNCTION;
    blas_complex<double> atanh(const blas_complex<double>& x) PURE_FUNCTION;
    
    //////////////////////////////////////////////////////////////////////
    
    inline float atanh(float x){ return ::atanhf(x); }
    inline double atanh(double x){ return ::atanh(x); }
    inline int atanh(int x){ return (int)::atanhf((float)x); }
    inline unsigned int atanh(unsigned int x){ return (unsigned int)::atanhf((float)x); }
    inline char atanh(char x){ return (char)::atanhf((float)x); }
    inline unsigned char atanh(unsigned char x){ return (unsigned char)::atanhf((float)x); }
    
    inline realnumber atanh(const realnumber& x){
      assert(0);
      return realnumber(0.0);
    }
    
    inline blas_real<float> atanh(const blas_real<float>& x){
      return blas_real<float>(::atanhf(x.c[0]));
    }
    
    inline blas_real<double> atanh(const blas_real<double>& x){
      return blas_real<double>(::atanh(x.c[0]));
    }
    
    inline blas_complex<float> atanh(const blas_complex<float>& x){
      return blas_complex<float>(std::atanh<float>(std::complex<float>(x.c[0],x.c[1])));
    }
    
    inline blas_complex<double> atanh(const blas_complex<double>& x){
      return blas_complex<double>(std::atanh<double>(std::complex<double>(x.c[0],x.c[1])));
    }
    
    //////////////////////////////////////////////////////////////////////
    // sinh
    
    float sinh(float x) PURE_FUNCTION;
    double sinh(double x) PURE_FUNCTION;
    int sinh(int x) PURE_FUNCTION;
    unsigned int sinh(unsigned int x) PURE_FUNCTION;
    char sinh(char x) PURE_FUNCTION;
    unsigned char sinh(unsigned char x) PURE_FUNCTION;
    
    realnumber sinh(const realnumber& x);
    
    blas_real<float> sinh(const blas_real<float>& x) PURE_FUNCTION;
    blas_real<double> sinh(const blas_real<double>& x) PURE_FUNCTION;
    blas_complex<float> sinh(const blas_complex<float>& x) PURE_FUNCTION;
    blas_complex<double> sinh(const blas_complex<double>& x) PURE_FUNCTION;
    
    //////////////////////////////////////////////////////////////////////
    
    inline float sinh(float x){ return ::sinhf(x); }
    inline double sinh(double x){ return ::sinh(x); }
    inline int sinh(int x){ return (int)::sinhf((float)x); }
    inline unsigned int sinh(unsigned int x){ return (unsigned int)::sinhf((float)x); }
    inline char sinh(char x){ return (char)::sinhf((float)x); }
    inline unsigned char sinh(unsigned char x){ return (unsigned char)::sinhf((float)x); }
    
    inline realnumber sinh(const realnumber& x){
      assert(0);
      return realnumber(0.0);
    }
    
    inline blas_real<float> sinh(const blas_real<float>& x){
      return blas_real<float>(::sinhf(x.c[0]));
    }
    
    inline blas_real<double> sinh(const blas_real<double>& x){
      return blas_real<double>(::sinh(x.c[0]));
    }
    
    inline blas_complex<float> sinh(const blas_complex<float>& x){
      return blas_complex<float>(std::sinh<float>(std::complex<float>(x.c[0],x.c[1])));
    }
    
    inline blas_complex<double> sinh(const blas_complex<double>& x){
      return blas_complex<double>(std::sinh<double>(std::complex<double>(x.c[0],x.c[1])));
    }

    //////////////////////////////////////////////////////////////////////
    // asinh

    float asinh(float x) PURE_FUNCTION;
    double asinh(double x) PURE_FUNCTION;
    int asinh(int x) PURE_FUNCTION;
    unsigned int asinh(unsigned int x) PURE_FUNCTION;
    char asinh(char x) PURE_FUNCTION;
    unsigned char asinh(unsigned char x) PURE_FUNCTION;
    
    realnumber asinh(const realnumber& x);
    
    blas_real<float> asinh(const blas_real<float>& x) PURE_FUNCTION;
    blas_real<double> asinh(const blas_real<double>& x) PURE_FUNCTION;
    blas_complex<float> asinh(const blas_complex<float>& x) PURE_FUNCTION;
    blas_complex<double> asinh(const blas_complex<double>& x) PURE_FUNCTION;
    
    //////////////////////////////////////////////////////////////////////
    
    inline float asinh(float x){ return ::asinhf(x); }
    inline double asinh(double x){ return ::asinh(x); }
    inline int asinh(int x){ return (int)::asinhf((float)x); }
    inline unsigned int asinh(unsigned int x){ return (unsigned int)::asinhf((float)x); }
    inline char asinh(char x){ return (char)::asinhf((float)x); }
    inline unsigned char asinh(unsigned char x){ return (unsigned char)::asinhf((float)x); }
    
    inline realnumber asinh(const realnumber& x){
      assert(0);
      return realnumber(0.0);
    }
    
    inline blas_real<float> asinh(const blas_real<float>& x){
      return blas_real<float>(::asinhf(x.c[0]));
    }
    
    inline blas_real<double> asinh(const blas_real<double>& x){
      return blas_real<double>(::asinh(x.c[0]));
    }

#if 0    
    inline blas_complex<float> asinh(const blas_complex<float>& x){
      return blas_complex<float>(std::asinh<float>(std::complex<float>(x.c[0],x.c[1])));
    }
    
    inline blas_complex<double> asinh(const blas_complex<double>& x){
      return blas_complex<double>(std::asinh<double>(std::complex<double>(x.c[0],x.c[1])));
    }
#endif

    //////////////////////////////////////////////////////////////////////
    // cosh
    
    float cosh(float x) PURE_FUNCTION;
    double cosh(double x) PURE_FUNCTION;
    int cosh(int x) PURE_FUNCTION;
    unsigned int cosh(unsigned int x) PURE_FUNCTION;
    char cosh(char x) PURE_FUNCTION;
    unsigned char cosh(unsigned char x) PURE_FUNCTION;
    
    realnumber cosh(const realnumber& x);
    
    blas_real<float> cosh(const blas_real<float>& x) PURE_FUNCTION;
    blas_real<double> cosh(const blas_real<double>& x) PURE_FUNCTION;
    blas_complex<float> cosh(const blas_complex<float>& x) PURE_FUNCTION;
    blas_complex<double> cosh(const blas_complex<double>& x) PURE_FUNCTION;
    
    //////////////////////////////////////////////////////////////////////
    
    inline float cosh(float x){ return ::coshf(x); }
    inline double cosh(double x){ return ::cosh(x); }
    inline int cosh(int x){ return (int)::coshf((float)x); }
    inline unsigned int cosh(unsigned int x){ return (unsigned int)::coshf((float)x); }
    inline char cosh(char x){ return (char)::coshf((float)x); }
    inline unsigned char cosh(unsigned char x){ return (unsigned char)::coshf((float)x); }
    
    inline realnumber cosh(const realnumber& x){
      assert(0);
      return realnumber(0.0);
    }
    
    inline blas_real<float> cosh(const blas_real<float>& x){
      return blas_real<float>(::coshf(x.c[0]));
    }
    
    inline blas_real<double> cosh(const blas_real<double>& x){
      return blas_real<double>(::cosh(x.c[0]));
    }
    
    inline blas_complex<float> cosh(const blas_complex<float>& x){
      return blas_complex<float>(std::cosh<float>(std::complex<float>(x.c[0],x.c[1])));
    }
    
    inline blas_complex<double> cosh(const blas_complex<double>& x){
      return blas_complex<double>(std::cosh<double>(std::complex<double>(x.c[0],x.c[1])));
    }
    
    
    //////////////////////////////////////////////////////////////////////
    // absolute value        
    
    double abs(double x) PURE_FUNCTION;
    float abs(float& x) PURE_FUNCTION;
    int abs(int x) PURE_FUNCTION;
    unsigned int abs(unsigned int x) PURE_FUNCTION;
    
    realnumber abs(const realnumber& x);
    
    template <typename T>
      blas_real<T> abs(blas_real<T>& x) PURE_FUNCTION;
    
    template <typename T>
      vertex<T> abs(const vertex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      quaternion<T> abs(const quaternion<T>& x) PURE_FUNCTION;
    
    template <typename T>
      matrix<T> abs(const matrix<T>& X) PURE_FUNCTION;
    
    template <typename T>
      gvertex<T> abs(const gvertex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      gmatrix<T> abs(const gmatrix<T>& X) PURE_FUNCTION;
    
    template <typename T>
      std::complex<T> abs(const std::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      whiteice::math::complex<T> abs(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    
    integer abs(const integer& x) PURE_FUNCTION;

    
    //////////////////////////////////////////////////////////////////////
    
    inline double abs(double x){ return std::fabs(x); }
    inline float abs(float& x){ return fabsf(x); }
    inline int abs(int x){ return std::abs(x); }
    inline unsigned int abs(unsigned int x){ return x; }
    
    inline realnumber abs(const realnumber& x){
      mpf_t result;
      mpf_init2(result, mpf_get_prec(x.data));
      mpf_abs(result, x.data);
      
      return realnumber(result);
    }
    
    
    template <typename T>
      blas_real<T> abs(blas_real<T>& x)
      {
	blas_real<T> y(x);
	y.c[0] = whiteice::math::abs(y.c[0]);
	
	return y;
      }
    
    template <typename T>
      vertex<T> abs(const vertex<T>& x){
	vertex<T> y = x;
	return y.abs();
      }
    
    
    template <typename T>
      quaternion<T> abs(const quaternion<T>& x){
	quaternion<T> y = x;
	return y.abs();
      }
    
    
    template <typename T>
      matrix<T> abs(const matrix<T>& X){
	matrix<T> Y = X;
	return Y.abs();
      }
    

    template <typename T>
      gvertex<T> abs(const gvertex<T>& x){
	gvertex<T> y = x;
	return y.abs();
      }
    
    
    template <typename T>
      gmatrix<T> abs(const gmatrix<T>& X){
	gmatrix<T> Y = X;
	return Y.abs();
      }    
    
    
    template <typename T>
      std::complex<T> abs(const std::complex<T>& x){
	return std::abs(x);
      }
    

    template <typename T>
      whiteice::math::complex<T> abs(const whiteice::math::complex<T>& x){
	return whiteice::math::complex<T>(std::abs(x));
      }
    
    
    inline integer abs(const integer& x){
      integer y(x);
      return y.abs();
    }
    
    
    // abs() for atlas primitives are in
    // "blas_primitives.h"
    
    
    //////////////////////////////////////////////////////////////////////
    // conjugate number
    
    double conj(double x) PURE_FUNCTION;
    float conj(float x) PURE_FUNCTION;
    int conj(int x) PURE_FUNCTION;
    unsigned int conj(unsigned int x) PURE_FUNCTION;
    
    realnumber conj(const realnumber& x);
    
    
    template <typename T>
      vertex<T> conj(const vertex<T>& x) PURE_FUNCTION;
    
    // interpret external conjugate of matrix
    // as hermite M^h operator
    template <typename T, typename S>
      matrix<T> conj(const matrix<T>& X) PURE_FUNCTION;
    
    template <typename T>
      std::complex<T> conj(const std::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      whiteice::math::complex<T> conj(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      blas_real<T> conj(const blas_real<T>& a) PURE_FUNCTION;
    
    template <typename T>
      blas_complex<T> conj(const blas_complex<T>& a) PURE_FUNCTION;
    
    
    //////////////////////////////////////////////////////////////////////
    
    inline double conj(double x){ return x; }
    inline float conj(float x){ return x; }
    inline int conj(int x){ return x; }
    inline unsigned int conj(unsigned int x){ return x; }
    
    inline realnumber conj(const realnumber& x){ return x; }
    
    
    template <typename T>
      vertex<T> conj(const vertex<T>& x)
      {
	vertex<T> y(x);
	return y.conj();
      }
    
    // interpret external conjugate of matrix
    // as hermite M^h operator
    template <typename T, typename S>
      matrix<T> conj(const matrix<T>& X)
      {
	matrix<T> Y(X);
	return Y.hermite();
      }

    
    template <typename T>
      std::complex<T> conj(const std::complex<T>& x)
      {
	return std::conj(x);
      }

    
    template <typename T>
      whiteice::math::complex<T> conj(const whiteice::math::complex<T>& x)
      {
	return std::conj(x);
      }
    
    
    template <typename T>
      blas_real<T> conj(const blas_real<T>& a)
      {
	return a;
      }
    
    template <typename T>
      blas_complex<T> conj(const blas_complex<T>& a)
      {
	blas_real<T> r;
	r.c[0] =  a.c[0];
	r.c[1] = -a.c[1];
	
	return r;
      }

    
    //////////////////////////////////////////////////////////////////////
    // returns real/imag part of number
    
    double real(double x) PURE_FUNCTION;
    float real(float x) PURE_FUNCTION;
    int real(int x) PURE_FUNCTION;
    unsigned int real(unsigned int x) PURE_FUNCTION;
    
    realnumber real(const realnumber& x);
    
    
    template <typename T>
      T real(const std::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      T real(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      blas_real<T> real(const blas_real<T>& x) PURE_FUNCTION;
    
    template <typename T>
      blas_complex<T> real(const blas_complex<T>& x) PURE_FUNCTION;
    
    double imag(double x) PURE_FUNCTION;
    float imag(float x) PURE_FUNCTION;
    int imag(int x) PURE_FUNCTION;
    unsigned int imag(unsigned int x) PURE_FUNCTION;
    
    realnumber imag(const realnumber& x);
    
    template <typename T>
      T imag(const std::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      T imag(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      blas_real<T> imag(const blas_real<T>& x) PURE_FUNCTION;
    
    template <typename T>
      blas_complex<T> imag(const blas_complex<T>& x) PURE_FUNCTION;
    
    
    //////////////////////////////////////////////////////////////////////
    
    inline double real(double x){ return x; }
    inline float real(float x){ return x; }
    inline int real(int x){ return x; }
    inline unsigned int real(unsigned int x){ return x; }
    
    inline realnumber real(const realnumber& x){ return x; }
    
    
    template <typename T>
      T real(const std::complex<T>& x)
      {
	return std::real(x);
      }    
    
    template <typename T>
      T real(const whiteice::math::complex<T>& x)
      {
	return std::real(x);
      }
    
    template <typename T>
      blas_real<T> real(const blas_real<T>& x)
      {
	return x.c[0];
      }    
    
    template <typename T>
      blas_complex<T> real(const blas_complex<T>& x)
      {
	return x.c[0];
      }
    
    
    inline double imag(double x){ return 0.0; }
    inline float imag(float x){ return 0.0f; }
    inline int imag(int x){ return 0; }
    inline unsigned int imag(unsigned int x){ return 0; }
    
    
    inline realnumber imag(const realnumber& x){
      return realnumber(0.0);
    }
    
    
    template <typename T>
      T imag(const std::complex<T>& x)
      {
	return std::imag(x);
      }    
    
    template <typename T>
      T imag(const whiteice::math::complex<T>& x)
      {
	return std::imag(x);
      }
    
    template <typename T>
      blas_real<T> imag(const blas_real<T>& x)
      {
	return T(0.0);
      }    
    
    template <typename T>
      blas_complex<T> imag(const blas_complex<T>& x)
      {
	return x.c[1];
      }
    
    
    //////////////////////////////////////////////////////////////////////
    
    double ceil(double x) PURE_FUNCTION;
    double floor(double x) PURE_FUNCTION;
    double trunc(double x) PURE_FUNCTION;
    
    float ceil(float x) PURE_FUNCTION;
    float floor(float x) PURE_FUNCTION;
    float trunc(float x) PURE_FUNCTION;
    
    int ceil(int x) PURE_FUNCTION;
    int floor(int x) PURE_FUNCTION;
    int trunc(int x) PURE_FUNCTION;
    
    unsigned int ceil(unsigned int x) PURE_FUNCTION;
    unsigned int floor(unsigned int x) PURE_FUNCTION;
    unsigned int trunc(unsigned int x) PURE_FUNCTION;
    
    realnumber ceil(realnumber x);
    realnumber floor(realnumber x);
    realnumber trunc(realnumber x);
    
    template <typename T>
      std::complex<T> ceil(const std::complex<T>& x) PURE_FUNCTION;
    template <typename T>
      std::complex<T> floor(const std::complex<T>& x) PURE_FUNCTION;
    template <typename T>
      std::complex<T> trunc(const std::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      whiteice::math::complex<T> ceil(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    template <typename T>
      whiteice::math::complex<T> floor(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    template <typename T>
      whiteice::math::complex<T> trunc(const whiteice::math::complex<T>& x) PURE_FUNCTION;
    
    template <typename T>
      blas_real<T> ceil(const blas_real<T>& x) PURE_FUNCTION;
    template <typename T>
      blas_real<T> floor(const blas_real<T>& x) PURE_FUNCTION;
    template <typename T>
      blas_real<T> trunc(const blas_real<T>& x) PURE_FUNCTION;
    
    template <typename T>
      blas_complex<T> ceil(const blas_complex<T>& x) PURE_FUNCTION;
    template <typename T>
      blas_complex<T> floor(const blas_complex<T>& x) PURE_FUNCTION;
    template <typename T>
      blas_complex<T> trunc(const blas_complex<T>& x) PURE_FUNCTION;
    
    //////////////////////////////////////////////////////////////////////
    
    
    inline double ceil(double x) { return ::ceil(x); }
    inline double floor(double x){ return ::floor(x); }
    inline double trunc(double x){ return ::trunc(x); }
    
    inline float ceil(float x) { return ::ceilf(x); }
    inline float floor(float x){ return ::floorf(x); }
    inline float trunc(float x){ return ::truncf(x); }
    
    inline int ceil(int x) { return x; }
    inline int floor(int x){ return x; }
    inline int trunc(int x){ return x; }
    
    inline unsigned int ceil(unsigned int x) { return x; }
    inline unsigned int floor(unsigned int x){ return x; }
    inline unsigned int trunc(unsigned int x){ return x; }
    
    inline realnumber ceil(realnumber x){
      return realnumber(x).ceil();
    }
    
    inline realnumber floor(realnumber x){
      return realnumber(x).floor();
    }
    
    inline realnumber trunc(realnumber x){
      return realnumber(x).trunc();
    }
    
    
    template <typename T>
    inline std::complex<T> ceil(const std::complex<T>& x){ 
      x.real() = whiteice::math::ceil(x.real());
      x.imag() = whiteice::math::ceil(x.imag());
    }
    
    template <typename T>
    inline std::complex<T> floor(const std::complex<T>& x){
      x.real() = whiteice::math::floor(x.real());
      x.imag() = whiteice::math::floor(x.imag());
    }
    
    template <typename T>
    inline std::complex<T> trunc(const std::complex<T>& x){
      x.real() = whiteice::math::trunc(x.real());
      x.imag() = whiteice::math::trunc(x.imag());
    }
    
    
    template <typename T>
    inline whiteice::math::complex<T> ceil(const whiteice::math::complex<T>& x){
      x.real() = whiteice::math::ceil(x.ceil());
      x.imag() = whiteice::math::ceil(x.imag());
    }
    
    template <typename T>
    inline whiteice::math::complex<T> floor(const whiteice::math::complex<T>& x){
      x.real() = whiteice::math::floor(x.real());
      x.imag() = whiteice::math::floor(x.imag());
    }
    
    template <typename T>
    inline whiteice::math::complex<T> trunc(const whiteice::math::complex<T>& x){
      x.real() = whiteice::math::trunc(x.real());
      x.imag() = whiteice::math::trunc(x.imag());
    }
    
    
    template <typename T>
    inline blas_real<T> ceil(const blas_real<T>& x){
      return blas_real<T>(whiteice::math::ceil(x.c[0]));
    }
    
    template <typename T>
    inline blas_real<T> floor(const blas_real<T>& x){
      return blas_real<T>(whiteice::math::floor(x.c[0]));
    }
    
    template <typename T>
    inline blas_real<T> trunc(const blas_real<T>& x){
      return blas_real<T>(whiteice::math::trunc(x.c[0]));
    }
    
    
    template <typename T>
    inline blas_complex<T> ceil(const blas_complex<T>& x){
      return blas_complex<T>(whiteice::math::ceil(x.c[0]),
			      whiteice::math::ceil(x.c[1]));
    }
    
    template <typename T>
    inline blas_complex<T> floor(const blas_complex<T>& x){
      return blas_complex<T>(whiteice::math::floor(x.c[0]),
			      whiteice::math::floor(x.c[1]));
    }
    
    template <typename T>
    inline blas_complex<T> trunc(const blas_complex<T>& x){
      return blas_complex<T>(whiteice::math::trunc(x.c[0]),
			     whiteice::math::trunc(x.c[1]));
    }
    
    
    //////////////////////////////////////////////////////////////////////
    // FFT implementation
    
    unsigned int bitreverse(unsigned int index,
			    unsigned int bits); // for (i)fft
    
    /* template class for 2^K length FFT
     * there's lots of room for (non-assymptotic) improvements (this is basic FFT),
     * (for example MMX/SIMD/3dNow can be used on x86)
     * for example of good implementation FFTW (www.fftw.org)
     * (GPLed though (so don't look at the exact code, only released public papers))
     */
    template <unsigned int K, typename T>
      bool fft(vertex< whiteice::math::complex<T> >& v) throw()
      {
	using namespace std;
	
	const unsigned int N = (unsigned int)(pow(2.0,(double)K)); // 2^K size
	if(v.size() != N) return false;
	
	// reorders v
	{
	  const vertex< whiteice::math::complex<T> > vv(v);
	  
	  for(unsigned int i=0;i<N;i++)
	    v[bitreverse(i,K)] = vv[i]; // bitreverse is slow, should be precalculated
	}
	
	// calculates fft iteratively
	// (ref Introduction to Algoritms / Thomas Cormen)
	
	unsigned int m = 1;      
	
	for(unsigned int s=1;s<=K;s++){
	  
	  m *= 2; // m = 2^s
	  
	  whiteice::math::complex<T> u, t;
	  whiteice::math::complex<T> w   = 1;
	  whiteice::math::complex<T> w_m(T(0), 
					 T(2.0*M_PI/T(m)));
	  
	  w_m = whiteice::math::exp(w_m);
	  
	  for(unsigned int j=0;j<m/2;j++){
	    for(unsigned int k=j;k<N;k+=m){
	      t = w * v[ k + m/2 ];
	      u = v[ k ];
	      v[k] = u + t;
	      v[k + m/2] = u - t;
	    }
	    
	    w *= w_m;
	  }
	}
	
	return true;
      }
    
    
    /*
     * inverse fft
     * K is gives the length of fft, length = 2^K
     */
    template <unsigned int K, typename T>
      bool ifft(vertex< whiteice::math::complex<T> >& v) throw()
      {
	using namespace std;
	
	const unsigned int N = (unsigned int)(pow(2.0,(double)K)); // 2^K size
	if(v.size() != N) return false;
	
	// reorders v
	{
	  const vertex< whiteice::math::complex<T> > vv(v);
	  
	  for(unsigned int i=0;i<N;i++)
	    v[bitreverse(i,K)] = vv[i];
	}
	
	// calculates ifft iteratively
	// (ref Introduction to Algoritms / Thomas Cormen)
	
	unsigned int m = 1;      
	
	for(unsigned int s=1;s<=K;s++){
	  
	  m *= 2; // m = 2^s
	  
	  whiteice::math::complex<T> u, t;
	  whiteice::math::complex<T> w   = 1;
	  whiteice::math::complex<T> w_m(T(0), 
					 T(-2.0*M_PI/T(m)));
	  
	  w_m = whiteice::math::exp(w_m);
	  
	  for(unsigned int j=0;j<m/2;j++){
	    for(unsigned int k=j;k<N;k+=m){
	      t = w * v[ k + m/2 ];
	      u = v[ k ];
	      v[k] = u + t;
	      v[k + m/2] = u - t;
	    }
	    
	    w *= w_m;
	  }
	}
	
	
	for(unsigned int i=0;i<N;i++)
	  v[i] /= N;
	
	return true;      
      }
    
  }
}



#include "vertex.h"
#include "matrix.h"
#include "quaternion.h"


#endif








