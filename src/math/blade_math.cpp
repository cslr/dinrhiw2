

#include "blade_math.h"
#include "real.h"

#include <cmath>
#include <math.h>
#include <gmp.h>

namespace whiteice
{  
  namespace math
  {
    
    realnumber exp(const realnumber& x)
    {
      /*
       * Taylor series based exponentation implementation
       * with lots of tricks to calculate exp(x) where x
       * is close to zero.
       * 
       * Original MATLAB implementations:
       * m2_exp.m m1_exp.m
       * Theory: AlgoChapter4.html && AlgoChapter5.html (not by me)
       * 
       */
      
      signed long int E;
      mpf_get_d_2exp(&E, x.data); // finds E: x = d*2^E
      
      bool negative = false;
      if(E < 0){
	negative = true;
	E = -E;
      }
      
      const unsigned long int P = mpf_get_prec(x.data);
      
      mpf_t d, t, t2, t3, t4, t5;      
      mpf_init2(d, P);
      mpf_init2(t, P);
      mpf_init2(t2, P);
      mpf_init2(t3, P);
      mpf_init2(t4, P);
      mpf_init2(t5, P);
      
      // d = x*2^-E
      
      if(negative == false)
	mpf_div_2exp(t2, x.data, (unsigned)E);
      else
	mpf_set(t2, x.data);
      
      ///////////////////////////////////////////////////////////
      // exponentation for values near zero
      // d = exp(d); (taylor series with extra tricks)
      // 
      
      {
	mpf_div_ui(d, t2, 2); // d = d/2  (0.25, 0.5)
	
	unsigned long int iters = 0;
	
	// calculates number of iterations required
	// for a working accuracy
	if(mpz_sgn(d)){
	  
	  signed long int N;
	  mpf_get_d_2exp(&N, d);
	  N = -N;
	  if(!N)
	    N = 1;
	    
	  iters = P/N;
	  iters += 3;
	}
	
	
	mpf_set_ui(t, 0);  // t = exp(d) approx.
	mpf_set_ui(t2, 1); // t2 = 1/(k!)
	mpf_set_ui(t3, 1); // t3 = d^k
	
	
	iters /= 2; // (unrolled for efficiency: save num vars used)
	
	for(unsigned long int i=1;i<=iters;i++){
	  mpf_div_ui(t4, t2, 2*i - 1); // b = b/i  (so that b(x) = 1/x!)
	  mpf_mul(t2, t3, d);          // z = z*x
	  mpf_mul(t3, t4, t2);         // b*z
	  mpf_add(t5, t, t3);          // y = y + b*z
	  
	  mpf_mul(t3, t2, d);          // z = z*x
	  mpf_div_ui(t2, t4, 2*i);     // b = b/i
	  mpf_mul(t4, t2, t3);         // b*z
	  mpf_add(t,  t5, t4);         // y = y + b*z
	  
	}
	
	// exp(d) = 1 + 2*t + t*t = (t+1)*(t+1)
	mpf_add_ui(t2, t, 1);
	mpf_mul(d, t2, t2);  // both input arguments can be same(?)
      }
      
      
      ///////////////////////////////////////////////////////////
      
      // calculates d^(2^E)
      
      if(negative){ // E < 0:
	// we didn't divide x with 2^E because
	// x was already small enough (x <= 0.5)
	//
	// -> we can directly use the result
	
	mpf_clear(t);
	mpf_clear(t2);
	mpf_clear(t3);
	mpf_clear(t4);
	mpf_clear(t5);
	
	return realnumber(d);
      }
      else{

	if(E != 0){
	  mpz_t temp, q, r;
	  
	  // note: this is very slow for large exponents
	  // is there smart way to get away from creating so large
	  // exponents?

	  mpz_init2(temp, E+1); // 2^E requires E+1 bits
	  mpz_init2(q, E+1);    // 2^E requires E+1 bits
	  mpz_init2(r, E+1);    // 2^E requires E+1 bits
	  
	  
	  // temp = 2^|E|
	  mpz_ui_pow_ui(temp, 2, (unsigned)E);
	  
	  // implementation supports exponents > MAX_UNSIGNED_LONT_INT
	  
	  // temp = q*MAX_UNSIGNED_LONG_INT + r
	  mpz_tdiv_qr_ui(q, r, temp, (unsigned long int)(-1L));
	  // mpz_clear(temp);
	  
	  // d^temp = (d^(MAX_UNSIGNED_LONG_INT))^q * d^r
	  
	  mpf_set(t2, d);
	  mpf_pow_ui(t, d, (unsigned long int)(-1L));
	  mpf_pow_ui(d, t, mpz_get_ui(q));
	  mpf_pow_ui(t, t2, mpz_get_ui(r));
	  
	  mpf_mul(t2, t, d);
	  
	  mpz_clear(q);
	  mpz_clear(r);
	  mpz_clear(temp);
	  
	  mpf_clear(t);
	  mpf_clear(t3);
	  mpf_clear(t4);
	  mpf_clear(t5);
	  mpf_clear(d);
	  
	  return realnumber(t2);
	}
	else{
	  mpf_clear(t);
	  mpf_clear(t2);
	  mpf_clear(t3);
	  mpf_clear(t4);
	  mpf_clear(t5);
	  
	  return realnumber(d);
	}
      }
      
    }
    
    
    //////////////////////////////////////////////////////////////////////
    
    
    // logarithm function for realnumber
    inline void __primi_log(mpf_t& x);
    
    realnumber log(const realnumber& x){
      /*
       * Taylor series (at x = 1) based natural logarithm with 
       * lots of tricks to calculate log(x) only when it is close to 1.
       * 
       * Implements matlab code: m1_log.m and m2_log2.m
       */
      
      signed long int E;
      mpf_get_d_2exp(&E, x.data); // finds E: x = d*2^E
      
      mpf_t d, t, t2;
      unsigned long int P = mpf_get_prec(x.data);
      
      mpf_init2(d,  P);
      mpf_init2(t,  P);
      mpf_init2(t2, P);
      
      // d = x*2^-E
      if(E >= 0)
	mpf_div_2exp(d, x.data, (unsigned long int)E);
      else{
	mpf_mul_2exp(d, x.data, (unsigned long int)(-E));
      }
      
      // y = log(d) + ex*log(2)
      
      mpf_set_ui(t, 2);
      
      __primi_log(d);
      __primi_log(t); // can be cached
      
      // we can only multiple with unsigned integers
      if(E >= 0){
	mpf_mul_ui(t2, t, (unsigned long int)E);
	mpf_add(t, d, t2);
      }
      else{
	mpf_mul_ui(t2, t, (unsigned long int)(-E));
	mpf_sub(t, d, t2);
      }
      
      
      mpf_clear(d);
      mpf_clear(t2);
      
      return realnumber(t);
    }
    
    
    // logarithm which calculates logarithm fastest
    // for values close to 1 but works with any x
    // although calculations can be considerable slower
    inline void __primi_log(mpf_t& x){
      
      bool negate = false;
      unsigned long int iters = 0;
      unsigned long int P = mpf_get_prec(x);
      
      mpf_t y ,t, t2, t3;
      mpf_init2(t, P);
      
      if(mpf_cmp_ui(x, 1UL) > 0){
	mpf_ui_div(t, 1, x); // x = 1/x
	negate = true;
	
	mpf_ui_sub(x, 1, t); // x = 1 - x
      }
      else{
	mpf_ui_sub(t, 1, x); // x = 1 - x
	mpf_swap(t, x);
      }
      
      // x < 1
      
      
      if(mpf_sgn(x)){
	signed long int N;
	mpf_get_d_2exp(&N, x);
	N = -N;
	if(N == 0)
	  N = 1;
	
	iters = P;
	iters /= N;
	iters += 3;
      }
      
      mpf_init2(y, P);
      mpf_init2(t2, P);
      mpf_init2(t3, P);
      
      mpf_set_ui(y, 0);
      mpf_set_ui(t, 1);
      
      iters /= 2; // unrolling: saves number of variables in loop
      
      for(unsigned long int i=1;i<=iters;i++){
	mpf_mul(t2, t, x);    // t2 = x^k-1 * x
	mpf_div_ui(t, t2, 2*i - 1); // t  = x^k / k
	mpf_sub(t3, y, t);    // t3 = y - t
	
	mpf_mul(t, t2, x);    // .. same operations again
	mpf_div_ui(t2, t, 2*i);
	mpf_sub(y, t3, t2);
      }
      
      
      if(negate)
	mpf_neg(x, y);
      else
	mpf_swap(x, y);
      
      mpf_clear(y);
      mpf_clear(t);
      mpf_clear(t2);
      mpf_clear(t3);
      
    }
    
    //////////////////////////////////////////////////////////////////////
    // calculates pi for given precision
    
    realnumber pi(unsigned long int prec){
      // Salamin-Brent algorithm
      
      if(prec < 1)
	prec = 1;
      
      mpf_t ak, bk, ck, sk, pk, t;
      mpf_init2(ak, prec);
      mpf_init2(bk, prec);
      mpf_init2(ck, prec);
      mpf_init2(sk, prec);
      mpf_init2(pk, prec);
      mpf_init2(t, prec);
      
      mpf_set_ui(ak, 1UL);           // a0 = 1
      mpf_div_2exp(sk, ak, 1UL);     // s0 = 1/2
      mpf_sqrt(bk, sk);              // b0 = 1/sqrt(2);
      
      // calculate N from prec
      // 
      // approximatedly (from below): 
      // prec(i) = 2^(1.015*iter)
      // (where prec is in digits, in bits this 10 ~ 8 ~ 2**3 (3 bits)
      // iter    = log(prec)/(1.015*log(2))
      // 
      // it is important to *NOT* to use too many iterations either
      // otherwise roundoff errors can creep into equations and become
      // very large. The equation below has been hand-tuned to give
      // correct results. Usually only the last 5 decimal digits are wrong.
      // 
      // The equation below was found by knowing that precision increases as O(2^x)
      // and by manually testing how well the last digits were calculated
      // by comparing results with one which were computed with higher precision.
      // More formal mathematical optimization approach could probably find somewhat
      // better formula for calculating optimal number of iterations.
      // 
      // For digits < 1000, computer approximations were compared to pi values
      // given in various books.
      // 
      // Too many iterations can really destroy the accuracy in computations.
      // For example, with precision of 50000 and too many iterations you
      // get 4.1xxx*10e-4 as approximated value of pi if you use very many iterations 
      // (150 000 iterations).
      
      const unsigned long int N = (unsigned long int)
	(0.5 + whiteice::math::log((double)prec)/(1.015*whiteice::math::log(2.0)));
      
      for(unsigned long int k=1;k<N;k++){
	// bk = sqrt(a(k-1)*b-(k-1))
	mpf_mul(t, ak, bk); // bk^2 = a(k-1)*b(k-1)	
	
	// ak = (a(k-1) + b(k-1))/2
	mpf_add(pk, ak, bk);
	mpf_div_ui(ak, pk, 2UL);
	mpf_mul(pk, ak, ak); // ak^2
	
	// updates bk when it is no longer needed
	mpf_sqrt(bk, t); // bk = sqrt(a(k-1)*b(k-1))
	
	// ck = ak*ak - bk*bk
	mpf_sub(ck, pk, t);
	
	// sk = sk - 2^k * ck
	mpf_mul_2exp(t, ck, k);
	mpf_sub(ck, sk, t);
	mpf_swap(ck, sk);
      }
      
      // pk = (2*ak*ak)/sk
      mpf_mul_2exp(t, pk, 1);
      mpf_div(pk, t, sk);
      
      
      mpf_clear(ak);
      mpf_clear(bk);
      mpf_clear(ck);
      mpf_clear(sk);
      mpf_clear(t);
      
      
      return realnumber(pk);
    }
    
    
    
    //////////////////////////////////////////////////////////////////////
    
    // conversion function between doubles and floats
    bool convert(float&  B, const float&  A) throw(){ B = A; return true; }
    bool convert(float&  B, const double& A) throw(){ B = (float)A; return true; }
    bool convert(double& B, const float&  A) throw(){ B = (double)A; return true; }
    bool convert(double& B, const double& A) throw(){ B = A; return true; }
    
    bool convert(double& B, const char& A) throw(){ B = (double)A; return true; }
    bool convert(double& B, const unsigned char A) throw(){ B = (double)A; return true; }
    bool convert(double& B, const int& A) throw(){ B = (double)A; return true; }
    bool convert(double& B, const unsigned int& A) throw(){ B = (double)A; return true; }
    
    bool convert(unsigned int& B, const float& A) throw()  { B = (unsigned int)A; return true; }
    bool convert(unsigned int& B, const double& A) throw() { B = (unsigned int)A; return true; }
    bool convert(int& B, const float& A) throw()           { B = (int)A; return true; }
    bool convert(int& B, const double& A) throw()          { B = (int)A; return true; }

    bool convert(unsigned int& B, const blas_real<float>& A) throw()     { B = (unsigned int)A.c[0]; return true; }
    bool convert(unsigned int& B, const blas_complex<float>& A) throw()  { B = (unsigned int)A.real(); return true; }
    bool convert(unsigned int& B, const blas_real<double>& A) throw()    { B = (unsigned int)A.c[0]; return true; }
    bool convert(unsigned int& B, const blas_complex<double>& A) throw() { B = (unsigned int)A.real(); return true; }

    bool convert(int& B, const blas_real<float>& A) throw()              { B = (int)A.c[0]; return true; }
    bool convert(int& B, const blas_complex<float>& A) throw()           { B = (int)A.real(); return true; }
    bool convert(int& B, const blas_real<double>& A) throw()             { B = (int)A.c[0]; return true; }
    bool convert(int& B, const blas_complex<double>& A) throw()          { B = (int)A.real(); return true; }

    bool convert(float& B,  const blas_real<float>& A) throw(){ B = (float)A.c[0]; return true; }
    bool convert(float& B,  const blas_real<double>& A) throw(){ B = (float)A.c[0]; return true; }
    bool convert(double& B, const blas_real<float>& A) throw(){ B = (double)A.c[0]; return true; }
    bool convert(double& B, const blas_real<double>& A) throw(){ B = (double)A.c[0]; return true; }
    
    bool convert(float& B,  const blas_complex<float>& A) throw(){ B = (float)A.real(); return false; }
    bool convert(float& B,  const blas_complex<double>& A) throw(){ B = (float)A.real(); return false; }
    bool convert(double& B, const blas_complex<float>& A) throw(){ B = (double)A.real(); return false; }
    bool convert(double& B, const blas_complex<double>& A) throw(){ B = (double)A.real(); return false; }


    bool convert(blas_real<float>& B, const blas_complex<float>& A)  { B = (float)A.real(); return true; }
    bool convert(blas_real<float>& B, const blas_complex<double>& A) { B = (float)A.real(); return true; }
    bool convert(blas_real<float>& B, const complex<float>& A)       { B = (float)A.real(); return true; }
    bool convert(blas_real<float>& B, const complex<double>& A)      { B = (float)A.real(); return true; }
    bool convert(blas_complex<float>& B, const blas_real<float>& A)  { B = (float)A.c[0]; return true; }
    bool convert(blas_complex<double>& B, const blas_real<float>& A) { B = (double)A.c[0]; return true; }
    bool convert(complex<float>& B, const blas_real<float>& A)       { B = (float)A.c[0]; return true; }
    bool convert(complex<float>& B, const blas_real<double>& A)      { B = (float)A.c[0]; return true; }
    bool convert(blas_real<double>& B, const blas_real<float>& A)    { B = (double)A.c[0]; return true; }
    bool convert(blas_real<float>& B, const blas_real<double>& A)    { B = (float)A.c[0]; return true; }

    bool convert(blas_real<float>& B, const double& A){ B = (float)A; return true; }

    bool convert(complex<float>& B, const float& A){ B = (float)A; return true; }
    bool convert(complex<double>& B, const double& A){ B = (double)A; return true; }

    
    
    //////////////////////////////////////////////////////////////////////
    
    // for FFT . bits must be smaller than 8*sizeof(int)
    // (32bits nowadays - assuming 8 bit bytes)
    unsigned int bitreverse(unsigned int index,
			    unsigned int bits)
    {
      unsigned int result = 0;
      
      for(unsigned i=0;i<bits;i++)
      {
	result <<= 1;
	
	if(index & 1) result++;
	
	index >>= 1;
      }
      
      return result;
    }
    
  }
}






