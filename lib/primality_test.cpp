/*
 * simple primality testing
 * 
 *  - don't know about theory
 */

#ifndef primality_test_cpp
#define primality_test_cpp

#include "primality_test.h"


namespace whiteice
{
  template <typename T>
  bool nth_bit(T b, T n);
  
  template <typename T>
  T modular_exponentation(T a, T b, T n);
  
  
  template <typename T>
  bool pseudoprime(const T n)
  {
    if(modular_exponentation<T>(2, n-1, n) != 1)
      return false;
    else
      return true;
  }
  
  
  template <typename T>
  bool nth_bit(T b, T n)
  {
    while(n > 0){
      b = b >> 1;
      n--;
    }
    
    return ((bool)(b & 1));
  }
  
  
  template <typename T>
  T modular_exponentation(T a, T b, T n)
  {
    T c = 0;
    T d = 1;
    
    for(int i = (sizeof(T)*8) - 1; i >= 0; i--){	
      c = 2*c;
      d = (d * d) % n;
      
      if(nth_bit(b, i)){
	c = c + 1;
	d = (d * a) % n;
      }
    }
    
    return d;
  }
  
    
  /************************************************************/
  
  /* factorizes number with trial division
   * up to some limit and returns that limit.
   * n is remaining non-factored part of the original
   * numbers and factors are added to f
   */
  template <typename T>
  T trial_division(T& n, std::map<T,T>& f);
  
  
  template <typename T>
  bool factorize(T n, std::map<T,T>& f) throw()
  {
    // with very small numbers ( < 10**9 ) trial
    // division is reasonable fast
    
    if(n <= T(0)) return false;
    if(n == T(1)){
      f[T(1)] = 1;
      return false;
    }
    
    T trial_limit = trial_division(n, f);
    
    if(n <= trial_limit*trial_limit)
      return true;
    
    // if there's work left switch to more advanced
    // methods
    
    // implement me
    
    return false;
  }
  
  
  
  template <typename T>
  T trial_division(T& n, std::map<T,T>& f)
  {
    const T LIMIT = T(10000);
    T t = T(2);
    
    while(t <= LIMIT && n <= t){
      while((n % t) == 0){
	
	n /= t;
	f[t]++;
      }
      
      t++;
    }
    
    return t;
  }
  
  
}

#endif

