

#ifndef primality_test_h
#define primality_test_h

#include <map>


namespace whiteice
{
  
  template <typename T>
    bool pseudoprime(const T n);
  
  /* factorizes number into prime factors,
   * pairs in the map are form (prime, number of times)
   */
  template <typename T>
    bool factorize(T n, std::map<T,T>& f) throw(); 
  
}

#include "primality_test.cpp"


#endif

