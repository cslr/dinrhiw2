/*
 * gcd (and maybe other)
 * number algorithms
 */
#ifndef gcd_h
#define gcd_h

#include <vector>
#include "atlas.h"
#include "blade_math.h"
#include "integer.h"

namespace whiteice
{

  /* calculates 
   * gcd of two numbers
   * with euclid algorithm
   */
  template <typename T> 
    T gcd(T a, T b) throw();
  
  /*
   * (trival algorithm)
   * calculates random
   * permutation this
   * is O(n) algorithm but
   * does random memory accesses
   * (causes relatively much swapping)
   */  
  template <typename T>
    void permutation(std::vector<T>& A) throw();
  

  /*
   * (own algorithm - 'divide and conquer',
   *  dq-permutation)
   * calculates random
   * permutation this
   * is O(n*log n) algorithm but
   * it memory accesses mostly
   * locally (-> probably faster)
   * with reasonable problem
   * sizes (n is so big that
   * swapping would happen often even
   * when this algorithm is used)
   */  
  template <typename T>
    void dqpermutation(std::vector<T>& A) throw();
  
  
};

#include "gcd.cpp"  
  
#endif
