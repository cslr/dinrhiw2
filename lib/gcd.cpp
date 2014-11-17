
#ifndef gcd_cpp
#define gcd_cpp

#include "gcd.h"
#include <stdlib.h>
#include <algorithm>


namespace whiteice
{

  // calculates gcd between two numbers
  template <typename T> 
  T gcd(T a, T b) throw()
  {
    if(a < b) // after this one a >= b
      std::swap<T>(a,b);
    
    if(b <= 0)
      return T(0);
    
    T r = a;
    
    while(r){
      r = a % b;
      a = b;
      b = r;
    }
    
    return a;
  }
  
  
  template <typename T>
  void permutation(std::vector<T>& A) throw()
  {
    if(A.size() <= 1) return;
    
    const unsigned int N = A.size();
    unsigned int j;
    
    for(unsigned int i=0;i<(N - 1);i++){
      
      j = i + ( rand() % (N - i) );
      
      std::swap<T>(A[i], A[j]);
    }
    
  }
  
  
  template <typename T>
  void dqpermutation(std::vector<T>& A) throw()
  {
    if(A.size() <= 1) return;
    
    assert(0); // not done
  }
  
  
};



#endif

