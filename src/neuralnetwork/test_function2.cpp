
#include <new>
#include "test_function2.h"

namespace whiteice
{

  double test_function2::operator() (const int& x) const
  {
    return calculate(x);;
  }
  
  double test_function2::calculate(const int& x) const
  {
    double xf = (double)(x / 1000.0f); // representation in "x,10^3 format"
    
    return (xf*xf*xf*xf + xf*xf - 2 * xf + 10.0f); // min at x = 1, value 9.0
  }
  
  
  void test_function2::calculate(const int& x, double& y) const
  {
    y = calculate(x);
  }
  
  function<int,double>* test_function2::clone() const
  {
    return new test_function2;
  }

  
  //////////////////////////////////////////////////////////////////////
  

  float test_function2b::operator() (const dynamic_bitset& x) const
  {
    return calculate(x);
  }
  
  
  float test_function2b::calculate(const dynamic_bitset& x) const
  {
    float sum = 0.0f;
    
    for(unsigned i=0;i<x.size();i++)
      if(x[i] && ((i & 1) == 1))
	sum += 1.0f;
    
    return sum;
  }
  
  
  void test_function2b::calculate(const dynamic_bitset& x, float& y) const{
    y = calculate(x);
  }
  
  function<dynamic_bitset, float>* test_function2b::clone() const
  {
    return new test_function2b;
  }
  
}
