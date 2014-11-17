/*
 * todo: fix name to correct one (not)
 * (target of optimization)
 */
#ifndef test_function2_h
#define test_function2_h

#include "function.h"
#include "dynamic_bitset.h"

namespace whiteice
{

  class test_function2 : public function<int, double>
  {
  public:  
    double operator() (const int& x) const;
    
    double calculate(const int& x) const;
    
    void calculate(const int& x, double& y) const;
    
    function<int,double>* clone() const;  
  };


  class test_function2b : public function<dynamic_bitset, float>
  {
  public:  
    float operator() (const dynamic_bitset& x) const;
    
    float calculate(const dynamic_bitset& x) const;
    
    void calculate(const dynamic_bitset& x, float& y) const;
    
    function<dynamic_bitset, float>* clone() const;  
  };
  
};
  
#endif

