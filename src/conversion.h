/*
 * functions (and their inverses) for isomorphisms between 
 * floating points formats and unsigned integer formats
 * with respect to "<" operator
 * 
 */

#ifndef conversion_h
#define conversion_h

namespace whiteice
{
  // (note: c++ compiler specific, for g++)

  // maps IEEE 754 single precision floating point number to
  // unsigned 32-bit integer
  unsigned int ieee754spf2uint32(const float& f);
  
  float uint322ieee754spf(const unsigned int& i); // inverse
  
  
  // maps IEEE 754 double precision floating point to
  // unsigned 64-bit integer
  unsigned long long ieee754dpf2uint64(const double& f);
   
  double uint642ieee754dpf(const unsigned long long& i); // inverse

}

#endif
