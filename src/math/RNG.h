/*
 * RNG.h
 *
 *  Created on: 28.6.2015
 *      Author: Tomas Ukkonen
 */

#ifndef MATH_RNG_H_
#define MATH_RNG_H_

#include <exception>
#include <stdexcept>
#include "vertex.h"

namespace whiteice {

  /**
   * Implements **thread-safe** hardware random number generator
   * using Intel RDRAND if it is available. Otherwise falls back to rand()
   * which is NOT thread-safe.
   *
   * NOTE: It seems that software C++ RNG is actually faster in generating
   *       normally distributed variables than this when using software RNG.
   *       => Currently only use this when thread-safety is an issue.
   *       => Study C++ normal distribution random number generation in detail.
   */
  template <typename T=math::blas_real<float> >
    class RNG {
  public:
  
  RNG(); // uses regular rand() if rdrand is not supported
  virtual ~RNG(){ }
  
  unsigned int rand() const; // 32bit
  unsigned long long rand64() const; // 64bit
  
  T uniform() const; // [0,1]
  void uniform(math::vertex<T>& u) const;
  
  T normal() const; // N(0,1)
  void normal(math::vertex<T>& n) const;
  
  T exp() const; // Exp(lambda=2) [not lambda != 1]
  void exp(math::vertex<T>& e) const;
  
  protected:
  
  // ziggurat method lookup tables (read-only)
  unsigned int kn[128], ke[256];
  float wn[128], fn[128], we[256], fe[256];
  
  float rnor() const;
  float rexp() const;
  void calculate_ziggurat_tables();
  
  double unid() const;
  float unif() const; // floating point uniform distribution [for ziggurat method]

  // function pointers to generate random numbers (initialized appropriately by ctor)
  unsigned int (RNG<T>::*rdrand32)() const = &whiteice::RNG<T>::_rand32;
  unsigned long long (RNG<T>::*rdrand64)() const = &whiteice::RNG<T>::_rand64;
  
  // functions to access assembly level instructionxs
  virtual unsigned int _rdrand32() const;
  virtual unsigned long long _rdrand64() const;
  
  // rand() using instructions to use if RDRAND is not available
  virtual unsigned int _rand32() const;
  virtual unsigned long long _rand64() const;
  
  void cpuid(unsigned int leaf, unsigned int subleaf, unsigned int regs[4]);
  };
  
} /* namespace whiteice */


#include "RNG.cpp"


#endif /* MATH_RNG_H_ */

