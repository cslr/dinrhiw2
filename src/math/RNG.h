/*
 * RNG.h
 *
 *  Created on: 28.6.2015
 *      Author: Tomas
 */

#ifndef MATH_RNG_H_
#define MATH_RNG_H_

#include <exception>
#include <stdexcept>
#include "vertex.h"

namespace whiteice {

/**
 * Implements F.A.S.T. **thread-safe** hardware random number generator
 * using Intel RDRAND.
 */
template <typename T=math::blas_real<float> >
class RNG {
public:
	RNG() throw(std::runtime_error); // throws runtime error if RDRAND is not supported
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

	// functions to access assembly level instruction
	virtual unsigned int rdrand32() const;
	virtual unsigned long long rdrand64() const;
};

} /* namespace whiteice */


#include "RNG.cpp"


#endif /* MATH_RNG_H_ */

