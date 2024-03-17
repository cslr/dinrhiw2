/*
 * RBMvarianceerrorfunction.h
 *
 *  Created on: 22.6.2015
 *      Author: Tomas
 */

#ifndef NEURALNETWORK_RBMVARIANCEERRORFUNCTION_H_
#define NEURALNETWORK_RBMVARIANCEERRORFUNCTION_H_

#include "optimized_function.h"
#include "GBRBM.h"
#include "vertex.h"

#include <vector>

namespace whiteice {

/**
 * optimized function for PSO optimizer
 * takes in [0,1]^D valued variance values [0 = zero variance, 1 = total data variance]
 *
 */
template <typename T = math::blas_real<float> >
class GBRBM_variance_error_function: public optimized_function<T> {
public:
	GBRBM_variance_error_function(const std::vector< math::vertex<T> >& samples,
			unsigned int numHidden);
	virtual ~GBRBM_variance_error_function();

	// converts input into real variance values
	void getRealVariance(math::vertex<T>& var) const;

    // calculates value of function
    virtual T operator() (const math::vertex<T>& x) const PURE_FUNCTION;

    // calculates value
    virtual T calculate(const math::vertex<T>& x) const PURE_FUNCTION;

    // calculates value
    // (optimized version, this is faster because output value isn't copied)
    virtual void calculate(const math::vertex<T>& x, T& y) const;

    // creates copy of object
    virtual function< math::vertex<T>, T>* clone() const;

    // returns input vectors dimension
    unsigned int dimension() const  PURE_FUNCTION;


    bool hasGradient() const  PURE_FUNCTION;
    math::vertex<T> grad(math::vertex<T>& x) const PURE_FUNCTION;
    void grad(math::vertex<T>& x, math::vertex<T>& y) const;

private:
    const std::vector< math::vertex<T> >& samples;
    math::vertex<T> variance; // data variance

    unsigned int numHidden;
};


extern template class GBRBM_variance_error_function<float>;
extern template class GBRBM_variance_error_function<double>;
extern template class GBRBM_variance_error_function< math::blas_real<float> >;
extern template class GBRBM_variance_error_function< math::blas_real<double> >;


} /* namespace whiteice */

#endif /* NEURALNETWORK_RBMVARIANCEERRORFUNCTION_H_ */
