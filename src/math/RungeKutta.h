/*
 * calculates standard 4th order 
 * Runge-Kutta (RK4) integration
 * with adaptive step length
 * 
 * TODO: later make this start computation
 * to the own thread + ETA updates + ability to
 * access partial results + ability stop/continue computations
 * if needed
 */

#ifndef RungeKutta_h
#define RungeKutta_h

#include "odefunction.h"
#include "vertex.h"
#include <vector>

namespace whiteice
{
  namespace math
  {
    template <typename T>
      class RungeKutta
      {
      public:
	RungeKutta(odefunction<T>* f = 0);
	~RungeKutta();
	
	odefunction<T>* getFunction() const throw();
	void setFunction(odefunction<T>* f) throw();
	
	// calculates values from the starting point y0
	// with adaptive step length, adds results to the end of vector
	// (h_new = h_old (e0/e)^(1/5)), initial h0 = 10e-4, e0=10e-8
	// errors are absolute
	void calculate(const T t0, const T t_end,
		       const whiteice::math::vertex<T>& y0,
		       std::vector< whiteice::math::vertex<T> >& points,
		       std::vector< T >& times);
	
      private:
	odefunction<T>* f;
	
      };
    
    
    //////////////////////////////////////////////////////////////////////
    
    extern template class RungeKutta< float >;
    extern template class RungeKutta< double >;
    extern template class RungeKutta< blas_real<float> >;
    extern template class RungeKutta< blas_real<double> >;
    //extern template class RungeKutta< blas_complex<float> >;
    //extern template class RungeKutta< blas_complex<double> >;
    
  };
};



#endif

