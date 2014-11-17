/*
 * numerical ode function interface
 * y'(t) = f(t, y(t))
 */

#ifndef whiteice_odefunction_h
#define whiteice_odefunction_h

#include "function.h"
#include "vertex.h"

namespace whiteice
{
  namespace math
  {
    // this structure only exists to
    // make parameter passing to odefunction<T> easy
    // y' = f->call(odeparam(time, y));
    template <typename T=double>
      struct odeparam {
	public:
	
	odeparam(const T& _t, const whiteice::math::vertex<T>& _y) : 
	  t(_t), y(_y){ }
	
	virtual ~odeparam(){ }
	
	const T& t;
	const whiteice::math::vertex<T>& y;
      };
    
    
    /*
     * ODE function interface
     */
    template <typename T=double>
      class odefunction : public whiteice::function< odeparam<T>, whiteice::math::vertex<T> >
      {
	public:
	
	virtual ~odefunction(){ }
	
	// returns number of input dimensions
	virtual unsigned int dimensions() const PURE_FUNCTION = 0;
	
	// rest of the interface function definitions are in
	// whiteice::function interface
	
	private:
      };
    
  };
};


#endif


