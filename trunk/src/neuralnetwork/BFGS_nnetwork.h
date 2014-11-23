/*
 * BFGS optimizer for neural networks
 * 
 *
 */

#ifndef BFGS_nnetwork_h
#define BFGS_nnetwork_h

#include "BFGS.h"
#include "nnetwork.h"
#include "dataset.h"
#include "vertex.h"

namespace whiteice
{
  template <typename T=math::blas_real<float> >
    class BFGS_nnetwork : public whiteice::math::BFGS<T>
    {
    public:
      BFGS_nnetwork(const nnetwork<T>& net,
		    const dataset<T>& d);
    
      virtual ~BFGS_nnetwork();
    
    protected:

      // optimized function
      virtual T U(const math::vertex<T>& x) const;
      virtual math::vertex<T> Ugrad(const math::vertex<T>& x) const;

    public:
    
      // calculates the current solution's "real" error
      // (we keep iterating until U(x) converges or getError()
      //  increases instead of decreasing)
      T getError(const math::vertex<T>& x) const;

    private:
      const nnetwork<T> net;
      const dataset<T>& data;
    
      dataset<T> dtrain;
      dataset<T> dtest;
      
    };


  extern template class BFGS_nnetwork< float >;
  extern template class BFGS_nnetwork< double >;
  extern template class BFGS_nnetwork< math::blas_real<float> >;
  extern template class BFGS_nnetwork< math::blas_real<double> >;
  
}


#endif
