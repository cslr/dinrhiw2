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

namespace whiteice
{
  template <typename T=math::blas_real<float> >
    class BFGS_nnetwork : public whiteice::math::BFGS<T>
    {
    public:
      BFGS_nnetwork(const nnetwork<T>& net,
		    const dataset<T>& d);
    
      ~BFGS_nnetwork();
    
      // calculates the current solution's best error
      T getError() const;

    protected:

      virtual T U(const math::vertex<T>& x) const;
      virtual math::vertex<T> Ugrad(const math::vertex<T>& x) const;

      const nnetwork<T> net;
      const dataset<T>& data;
      
    };


  extern template class BFGS_nnetwork< float >;
  extern template class BFGS_nnetwork< double >;
  extern template class BFGS_nnetwork< math::blas_real<float> >;
  extern template class BFGS_nnetwork< math::blas_real<double> >;
  
}


#endif
