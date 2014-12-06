/*
 * L-BFGS optimizer for neural networks
 * 
 *
 */

#ifndef LBFGS_nnetwork_h
#define LBFGS_nnetwork_h

#include "LBFGS.h"
#include "nnetwork.h"
#include "dataset.h"
#include "vertex.h"

namespace whiteice
{
  template <typename T=math::blas_real<float> >
    class LBFGS_nnetwork : public whiteice::math::LBFGS<T>
    {
    public:
      LBFGS_nnetwork(const nnetwork<T>& net,
		     const dataset<T>& d, bool overfit=false, bool negativefeedback=false);
    
      virtual ~LBFGS_nnetwork();
    
    protected:

      // optimized function
      virtual T U(const math::vertex<T>& x) const;
      virtual math::vertex<T> Ugrad(const math::vertex<T>& x) const;
    
      virtual bool heuristics(math::vertex<T>& x) const;

    public:
    
      // calculates the current solution's "real" error
      // (we keep iterating until U(x) converges or getError()
      //  increases instead of decreasing)
      T getError(const math::vertex<T>& x) const;

    private:
      const nnetwork<T> net;
      const dataset<T>& data;
    
      bool negativefeedback;
    
      dataset<T> dtrain;
      dataset<T> dtest;
      
    };


  extern template class LBFGS_nnetwork< float >;
  extern template class LBFGS_nnetwork< double >;
  extern template class LBFGS_nnetwork< math::blas_real<float> >;
  extern template class LBFGS_nnetwork< math::blas_real<double> >;
  
}


#endif
