/*
 * neural network using logistic regression non-linearity
 * f(x) = 1 / (1 + exp(-x))
 */

#ifndef __lreg_nnetwork_h
#define __lreg_nnetwork_h

#include "nnetwork.h"

namespace whiteice
{
  
  template < typename T = math::blas_real<float> >
    class lreg_nnetwork : public nnetwork<T>
    {
    public:
    
    lreg_nnetwork(); 
    lreg_nnetwork(const nnetwork<T>& nn);
    lreg_nnetwork(const std::vector<unsigned int>& nnarch) ;
    
    virtual ~lreg_nnetwork();
    
    protected:
    
    inline T nonlin(const T& input, unsigned int layer) const ; // non-linearity used in neural network
    inline T Dnonlin(const T& input, unsigned int layer) const ; // derivate of non-linearity used in neural network
    
    };

  
  extern template class lreg_nnetwork< float >;
  extern template class lreg_nnetwork< double >;  
  extern template class lreg_nnetwork< math::blas_real<float> >;
  extern template class lreg_nnetwork< math::blas_real<double> >;

};

#endif
