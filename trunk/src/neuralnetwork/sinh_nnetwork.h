/*
 * neural network using sinh(x) non-linearity
 */

#ifndef __sinh_nnetwork_h
#define __sinh_nnetwork_h

#include "nnetwork.h"

namespace whiteice
{
  
  template < typename T = math::blas_real<float> >
    class sinh_nnetwork : public nnetwork<T>
    {
    public:
    
    sinh_nnetwork(); 
    sinh_nnetwork(const nnetwork<T>& nn);
    sinh_nnetwork(const std::vector<unsigned int>& nnarch) throw(std::invalid_argument);
    
    virtual ~sinh_nnetwork();
    
    protected:
    
    inline T nonlin(const T& input, unsigned int layer) const throw(); // non-linearity used in neural network
    inline T Dnonlin(const T& input, unsigned int layer) const throw(); // derivate of non-linearity used in neural network
    
    };

  
  extern template class sinh_nnetwork< float >;
  extern template class sinh_nnetwork< double >;  
  extern template class sinh_nnetwork< math::blas_real<float> >;
  extern template class sinh_nnetwork< math::blas_real<double> >;

};

#endif
