
#include "lreg_nnetwork.h"
#include "blade_math.h"

namespace whiteice
{
  
  template <typename T>
  lreg_nnetwork<T>::lreg_nnetwork() : nnetwork<T>()
  {
    
  }

  template <typename T>
  lreg_nnetwork<T>::lreg_nnetwork(const nnetwork<T>& nn) : nnetwork<T>(nn)
  {
    
  }

  template <typename T>
  lreg_nnetwork<T>::lreg_nnetwork(const std::vector<unsigned int>& nnarch) throw(std::invalid_argument) : nnetwork<T>(nnarch)
  {
    
  }
  
  template <typename T>
  lreg_nnetwork<T>::~lreg_nnetwork()
  {
    
  }
  
  template <typename T>
  inline T lreg_nnetwork<T>::nonlin(const T& input, unsigned int layer) const throw() // non-linearity used in neural network
  {
    T output = T(1.0) / (T(1.0) + math::exp(-input));
    return output;
  }
  
  template <typename T>
  inline T lreg_nnetwork<T>::Dnonlin(const T& input, unsigned int layer) const throw() // derivate of non-linearity used in neural network
  {
    T output = T(1.0) + math::exp(-input);
    
    output = math::exp(-input) / (output*output);
    
    return output;
  }
    
  

  
  template class lreg_nnetwork< float >;
  template class lreg_nnetwork< double >;  
  template class lreg_nnetwork< math::blas_real<float> >;
  template class lreg_nnetwork< math::blas_real<double> >;

};
