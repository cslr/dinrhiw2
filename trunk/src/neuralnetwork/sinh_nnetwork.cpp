
#include "sinh_nnetwork.h"
#include "blade_math.h"

namespace whiteice
{
  
  template <typename T>
  sinh_nnetwork<T>::sinh_nnetwork() : nnetwork<T>()
  {
    
  }

  template <typename T>
  sinh_nnetwork<T>::sinh_nnetwork(const nnetwork<T>& nn) : nnetwork<T>(nn)
  {
    
  }

  template <typename T>
  sinh_nnetwork<T>::sinh_nnetwork(const std::vector<unsigned int>& nnarch) throw(std::invalid_argument) : nnetwork<T>(nnarch)
  {
    
  }
  
  template <typename T>
  sinh_nnetwork<T>::~sinh_nnetwork()
  {
    
  }
  
  template <typename T>
  inline T sinh_nnetwork<T>::nonlin(const T& input, unsigned int layer) const throw() // non-linearity used in neural network
  {
    if(layer & 1)
      return math::sinh(input);
    else
      return math::asinh(input);
  }
  
  template <typename T>
  inline T sinh_nnetwork<T>::Dnonlin(const T& input, unsigned int layer) const throw() // derivate of non-linearity used in neural network
  {
    if(layer & 1)
      return math::cosh(input);
    else
      return T(1.0f)/math::sqrt(input*input + T(1.0f));
  }
    
  

  
  template class sinh_nnetwork< float >;
  template class sinh_nnetwork< double >;  
  template class sinh_nnetwork< math::blas_real<float> >;
  template class sinh_nnetwork< math::blas_real<double> >;

};
