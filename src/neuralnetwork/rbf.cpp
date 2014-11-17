/*
 * radial basis function implementation
 * (not fully implemented .. yet)
 */
 
#ifndef rbf_cpp
#define rbf_cpp

#include "multidimensional_gaussian.h"

namespace whiteice
{
  
#if 0
  template <typename T>
  RBF<T>::RBF(unsigned int size)
  {
    F = new multidimensional_gaussian<T>(size);
  }
  
  
  template <typename T>
  RBF<T>::RBF(const activation_function<T>& F)
  {
    this->F = F.clone();  
  }
  
  
  template <typename T>
  RBF<T>::~RBF()
  {
    
  }
  
  template <typename T>
  bool RBF<T>::set_activation(const activation_function<T>& F){ return false; }
  
  template <typename T>
  activation_function<T> RBF<T>::get_actication(){ return this->F; }
#endif
}
  
#endif
