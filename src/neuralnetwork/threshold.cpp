
#include "threshold.h"
#include <iostream>



namespace whiteice
{
  template class threshold<float>;
  template class threshold<double>;
  template class threshold< math::atlas_real<float> >;
  template class threshold< math::atlas_real<double> >;
  
  
  
  template <typename T>
  threshold<T>::threshold(){}
  
  
  template <typename T>
  virtual threshold<T>::~threshold(){}
  
  
  template <typename T>
  T threshold<T>::calculate(const T& x)    const  // calculates value
  {
    if(x < 0) return -1;
    else return 1;
  }
  
  
  template <typename T>
  T threshold<T>::operator() (const T& x)  const  // calculates value of activation function
  {
    return calculate(x);
  }
  
  
  template <typename T>
  T threshold<T>::derivate(const T& x)  const  // calculates derivate of activation function
  {
    // threshold function don't have derivate, try to approximate one by defining derivate to
    // be 1 at zero, maybe useful value in practice.
    
    if(x != 0) return 0;
    else{
      // temporarily hack
    std::cout << "WARNING: THRESHOLD DOESN'T HAVE DERIVATE AT ZERO" << endl;
    return 1;              // real value would be infinite but...
    }
  }

  
  template <typename T>
  bool threshold<T>::has_max() const   // has maximum
  {
    return true;
  }
  
  
  template <typename T>
  bool threshold<T>::has_min() const   // has minimum
  {
    return true;
  }
  
  
  template <typename T>
  bool threshold<T>::has_zero() const  // has unique zero
  {
    return true;
  }
  
  
  template <typename T>
  T threshold<T>::max() const   // gives maximum value of activation function
  {
    return 1;
  }
  
  
  template <typename T>
  T threshold<T>::min() const   // gives minimum value of activation function
  {
    return -1;
  }
  
  
  template <typename T>
  T threshold<T>::zero() const  // gives zero location of activation function
  {
    return 0;
  }
  
  
  
  template <typename T>
  function<T>* threshold<T>::clone() const  // creates copy of object
  {
    return static_cast< function<T>* >(new threshold<T>);
  }
  
}



