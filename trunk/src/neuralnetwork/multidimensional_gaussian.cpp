
#include "multidimensional_gaussian.h"
#include "blade_math.h"

#ifndef multidimensional_gaussian_cpp
#define multidimensional_gaussian_cpp

namespace whiteice
{
  
  template <typename T>
  multidimensional_gaussian<T>::multidimensional_gaussian(const unsigned int size)
  {
    mean.resize(size);
    
    for(unsigned int i=0;i<size;i++)
      mean[i] = T(0.0);
  }
  
  
  template <typename T>
  multidimensional_gaussian<T>::multidimensional_gaussian(const math::vertex<T>& m)
  {
    mean.resize(m.size());
    
    for(unsigned int i=0;i<m.size();i++)
      mean[i] = m[i];
  }
  
  template <typename T>
  multidimensional_gaussian<T>::multidimensional_gaussian(const multidimensional_gaussian<T>& g)
  {
    mean.resize(g.mean.size());
    
    for(unsigned int i=0;i<mean.size();i++)
      mean[i] = g.mean[i];
  }


  // calculates value of function
  template <typename T>
  math::vertex<T> multidimensional_gaussian<T>::operator() (const math::vertex<T>& x) const
  {
    return calculate(x);
  }
  
  
  // calculates value
  template <typename T>
  math::vertex<T> multidimensional_gaussian<T>::calculate(const math::vertex<T>& x) const
  {
    math::vertex<T> z;
    
    calculate(x, z);
    
    return z;
  }
  
  
  template <typename T>
  void multidimensional_gaussian<T>::calculate(const math::vertex<T>& x,
					       math::vertex<T>& z) const
  {
    z.resize(1);
    z[0] = T(0.0f);
    T delta;
    
    unsigned int i = 0;
    
    while(i < x.size()){
      delta = (x[i] - mean[i]);
      z[0] -= delta*delta;
      i++;
    }
    
    z[0] /= T(2.0f);
    
    z[0] = whiteice::math::exp(z[0]);
  }
  
  
  // creates copy of object
  template <typename T>
  function<math::vertex<T>, math::vertex<T> >* multidimensional_gaussian<T>::clone() const
  {
    return new multidimensional_gaussian<T>(*this);
  }
  
  
  // calculates derivate of activation function
  template <typename T>
  math::vertex<T> multidimensional_gaussian<T>::derivate(const math::vertex<T>& x) const
  {
    math::vertex<T> grad;
    grad.resize(mean.size());
    
    math::vertex<T> r;
    r.resize(1);
    r = calculate(x);
    
    for(unsigned int i=0;i<mean.size();i++){
      grad[i] = r[0] * T(-2.0) * x[i];
    }
    
    return grad;
  }
  
  
  template <typename T>
  bool multidimensional_gaussian<T>::has_max() const  // has maximum
  {
    return true;
  }
  
  
  template <typename T>
  bool multidimensional_gaussian<T>::has_min() const  // has minimum
  {
    return true;
  }
  
  
  template <typename T>
  bool multidimensional_gaussian<T>::has_zero() const // has uniq. zero
  {
    return false;
  }
  
  
  // gives maximum value of activation function
  template <typename T>
  math::vertex<T> multidimensional_gaussian<T>::max() const
  {
    math::vertex<T> max;
    max.resize(1);
    max[0] = T(1.0);
    
    return max;
  }
  
  
  // gives minimum value of activation function
  template <typename T>
  math::vertex<T> multidimensional_gaussian<T>::min() const
  {
    math::vertex<T> min;
    min.resize(1);
    min[0] = T(0.0);
    
    return min;
  }
  

  // gives zero location of activation function  
  template <typename T>
  math::vertex<T> multidimensional_gaussian<T>::zero() const
  {
    math::vertex<T> a;
    return a;
    
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class multidimensional_gaussian< float >;
  template class multidimensional_gaussian< double >;
  template class multidimensional_gaussian< math::atlas_real<float> >;
  template class multidimensional_gaussian< math::atlas_real<double> >;
  
}
  
#endif
