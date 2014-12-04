/*
 * negative function implementation for
 * optimized_function:s
 */

#ifndef negative_function_cpp
#define negative_function_cpp

#include <new>
#include "negative_function.h"

/**************************************************/

namespace whiteice
{
  
  // clones local copy
  template <typename T>
  negative_function<T>::negative_function(const optimized_function<T>& f)
  {
    this->f = dynamic_cast<optimized_function<T>*>(f.clone());
  }
  
  
  // uses ptr and deletes it later
  template <typename T>
  negative_function<T>::negative_function(const optimized_function<T>* f)
  {
    this->f = dynamic_cast<optimized_function<T>*>(f->clone());
  }
  
  
  template <typename T>
  negative_function<T>::~negative_function(){
    if(f) delete f;
  }
  
  
  // calculates value of function
  template <typename T>
  T negative_function<T>::operator() (const math::vertex<T>& x) const
  {
    return calculate(x);
  }

  
  // calculates value
  template <typename T>
  T negative_function<T>::calculate(const math::vertex<T>& x) const
  {
    return -(f->calculate(x));
  }
  
  
  template <typename T>
  void negative_function<T>::calculate(const math::vertex<T>& x, T& y) const {
    y = -(f->calculate(x));
  }
  
  
  // creates copy of object  
  template <typename T>
  function<math::vertex<T>,T>* negative_function<T>::clone() const
  {
    return new negative_function<T>(*f);
  }
  
  
  template <typename T>
  unsigned int negative_function<T>::dimension() const throw()
  {
    return f->dimension();
  }
  
  
  
  template <typename T>
  bool negative_function<T>::hasGradient() const throw(){
    return f->hasGradient();
  }
  
  template <typename T>
  math::vertex<T> negative_function<T>::grad(math::vertex<T>& x) const {
    return -(f->grad(x));
  }
  
  template <typename T>
  void negative_function<T>::grad(math::vertex<T>& x, math::vertex<T>& y) const {
    f->grad(x, y);
    y = -y;
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class negative_function< float >;
  template class negative_function< double >;  
  template class negative_function< math::blas_real<float> >;
  template class negative_function< math::blas_real<double> >;
  
}

#endif
