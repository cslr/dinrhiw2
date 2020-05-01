/*
 * gaussian function
 * with mean 
 */

#include "function.h"
#include <new>
#include <exception>


#ifndef gaussian_h
#define gaussian_h


namespace whiteice
{

  template <typename T>
    class gaussian : public function<T,T>
    {
    public:
      gaussian(T mean = 0, T var = 1);
      
      T operator() (const T& x) const;  // calculates value of function
      T calculate(const T& x) const;    // calculates value
      
      gaussian<T>* clone() const ; // creates copy of object
      
      T& mean() ;
      const T& mean() const ;
      
      T& variance() ;
      const T& variance() const ;
      
    private:
      
      T mean_val;
      T var_val;
      
    };
  
  
  
  extern template class gaussian<float>;
  extern template class gaussian<double>;
  extern template class gaussian< math::blas_real<float> >;
  extern template class gaussian< math::blas_real<double> >;
    
  
}



#endif
