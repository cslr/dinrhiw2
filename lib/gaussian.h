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
      
      gaussian<T>* clone() const throw(std::bad_alloc); // creates copy of object
      
      T& mean() throw();
      const T& mean() const throw();
      
      T& variance() throw();
      const T& variance() const throw();
      
    private:
      
      T mean_val;
      T var_val;
      
    };
  
  
  
  extern template class gaussian<float>;
  extern template class gaussian<double>;
  extern template class gaussian< math::atlas_real<float> >;
  extern template class gaussian< math::atlas_real<double> >;
    
  
}



#endif
