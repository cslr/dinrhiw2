
// multidimensional gaussian with identity covariance matrix

#include "activation_function.h"
#include "vertex.h"

#ifndef multidimensional_gaussian_h
#define multidimensional_gaussian_h

namespace whiteice
{
  
  template <typename T>
    class multidimensional_gaussian : public activation_function< math::vertex<T> >
    {
    public:
      multidimensional_gaussian(const unsigned int size);
      
      multidimensional_gaussian(const multidimensional_gaussian<T>& g);
      
      // sets mean and dimensionality of function
      multidimensional_gaussian(const math::vertex<T>& mean);
      
      // calculates value of function
      math::vertex<T> operator() (const math::vertex<T>& x) const;
      
      // calculates value
      math::vertex<T> calculate(const math::vertex<T>& x) const;
      
      // calculates value
      void calculate(const math::vertex<T>& x, math::vertex<T>& y) const;
      
      // creates copy of object
      function<math::vertex<T>, math::vertex<T> >* clone() const;
      
      // calculates derivate of activation function
      math::vertex<T> derivate(const math::vertex<T>& x) const;
      
      bool has_max() const;  // has maximum
      bool has_min() const;  // has minimum
      bool has_zero() const; // has uniq. zero
      
      math::vertex<T> max() const;  // gives maximum value of activation function
      math::vertex<T> min() const;  // gives minimum value of activation function
      math::vertex<T> zero() const; // gives zero location of activation function
      
    private:
      
      math::vertex<T> mean;
      
    };
  
  
  extern template class multidimensional_gaussian< float >;
  extern template class multidimensional_gaussian< double >;
  extern template class multidimensional_gaussian< math::atlas_real<float> >;
  extern template class multidimensional_gaussian< math::atlas_real<double> >;
  
}



#endif
