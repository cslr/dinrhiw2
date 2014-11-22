/*
 * quaternion class
 */
 
#ifndef quaternion_h
#define quaternion_h

#include "dinrhiw_blas.h"
#include "number.h"
#include <iostream>
#include <exception>
#include <stdexcept>
#include <memory>


namespace whiteice
{
  namespace math
  {

    template <typename T>
      class vertex;
    
    template <typename T>
      class matrix;
    
    
    template <typename T = blas_real<float> >
      class quaternion : public whiteice::number<quaternion<T>, T, T, unsigned int>
    {
      public:
      
      quaternion() throw();
      
      quaternion(const quaternion<T>&) throw();
      quaternion(const T& s) throw();
      virtual ~quaternion() throw();
      
      quaternion<T> operator+(const quaternion<T>&) const throw(illegal_operation);
      quaternion<T> operator-(const quaternion<T>&) const throw(illegal_operation);
      quaternion<T> operator*(const quaternion<T>&) const throw(illegal_operation);
      quaternion<T> operator/(const quaternion<T>&) const throw(illegal_operation);
      
      quaternion<T> operator!() const throw(illegal_operation);
      quaternion<T> operator-() const throw(illegal_operation);
      
      quaternion<T>& operator+=(const quaternion<T>&) throw(illegal_operation);
      quaternion<T>& operator-=(const quaternion<T>&) throw(illegal_operation);
      quaternion<T>& operator*=(const quaternion<T>&) throw(illegal_operation);
      quaternion<T>& operator/=(const quaternion<T>&) throw(illegal_operation);
      
      quaternion<T>& operator=(const quaternion<T>&) throw(illegal_operation);
      
      bool operator==(const quaternion<T>&) const throw(uncomparable);
      bool operator!=(const quaternion<T>&) const throw(uncomparable);
      bool operator>=(const quaternion<T>&) const throw(uncomparable);
      bool operator<=(const quaternion<T>&) const throw(uncomparable);
      bool operator< (const quaternion<T>&) const throw(uncomparable);
      bool operator> (const quaternion<T>&) const throw(uncomparable);
      
      /* interaction with 'primal' type */
      quaternion<T>& operator= (const T&) throw(illegal_operation);
      quaternion<T>& operator*=(const T&) throw();
      quaternion<T>& operator/=(const T&) throw(std::invalid_argument);
      quaternion<T>  operator* (const T&) const throw();
      quaternion<T>  operator/ (const T&) const throw(std::invalid_argument);
      
      quaternion<T>& operator= (const vertex<T>&) throw(std::invalid_argument);
      
      T& operator[](const unsigned int& index) throw(std::out_of_range, illegal_operation);
      const T& operator[](const unsigned int& index) const throw(std::out_of_range, illegal_operation);
      
      
      /* creates rotation matrix of quaternion */
      matrix<T> rotation_matrix() const throw();
      
      /* calculates quaternion rotator from matrix */
      bool create_quaternion_rotator(const matrix<T>& m) const throw();
      
      // setups rotation quaternion
      bool setup_rotation(T& alpha, const quaternion<T>& axis) throw(std::invalid_argument);
      
      // inverse quaternion
      quaternion<T> inv() const throw(std::invalid_argument);
      
      quaternion<T>& abs() throw();
      bool normalize() throw();
      
      bool comparable() throw(){ return false; }
      
      private:
      
      T* data; // for now..
      
    };
    
    
    template <typename T>
      std::ostream& operator<<(std::ostream& ios, const quaternion<T>& q);
    

    
    extern template class quaternion<float>;
    extern template class quaternion<double>;
    extern template class quaternion< blas_real<float> >;
    extern template class quaternion< blas_real<double> >;
    
    extern template std::ostream& operator<< <float>(std::ostream& ios, const quaternion<float>& q);
    extern template std::ostream& operator<< <double>(std::ostream& ios, const quaternion<double>& q);
    extern template std::ostream& operator<< <blas_real<float> >(std::ostream& ios, const quaternion<blas_real<float> >& q);
    extern template std::ostream& operator<< <blas_real<double> >(std::ostream& ios, const quaternion<blas_real<double> >& q);
    
  }
}

  
#include "matrix.h"
#include "vertex.h"
// #include "quaternion.cpp"


#endif

