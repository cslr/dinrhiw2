/* 
 * generic vertex
 * (math vector)
 *
 */

#ifndef gvertex_h
#define gvertex_h

#include "ownexception.h"
#include "number.h"

#include <stdexcept>
#include <exception>
#include <iostream>
#include <vector>


namespace whiteice
{
  namespace math
  {
    
    template <typename T, typename S>
      class gmatrix;

    // T is typename
    // S must be scalar which
    //   can operate with T
    
    
    template <typename T=float, typename S=T>
      class gvertex : public whiteice::number<gvertex<T,S>, S, T, unsigned int>
    {
      public:
      
      explicit gvertex(unsigned int i=4) throw();
      gvertex(const gvertex<T,S>& v) throw();
      gvertex(const std::vector<T>& v) throw();
      virtual ~gvertex() throw();
      
      typedef typename std::vector<T>::iterator iterator;
      typedef typename std::vector<T>::const_iterator const_iterator;
      
      unsigned int size() const throw();
      unsigned int resize(unsigned int d) throw();
      
      // TODO: move norm() to generic number class
      T norm() const throw();
      
      // calculates partial norm for gvertex(i:j)
      T norm(unsigned int i, unsigned int j) const throw();
      
      bool normalize() throw(); // length = 1

      gvertex<T,S> operator+(const gvertex<T,S>& v) const throw(illegal_operation);
      gvertex<T,S> operator-(const gvertex<T,S>& v) const throw(illegal_operation);

      // inner product - returns gvertex with size 1
      gvertex<T,S> operator*(const gvertex<T,S>& v) const throw(illegal_operation);
      
      gvertex<T,S> operator/(const gvertex<T,S>& v) const throw(illegal_operation);
      gvertex<T,S> operator!() const throw(illegal_operation);
      gvertex<T,S> operator-() const throw(illegal_operation);

      // cross product
      gvertex<T,S> operator^(const gvertex<T,S>& v) const throw(illegal_operation);
      
      gvertex<T,S>& operator+=(const gvertex<T,S>& v) throw(illegal_operation);
      gvertex<T,S>& operator-=(const gvertex<T,S>& v) throw(illegal_operation);
      gvertex<T,S>& operator*=(const gvertex<T,S>& v) throw(illegal_operation);
      gvertex<T,S>& operator/=(const gvertex<T,S>& v) throw(illegal_operation);
      
      gvertex<T,S>& operator=(const gvertex<T,S>& v) throw(illegal_operation);      
      
      bool operator==(const gvertex<T,S>& v) const throw(uncomparable);
      bool operator!=(const gvertex<T,S>& v) const throw(uncomparable);
      bool operator>=(const gvertex<T,S>& v) const throw(uncomparable);
      bool operator<=(const gvertex<T,S>& v) const throw(uncomparable);
      bool operator< (const gvertex<T,S>& v) const throw(uncomparable);
      bool operator> (const gvertex<T,S>& v) const throw(uncomparable);

      gvertex<T,S>& abs() throw();
      gvertex<T,S>& conj() throw();
      
      /* scalars */
      gvertex<T,S>& operator= (const S& s) throw(illegal_operation);
      gvertex<T,S>  operator* (const S& s) const throw();
      gvertex<T,S>  operator/ (const S& s) const throw(std::invalid_argument);
      gvertex<T,S>& operator*=(const S& s) throw();
      gvertex<T,S>& operator/=(const S& s) throw(std::invalid_argument);
      
      // multi from right
      template <typename TT, typename SS>
      friend gvertex<TT,SS> operator*(const SS& s, const gvertex<TT,SS>& v);
      
      // multiply from left
      gvertex<T,S>  operator* (const gmatrix<T,S>& m)
        const throw(std::invalid_argument);

      // outer product
      gmatrix<T,S> outerproduct(const gvertex<T,S>& v) const throw(std::domain_error);
      gmatrix<T,S> outerproduct(const gvertex<T,S>& v0,
			       const gvertex<T,S>& v1) const throw(std::domain_error);
      
      T& operator[](const unsigned int& index) throw(std::out_of_range, illegal_operation);
      const T& operator[](const unsigned int& index) const throw(std::out_of_range,
							  illegal_operation);
      
      iterator begin() throw(); // iterators
      iterator end() throw();
      const_iterator begin() const throw(); // iterators
      const_iterator end() const throw();
      
      bool comparable() throw(){ return false; }
      
      // TODO: add dummy    static const gvertex<T,S> null; for empty/null gvertex
      
      private:
      
      std::vector<T> c;
      
    };
    
    
    template <typename T, typename S>
      std::ostream& operator<<(std::ostream& ios,
			       const whiteice::math::gvertex<T,S>&);
    
    
    // tries to convert gvertex of type S to gvertex of type T (B = A)
    template <typename T, typename S>
      bool convert(gvertex<T>& B, const gvertex<S>& A) throw();
    
  }
}



#include "gmatrix.h"
#include "gvertex.cpp"

#endif
