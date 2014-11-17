/*
 * modular arithmetical numbers
 * implementation excepts that K is prime
 */

#ifndef modular_h
#define modular_h

#include "number.h"
#include "integer.h"
#include <iostream>


namespace whiteice
{
  namespace math
  {
    
    template <typename T=unsigned int>
      class modular : public number< modular<T>, T, T, unsigned int>
    {
      public:
      
      modular();
      modular(const modular<T>& a);	
      modular(const T& value, const T& modulo);
      modular(const T& modulo);
      virtual ~modular();
      
      // operators
      modular<T> operator+(const modular<T>&) const throw(illegal_operation);
      modular<T> operator-(const modular<T>&) const throw(illegal_operation);
      modular<T> operator*(const modular<T>&) const throw(illegal_operation);
      modular<T> operator/(const modular<T>&) const throw(illegal_operation);
      
      // complex conjugate (?)
      modular<T> operator!() const throw(illegal_operation);
      modular<T> operator-() const throw(illegal_operation);
      
      modular<T>& operator+=(const modular<T>&) throw(illegal_operation);
      modular<T>& operator-=(const modular<T>&) throw(illegal_operation);
      modular<T>& operator*=(const modular<T>&) throw(illegal_operation);
      modular<T>& operator/=(const modular<T>&) throw(illegal_operation);
      
      modular<T>& operator=(const modular<T>&) throw(illegal_operation);      
      
      bool operator==(const modular<T>&) const throw(uncomparable);
      bool operator!=(const modular<T>&) const throw(uncomparable);
      bool operator>=(const modular<T>&) const throw(uncomparable);
      bool operator<=(const modular<T>&) const throw(uncomparable);
      bool operator< (const modular<T>&) const throw(uncomparable);
      bool operator> (const modular<T>&) const throw(uncomparable);
      
      // scalar operation
      modular<T>& operator= (const T& s) throw(illegal_operation);
      modular<T>  operator* (const T& s) const throw();
      modular<T>  operator/ (const T& s) const throw(std::invalid_argument);
      modular<T>& operator*=(const T& s) throw();
      modular<T>& operator/=(const T& s) throw(std::invalid_argument);
      
      template <typename A>
      friend modular<A> operator*(const A& s, const modular<A>&) throw(std::invalid_argument);
      
      modular<T>& abs() throw();      
      
      T& operator[](const unsigned int& index) 
        throw(std::out_of_range, illegal_operation);
      
      const T& operator[](const unsigned int& index) const
        throw(std::out_of_range, illegal_operation);
      
      bool comparable() throw(){ return false; }
      
      private:
      
      T value;
      T modulo; 	
    };
    
    
    template <typename T>
      modular<T> operator*(const T& s, const modular<T>&) throw(std::invalid_argument);
    
    template <typename T>
      std::ostream& operator<<(std::ostream& ios,
			       const whiteice::math::modular<T>&);
    
    
    
    
    extern template class modular<unsigned int>;
    extern template class modular<unsigned short>;
    extern template class modular<unsigned char>;
    extern template class modular<integer>;
    
    
    extern template modular<unsigned int> operator*<unsigned int>
      (const unsigned int& s, const modular<unsigned int>&) throw(std::invalid_argument);
    
    extern template modular<unsigned short> operator*<unsigned short>
      (const unsigned short& s, const modular<unsigned short>&) throw(std::invalid_argument);
    
    extern template modular<unsigned char> operator*<unsigned char>
      (const unsigned char& s, const modular<unsigned char>&) throw(std::invalid_argument);
    
    extern template modular<integer> operator*<integer>
      (const integer& s, const modular<integer>&) throw(std::invalid_argument);
    
    
    extern template std::ostream& operator<< <unsigned int>(std::ostream& ios, const whiteice::math::modular<unsigned int>&);
    extern template std::ostream& operator<< <unsigned short>(std::ostream& ios, const whiteice::math::modular<unsigned short>&);
    extern template std::ostream& operator<< <unsigned char>(std::ostream& ios, const whiteice::math::modular<unsigned char>&);
    extern template std::ostream& operator<< <integer>(std::ostream& ios, const whiteice::math::modular<integer>&);
    
  }
}



#endif
