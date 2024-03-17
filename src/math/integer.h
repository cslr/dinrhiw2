/*
 * c++ abtritrary precision integer class
 * uses GMP library
 */

#ifndef __whiteice_gmp_integer_h
#define __whiteice_gmp_integer_h

#include "number.h"
#include <string>
#include <gmp.h>

namespace whiteice
{
  namespace math
  {
    
    class integer
    {
    public:
      
      integer();
      integer(const integer& i);
      integer(const long int& t);
      
      /* converts from a base-radix integer string number presentation to a number */
      explicit integer(const std::string& s, unsigned int base = 10);
      
      virtual ~integer();
      
      // operators
      integer operator+(const integer&) const ;
      integer operator-(const integer&) const ;
      integer operator*(const integer&) const ;
      integer operator/(const integer&) const ;
      
      integer operator%(const integer&) const ;
                  
      integer operator-() const ;
      
      integer& operator+=(const integer&) ;
      integer& operator-=(const integer&) ;
      integer& operator*=(const integer&) ;
      integer& operator/=(const integer&) ;
      
      integer& operator%=(const integer&) ;
      
      integer& operator=(const integer&) ;
      
      integer& operator++() ;
      integer& operator--() ;

      integer& operator++(int) ;
      integer& operator--(int) ;
      
      // bitwise operators
      integer operator!() const ; // one's complement
      
      integer& operator&=(const integer&) ;
      integer& operator|=(const integer&) ;
      integer& operator^=(const integer&) ;
      
      integer operator&(const integer&) ;
      integer operator|(const integer&) ;
      integer operator^(const integer&) ;
      
      bool getbit(unsigned int index) const ;
      void setbit(unsigned int index, bool value=true) ;
      void clrbit(unsigned int index) ;
      
      integer operator<<(unsigned int left) const ;
      integer operator>>(unsigned int right) const ;
      
      integer& operator<<=(unsigned int left) ;
      integer& operator>>=(unsigned int right) ;
      
      // left is the positive direction
      integer& circularshift(int shift) ;
      
      
      bool operator==(const integer&) const ;
      bool operator!=(const integer&) const ;
      bool operator>=(const integer&) const ;
      bool operator<=(const integer&) const ;
      bool operator< (const integer&) const ;
      bool operator> (const integer&) const ;
      
      // scalar operation
      integer& operator= (const int& s) ;
      integer  operator* (const int& s) const ;
      integer  operator/ (const int& s) const ;
      integer& operator*=(const int& s) ;
      integer& operator/=(const int& s) ;
      
      long int to_int() const ;
      std::string to_string(unsigned int base = 10) const ;
      
      integer& abs() ;
      
      // returns true if number is zero or positive
      bool positive() const ;
      
      unsigned long int bits() const ; // number of bits used by number
      
      
      // friend functions
      
      friend void modular_exponentation(integer& x, const integer& e, const integer& n);
      friend void gcd(integer& res, const integer& x, const integer& y);      
      friend bool probably_prime(const integer& x);
      friend void modular_inverse(integer& inv, const integer a, const integer& n);
      
    private:
      
      mpz_t integ;
      
    };
    
    
    void modular_exponentation(integer& x, const integer& e, const integer& n);
    void gcd(integer& res, const integer& x, const integer& y);
    void modular_inverse(integer& inv, const integer a, const integer& n);
    
    bool probably_prime(const integer& x);
    
    integer factorial(integer a);
    integer combinations(integer a, integer b);
    
    std::ostream& operator<<(std::ostream& ios,
			     const integer& i);
    
  };
};




#endif
