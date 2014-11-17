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
      integer operator+(const integer&) const throw(illegal_operation);
      integer operator-(const integer&) const throw(illegal_operation);
      integer operator*(const integer&) const throw(illegal_operation);
      integer operator/(const integer&) const throw(illegal_operation);
      
      integer operator%(const integer&) const throw(illegal_operation);
                  
      integer operator-() const throw(illegal_operation);
      
      integer& operator+=(const integer&) throw(illegal_operation);
      integer& operator-=(const integer&) throw(illegal_operation);
      integer& operator*=(const integer&) throw(illegal_operation);
      integer& operator/=(const integer&) throw(illegal_operation);
      
      integer& operator%=(const integer&) throw(illegal_operation);
      
      integer& operator=(const integer&) throw(illegal_operation);
      
      integer& operator++() throw(illegal_operation);
      integer& operator--() throw(illegal_operation);

      integer& operator++(int) throw(illegal_operation);
      integer& operator--(int) throw(illegal_operation);
      
      // bitwise operators
      integer operator!() const throw(illegal_operation); // one's complement
      
      integer& operator&=(const integer&) throw(illegal_operation);
      integer& operator|=(const integer&) throw(illegal_operation);
      integer& operator^=(const integer&) throw(illegal_operation);
      
      integer operator&(const integer&) throw(illegal_operation);
      integer operator|(const integer&) throw(illegal_operation);
      integer operator^(const integer&) throw(illegal_operation);
      
      bool getbit(unsigned int index) const throw();
      void setbit(unsigned int index, bool value=true) throw();
      void clrbit(unsigned int index) throw();
      
      integer operator<<(unsigned int left) const throw();
      integer operator>>(unsigned int right) const throw();
      
      integer& operator<<=(unsigned int left) throw();
      integer& operator>>=(unsigned int right) throw();
      
      // left is the positive direction
      integer& circularshift(int shift) throw();
      
      
      bool operator==(const integer&) const throw(uncomparable);
      bool operator!=(const integer&) const throw(uncomparable);
      bool operator>=(const integer&) const throw(uncomparable);
      bool operator<=(const integer&) const throw(uncomparable);
      bool operator< (const integer&) const throw(uncomparable);
      bool operator> (const integer&) const throw(uncomparable);
      
      // scalar operation
      integer& operator= (const int& s) throw(illegal_operation);
      integer  operator* (const int& s) const throw();
      integer  operator/ (const int& s) const throw(std::invalid_argument);
      integer& operator*=(const int& s) throw();
      integer& operator/=(const int& s) throw(std::invalid_argument);
      
      long int to_int() const throw();
      std::string to_string(unsigned int base = 10) const throw();
      
      integer& abs() throw();
      
      // returns true if number is zero or positive
      bool positive() const throw();
      
      unsigned long int bits() const throw(); // number of bits used by number
      
      
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
