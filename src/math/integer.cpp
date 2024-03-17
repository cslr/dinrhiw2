

#include <gmp.h>
#include <stdlib.h>
#include "integer.h"


namespace whiteice
{
  namespace math
  {
    
    integer::integer()
    {
      mpz_init_set_ui(integ, 0);
    }

    
    integer::integer(const integer& i)
    {
      mpz_init_set(integ, i.integ);
    }
    
    
    integer::integer(const long int& t)
    {
      mpz_init_set_si(integ, t);
    }
    
    
    integer::integer(const std::string& s, unsigned int base)
    {
      mpz_init_set_str(integ, s.c_str(), base);
    }
    
    
    integer::~integer()
    {
      mpz_clear(integ);
    }
    
    
    
    // operators
    integer integer::operator+(const integer& i) const 
    {      
      integer j;
      
      mpz_add(j.integ, this->integ, i.integ);
      
      return j;
    }
    
    
    integer integer::operator-(const integer& i) const 
    {
      integer j;
      
      mpz_sub(j.integ, this->integ, i.integ);
      
      return j;
    }
    
    
    integer integer::operator*(const integer& i) const 
    {
      integer j;
      
      mpz_mul(j.integ, this->integ, i.integ);
      
      return j;
    }
    
    
    integer integer::operator/(const integer& i) const 
    {
      integer j;
      
      if(mpz_cmp_ui(i.integ,0) == 0)
	throw illegal_operation("Divide by zero");
      
      mpz_tdiv_q(j.integ, this->integ, i.integ);
      
      return j;
    }
    
    
    integer integer::operator%(const integer& i) const 
    {
      integer j;
      
      if(mpz_cmp_ui(i.integ,0) == 0)
	throw illegal_operation("Divide by zero");
      
      mpz_tdiv_r(j.integ, this->integ, i.integ);
      // mpz_mod(j.integ, this->integ, i.integ);
      
      return j;
    }
    
    
    // complex conjugate (?)
    integer integer::operator!() const 
    {
      integer i;
      
      mpz_com(i.integ, this->integ);
      
      return i;
    }
    
    
    integer integer::operator-() const 
    {
      integer i(*this);

      // changes sign;
      mpz_neg(i.integ, this->integ);
      
      return i;
    }
    
    
    integer& integer::operator&=(const integer& i)
      
    {
      integer j(*this);
      
      mpz_and(this->integ, j.integ, i.integ);
      
      return (*this);
    }
    
    
    integer& integer::operator|=(const integer& i)
      
    {
      integer j(*this);
      
      mpz_ior(this->integ, j.integ, i.integ);
      
      return (*this);
    }
    
    
    integer& integer::operator^=(const integer& i)
      
    {
      integer j(*this);
      
      mpz_xor(this->integ, j.integ, i.integ);
      
      return (*this);
    }
    
    
    integer integer::operator&(const integer& i)
      
    {
      integer j;
      
      mpz_and(j.integ, this->integ, i.integ);
      
      return j;
    }
    
    
    integer integer::operator|(const integer& i)
      
    {
      integer j;
      
      mpz_ior(j.integ, this->integ, i.integ);
      
      return j;
    }
    
    
    integer integer::operator^(const integer& i)
      
    {
      integer j;
      
      mpz_xor(j.integ, this->integ, i.integ);
      
      return j;
    }
    
    
    bool integer::getbit(unsigned int index) const 
    {
      return ((bool)mpz_tstbit(this->integ, index));
    }
    
    
    void integer::setbit(unsigned int index, bool value) 
    {
      if(value)
	mpz_setbit(this->integ, index);
      else
	mpz_clrbit(this->integ, index);
    }
    
    
    void integer::clrbit(unsigned int index) 
    {
      mpz_clrbit(this->integ, index);
    }
    
    
    integer integer::operator<<(unsigned int left) const 
    {
      integer i;
      
      mpz_mul_2exp(i.integ, this->integ, left);
      
      return i;
    }
    
    
    integer integer::operator>>(unsigned int right) const 
    {
      integer i;
      
      mpz_tdiv_q_2exp(i.integ, this->integ, right);
      
      return i;
    }
    
    
    integer& integer::operator<<=(unsigned int left) 
    {
      integer i(*this);
      
      mpz_mul_2exp(this->integ, i.integ, left);
      
      return (*this);
    }
    
    
    integer& integer::operator>>=(unsigned int right) 
    {
      integer i(*this);
      
      mpz_tdiv_q_2exp(this->integ, i.integ, right);
      
      return (*this);
    }
    
    
    // left is the positive direction
    integer& integer::circularshift(int shift) 
    {
      if(shift > 0){ // left shift
	
	mpz_t a;
	mpz_t b;
	
	mpz_init(a);
	mpz_init(b);
	
	const unsigned long int B = mpz_sizeinbase(integ,2);
	
	mpz_tdiv_q_2exp(a, integ, B - shift);
	mpz_mul_2exp(b, integ, shift);
	
	mpz_ior(integ, a, b); // integ = [b_low | a]
	
	return (*this);
      }
      else{ // right shift
	shift = -shift;
	
	mpz_t a;
	mpz_t b;
	
	mpz_init(a);
	mpz_init(b);
	
	const unsigned long int B = mpz_sizeinbase(integ,2);
	
	mpz_tdiv_q_2exp(a, integ, shift);
	mpz_mul_2exp(b, integ, B - shift);
	
	mpz_ior(integ, a, b); // integ = [b_low | a]
	
	return (*this);	
      }
    }
    
    
    

    integer& integer::operator+=(const integer& i) 
    {
      integer j(*this);
      
      mpz_add(this->integ, j.integ, i.integ);
      
      return (*this);
    }
    
    
    integer& integer::operator-=(const integer& i) 
    {
      integer j(*this);
      
      mpz_sub(this->integ, j.integ, i.integ);
      
      return (*this);
    }
    
    
    integer& integer::operator*=(const integer& i) 
    {
      integer j(*this);
      
      mpz_mul(this->integ, j.integ, i.integ);
      
      return (*this);
    }
    
    
    integer& integer::operator/=(const integer& i) 
    {
      integer j(*this);
      
      if(mpz_cmp_ui(i.integ,0) == 0)
	throw illegal_operation("Divide by zero");
      
      mpz_tdiv_q(this->integ, j.integ, i.integ);
      
      return (*this);
    }
    
    
    integer& integer::operator%=(const integer& i) 
    {
      integer j(*this);
      
      if(mpz_cmp_ui(i.integ,0) == 0)
	throw illegal_operation("Divide by zero");
      
      //mpz_mod(this->integ, j.integ, i.integ);
      mpz_tdiv_r(this->integ, j.integ, i.integ);
      
      return (*this);
    }
    
    
    integer& integer::operator=(const integer& i) 
    {      
      mpz_set(this->integ, i.integ);
      
      return (*this);
    }
    
    
    integer& integer::operator++() 
    {
      integer j(*this);
      
      mpz_add_ui(this->integ, j.integ, 1);
      
      return (*this);
    }
    
    
    integer& integer::operator--() 
    {
      integer j(*this);
      
      mpz_sub_ui(this->integ, j.integ, 1);
      
      return (*this);
    }
    
    
    integer& integer::operator++(int d) 
    {
      integer j(*this);
      
      mpz_add_ui(this->integ, j.integ, 1);
      
      return (*this);
    }
    
    
    integer& integer::operator--(int d) 
    {
      integer j(*this);
      
      mpz_sub_ui(this->integ, j.integ, 1);
      
      return (*this);
    }
    
    
    bool integer::operator==(const integer& i) const 
    {
      return (mpz_cmp(this->integ, i.integ) == 0);
    }
    
    
    bool integer::operator!=(const integer& i) const 
    {
      return (mpz_cmp(this->integ, i.integ) != 0);
    }

    
    bool integer::operator>=(const integer& i) const 
    {
      return (mpz_cmp(this->integ, i.integ) >= 0);
    }

    
    bool integer::operator<=(const integer& i) const 
    {
      return (mpz_cmp(this->integ, i.integ) <= 0);
    }

    
    bool integer::operator< (const integer& i) const 
    {
      return (mpz_cmp(this->integ, i.integ) < 0);
    }
    
    
    bool integer::operator> (const integer& i) const 
    {
      return (mpz_cmp(this->integ, i.integ) > 0);
    }
    
    // scalar operation
    integer& integer::operator= (const int& s) 
    {
      mpz_set_si(integ, s);
      
      return (*this);
    }
    
    
    integer  integer::operator* (const int& s) const 
    {
      integer i;
      
      mpz_mul_si(i.integ, this->integ, s);
      
      return i;
    }
    
    
    integer  integer::operator/ (const int& s) const 
    {
      if(s == 0)
	throw illegal_operation("Divide by zero");
      
      integer i;
      integer si(s);
      
      mpz_tdiv_q(i.integ, integ, si.integ);
      
      return i;
    }
    
    
    integer& integer::operator*=(const int& s) 
    {
      integer j(*this);
      
      mpz_mul_si(integ, j.integ, s);
      
      return (*this);
    }
    
    
    integer& integer::operator/=(const int& s) 
    {
      if(s == 0)
	throw illegal_operation("Divide by zero");
      
      integer si(s);
      integer j(*this);
      
      mpz_tdiv_q(integ, j.integ, si.integ);
      
      return (*this);
    }
    
    
    integer& integer::abs() 
    {
      mpz_abs(integ, integ);
      
      return (*this);
    }
    
    
    
    long int integer::to_int() const 
    {
      return mpz_get_si(integ);
    }
    
    
    std::string integer::to_string(unsigned int base) const 
    {
      std::string str;
      
      char *str_ptr =
	mpz_get_str(0, base, integ);
      
      if(str_ptr){
	str = str_ptr;
	free(str_ptr);
      }
      
      return str;
    }
    
    
    // returns true if number is zero or positive
    bool integer::positive() const 
    {
      return (mpz_sgn(integ) >= 0);
    }

    
    // number of bits used by number
    unsigned long int integer::bits() const 
    {
      return mpz_sizeinbase(this->integ, 2);
    }
    
    /******************************************************************************************/
    // these implementations use directly gmp library functions.. (don't bother to reinvent wheel
    // + code working directly with gmp data structures is faster)
    
    void modular_exponentation(integer& x, const integer& e, const integer& n)
    {
      integer base(x);
      
      // (probably) uses squaring technique + keeping result small with "%" operator (congruence)
      
      mpz_powm(x.integ, base.integ, e.integ, n.integ); // x = base**e (mod n)
      
    }
    
    
    void gcd(integer& res, const integer& x, const integer& y)
    {
      // uses euclid algorithm
      
      mpz_gcd(res.integ, x.integ, y.integ);
    }
    
    
    bool probably_prime(const integer& x)
    {
      // performs trivial divisions with small primes and miller-rabin test
      
      // 4^-50 = 2^-100 = (2^-20)^5 = (10^-6)^5 = 10^-30 should be ok.
      // (above means this won't probably happen before the end of the universe
      //  (if it's 100 (american) billion years)
      //  assuming probably_prime() will be called once every second by million people)
      
      return (mpz_probab_prime_p(x.integ, 50) != 0);
    }
    
    
    void modular_inverse(integer& inv, const integer a, const integer& n)
    {
      mpz_invert(inv.integ, a.integ, n.integ);
    }
    
    
    
    integer factorial(integer a)
    {
      const integer zero(0);
      integer result(1);      
      
      while(a != zero){
	result *= a;
	a--;
      }
      
      return result;
    }
    
    
    integer combinations(integer a, integer b)
    {
      const integer zero(0);
      integer c = a - b;
      integer result(1);
      integer temp(1);
      
      
      if(c > b){
	while(a > c){
	  result *= a;
	  a--;
	}
	
	while(b != zero){
	  temp *= b;
	  b--;
	}
	
	result /= temp;
      }
      else{
	while(a > b){
	  result *= a;
	  a--;
	}
	
	while(c != zero){
	  temp *= c;
	  c--;
	}
	
	result /= temp;
      }
      
      
      return result;
    }
    
    
    /******************************************************************************************/
    
    
    std::ostream& operator<<(std::ostream& ios,
			     const integer& i)
    {
      
      if(ios.flags() & std::ios::dec){
	ios << i.to_string(10);
      }
      else if(ios.flags() & std::ios::hex){
	ios << i.to_string(16);
      }
      else if(ios.flags() & std::ios::oct){
	ios << i.to_string(8);
      }
      else{
	ios << i.to_string(10);
	ios << " [10-BASE]" << std::endl;
      }
      
      return ios;
    }
    
  };
};







