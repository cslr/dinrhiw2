
#ifndef modular_cpp
#define modular_cpp

#include "modular.h"
#include "blade_math.h"

#include <exception>
#include <stdexcept>
#include <math.h>


namespace whiteice
{
  namespace math
  {
    
    // ctors
    
    template <typename T>
    modular<T>::modular()
    {
      this->value = 0;
      this->modulo = 1;
    }
    
    
    template <typename T>
    modular<T>::modular(const modular<T>& a)
    {
      this->value  = a.value;
      this->modulo = a.modulo;
    }
    
    template <typename T>
    modular<T>::modular(const T& value, const T& modulo)
    {
      if(value >= modulo){
	this->value = value % modulo;
      }
      else if(value < T(0)){
	this->value = (T(1)+ (whiteice::math::abs(value) / this->modulo))*this->modulo;
	this->value += value;
	this->value %= modulo;
      }
      else{
	this->value = value;
      }
      
      this->modulo = modulo;
    }
    
    
    template <typename T>
    modular<T>::modular(const T& modulo)
    {
      this->value = T(0);
      this->modulo = modulo;
    }
    
    
    template <typename T>
    modular<T>::~modular(){ }
    
    //////////////////////////////////////////////////
        
    // operators    
    template <typename T>
    modular<T> modular<T>::operator+(const modular<T>& a) const 
      
    {
      if(a.modulo != this->modulo)
	throw illegal_operation("cannot mix numbers of different modulus");
      
      return modular<T>(this->value + a.value);
    }
    
    
    template <typename T>
    modular<T> modular<T>::operator-(const modular<T>& a) const 
      
    {
      if(a.modulo != this->modulo)
	throw illegal_operation("cannot mix numbers of different modulus");
      
      return modular<T>(this->value + (this->modulo - a.value),
			this->modulo);
    }
    
    
    template <typename T>
    modular<T> modular<T>::operator*(const modular<T>& a) const 
      
    {
      if(a.modulo != this->modulo)
	throw illegal_operation("cannot mix numbers of different modulus");
      
      return modular<T>(this->value * a.value);
    }
    
    
    template <typename T>
    modular<T> modular<T>::operator/(const modular<T>& a) const 
      
    {
      if(a.value == 0)
	throw illegal_operation("cannot divide by zero");
      
      modular<T> b(*this);
      b /= a;
      
      return b;
    }
    
    
    // complex conjugate (?)
    template <typename T>
    modular<T> modular<T>::operator!() const 
      
    {
      return (*this);
    }
    
    
    template <typename T>
    modular<T> modular<T>::operator-() const 
      
    {
      return modular<T>(((T(1) + whiteice::math::abs(this->value)/this->modulo)*this->modulo) - this->value,
			this->modulo);
    }
    
    
    template <typename T>
    modular<T>& modular<T>::operator+=(const modular<T>& a)
      
    {
      if(a.modulo != this->modulo)
	throw illegal_operation("cannot mix numbers of different modulus");
	
      this->value += a.value;
      if(this->value >= this->modulo)
	this->value %= this->modulo;
      else if(value < T(0)){
	this->value += (T(1)+ (whiteice::math::abs(value) / this->modulo))*this->modulo;
	this->value %= modulo;
      }
      
      return *this;
    }
    
    
    template <typename T>
    modular<T>& modular<T>::operator-=(const modular<T>& a) 
      
    {
      if(a.modulo != this->modulo)
	throw illegal_operation("cannot mix numbers of different modulus");
      
      this->value += ((T(1) + whiteice::math::abs(a.value)/this->modulo)*this->modulo) - a.value;
	
      if(this->value >= this->modulo)
	this->value %= this->modulo;
      else if(value < T(0)){
	this->value += (T(1)+ (whiteice::math::abs(value) / this->modulo))*this->modulo;
	this->value %= modulo;
      }
      
      return *this;
    }
    
    
    template <typename T>
    modular<T>& modular<T>::operator*=(const modular<T>& a)
      
    {
      if(a.modulo != this->modulo)
	throw illegal_operation("cannot mix numbers of different modulus");
      
      this->value *= a.value;
      
      if(this->value >= this->modulo){
	this->value %= this->modulo;
      }
      else if(value < T(0)){
	this->value += (T(1)+ (whiteice::math::abs(value) / this->modulo))*this->modulo;
	this->value %= modulo;
      }
      
      return *this;
    }
    

    template <typename T>
    modular<T>& modular<T>::operator/=(const modular<T>& a)
      
    {
      if(a.modulo != this->modulo)
	throw illegal_operation("cannot mix numbers of different modulus");
      
      // (lamely copied from free code from the web)
      // (optimize: calculation of correct value without saving
      //  dividends/divisors etc. is possible ("forward" algorithm)).
      //  This ('backward'): CPU: 2*N, MEM: 4*N  
      //  Forward: CPU: N, MEM 0
      
      std::vector<T> dvd,dvr,qnt,rem;
        
      unsigned int n = 0;
      dvd.push_back(a.modulo);
      dvr.push_back(a.value);
		    
      do{
	qnt.push_back(dvd[n]/dvr[n]);
	rem.push_back(dvd[n]%dvr[n]);

	dvd.push_back(dvr[n]);
	dvr.push_back(rem[n]);
	
	n++;
      }
      while(dvr[dvr.size()-1] != T(0));
      
      if(dvd[n] != T(1))
	throw illegal_operation("number don't have inverse (gcd != 1)");
      
      if(T(n) == T(1)){
	// this->value = this->value
	return *this;
      }
      
      std::vector<T> pterm, pcoef, mterm, mcoef;
      pterm.resize(n-1); pcoef.resize(n-1);
      mterm.resize(n-1); mcoef.resize(n-1);
      
      pterm[n-2] = dvd[n-2];
      pcoef[n-2] = T(1);
      mterm[n-2] = dvr[n-2];
      mcoef[n-2] = qnt[n-2];
      
      if(n >= 3){
	
	unsigned int j = n - 3;
      
	while(j>=0){
	
	  if(rem[j] == pterm[j+1]){
	    pterm[j] = dvd[j];
	    pcoef[j] = pcoef[j+1];
	    mterm[j] = mterm[j+1];
	    mcoef[j] = pcoef[j+1]*qnt[j] + mcoef[j+1];
	  }
	  else{
	    pterm[j] = pterm[j+1];
	    pcoef[j] = mcoef[j+1]*qnt[j] + pcoef[j+1];
	    mterm[j] = dvd[j];
	    mcoef[j] = mcoef[j+1];
	  }
	  
	  j--;
	}
	
      }

      if(pterm[0] == a.value){
	this->value *= pcoef[0];
	
	if(this->value >= this->modulo){
	  this->value %= this->modulo;
	}
	else if(value < T(0)){
	  this->value += (T(1)+ (whiteice::math::abs(value) / this->modulo))*this->modulo;
	  this->value %= modulo;
	}
	
	return *this;
      }
      else{
	// mcoef[0] = -mcoef[0] (mod this->modulo)
	mcoef[0] = 
	  (T(1) + whiteice::math::abs(mcoef[0]) / this->modulo)*this->modulo 
	  - mcoef[0];
	
	this->value *= mcoef[0];
	
	if(this->value >= this->modulo){
	  this->value %= this->modulo;
	}
	else if(value < T(0)){
	  this->value += (T(1)+ (whiteice::math::abs(value) / this->modulo))*this->modulo;
	  this->value %= modulo;
	}
	
	return *this;
      }
      
    }
    
    
    template <typename T>
    modular<T>& modular<T>::operator=(const modular<T>& a)
      
    {
      if(a.modulo != this->modulo)
	throw illegal_operation("cannot mix numbers of different modulus");
      
      this->value  = a.value;
      this->modulo = a.modulo;
      
      return *this;
    }
    
    
    template <typename T>
    modular<T> operator*(const T& s, const modular<T>& a)
      
    {
      return modular<T>(a.value * s);
    }
    
    
    template <typename T>
    bool modular<T>::operator==(const modular<T>& a) const 
      
    {
      if(a.modulo != this->modulo)
	throw uncomparable("cannot compare numbers with different modulus");
      
      return (this->value == a.value);
    }
    
    
    template <typename T>
    bool modular<T>::operator!=(const modular<T>& a) const
      
    {
      if(a.modulo != this->modulo)
	throw uncomparable("cannot compare numbers with different modulus");
      
      return (this->value != a.value);
    }
    
    
    template <typename T>
    bool modular<T>::operator>=(const modular<T>& a) const
      
    {
      if(a.modulo != this->modulo)
	throw uncomparable("cannot compare numbers with different modulus");
      
      return (this->value >= a.value);
    }
    
    
    template <typename T>
    bool modular<T>::operator<=(const modular<T>& a) const
      
    {
      if(a.modulo != this->modulo)
	throw uncomparable("cannot compare numbers with different modulus");
      
      return (this->value <= a.value);
    }
    
    
    template <typename T>
    bool modular<T>::operator< (const modular<T>& a) const
      
    {
      if(a.modulo != this->modulo)
	throw uncomparable("cannot compare numbers with different modulus");
      
      return (this->value < a.value);
    }
    
    
    template <typename T>
    bool modular<T>::operator> (const modular<T>& a) const
      
    {
      if(a.modulo != this->modulo)
	throw uncomparable("cannot compare numbers with different modulus");
      
      return (this->value > a.value);
    }
    
    
    // scalar operation
    template <typename T>
    modular<T>& modular<T>::operator= (const T& s)
      
    {
      this->value = s;
      
      if(this->value >= this->modulo)
	this->value %= this->modulo;
      else if(value < T(0)){
	this->value += (T(1)+ (whiteice::math::abs(value) / this->modulo))*this->modulo;
	this->value %= modulo;
      }
      
      return *this;
    }
    
    
    template <typename T>
    modular<T>  modular<T>::operator* (const T& s) const 
    {
      return modular<T>((this->value * s), this->modulo);
    }
    
    
    template <typename T>
    modular<T>  modular<T>::operator/ (const T& s) const 
      
    {
      if(s == T(0))
	throw std::invalid_argument("cannot divide by zero");
      
      modular<T> r(*this);
      modular<T> t(s, this->modulo);
      
      r /= t;
      
      return r;
    }
    
    
    template <typename T>
    modular<T>& modular<T>::operator*=(const T& s) 
    {
      this->value *= s;
      
      if(this->value >= this->modulo)
	this->value %= this->modulo;
      else if(value < T(0)){
	this->value += (T(1)+ (whiteice::math::abs(value) / this->modulo))*this->modulo;
	this->value %= modulo;
      }
      
      return *this;
    }
    
    
    template <typename T>
    modular<T>& modular<T>::operator/=(const T& s)
      
    {
      if(s == T(0))
	throw std::invalid_argument("cannot divide by zero");
      
      modular<T> t(s, this->modulo);
      
      *this /= t;
      
      return *this;
    }
    
    
    template <typename T>
    modular<T>& modular<T>::abs() 
    {
      this->value = whiteice::math::abs(this->value);
      return *this;
    }
    
    
    template <typename T>
    T& modular<T>::operator[](const unsigned int& index)
      
    {
      if(index == 0)
	return this->value;
      else if(index == 1)
	return this->modulo;
      else
	throw std::out_of_range("modular number: index too big >= 2");
    }
    
    
    template <typename T>
    const T& modular<T>::operator[](const unsigned int& index) const
      
    {
      if(index == 0)
	return this->value;
      else if(index == 1)
	return this->modulo;
      else
	throw std::out_of_range("modular number: index too big >= 2");      
    }
    

    /********************************************************/
    
    template <typename T>
    std::ostream& operator<<(std::ostream& ios,
			     const whiteice::math::modular<T>& m){
      ios << m[0];
      
      return ios;
    }
    
    
    
    // explicit template instantations
    
    template class modular<unsigned int>;
    template class modular<unsigned short>;
    template class modular<unsigned char>;
    template class modular<integer>;
    
    
    template modular<unsigned int> operator*<unsigned int>
      (const unsigned int& s, const modular<unsigned int>&) ;
    
    template modular<unsigned short> operator*<unsigned short>
      (const unsigned short& s, const modular<unsigned short>&) ;
    
    template modular<unsigned char> operator*<unsigned char>
      (const unsigned char& s, const modular<unsigned char>&) ;
    
    template modular<integer> operator*<integer>
      (const integer& s, const modular<integer>&) ;
    
    
    template std::ostream& operator<< <unsigned int>(std::ostream& ios, const whiteice::math::modular<unsigned int>&);
    template std::ostream& operator<< <unsigned short>(std::ostream& ios, const whiteice::math::modular<unsigned short>&);
    template std::ostream& operator<< <unsigned char>(std::ostream& ios, const whiteice::math::modular<unsigned char>&);
    template std::ostream& operator<< <integer>(std::ostream& ios, const whiteice::math::modular<integer>&);
    
    
  };
};

#endif
