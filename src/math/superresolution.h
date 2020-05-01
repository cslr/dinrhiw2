/*
 * superresolutional / dimensional numbers with basis
 * values of type T and exponents/dimensions of type U
 */
#ifndef superresolution_h
#define superresolution_h

#include "number.h"
#include <vector>


namespace whiteice
{
  namespace math
  {
    /* superresolutional numbers
     * made out of field T with exponent field U
     */
    template <typename T, typename U>
      class superresolution : 
        public number<superresolution<T,U>, T, T, U>
      {
      public:
	
	superresolution();
	superresolution(const U& resolution);
	superresolution(const superresolution<T,U>& s);
	superresolution(const std::vector<T>& values);
	virtual ~superresolution();
	
	// operators
	superresolution<T,U> operator+(const superresolution<T,U>&) const ;
	superresolution<T,U> operator-(const superresolution<T,U>&) const ;
	superresolution<T,U> operator*(const superresolution<T,U>&) const ;
	superresolution<T,U> operator/(const superresolution<T,U>&) const ;
	
	// complex conjugate (?)
	superresolution<T,U> operator!() const ;
	superresolution<T,U> operator-() const ;
	
	superresolution<T,U>& operator+=(const superresolution<T,U>&) ;
	superresolution<T,U>& operator-=(const superresolution<T,U>&) ;
	superresolution<T,U>& operator*=(const superresolution<T,U>&) ;
	superresolution<T,U>& operator/=(const superresolution<T,U>&) ;
	
	superresolution<T,U>& operator=(const superresolution<T,U>&) ;      
	
	bool operator==(const superresolution<T,U>&) const ;
	bool operator!=(const superresolution<T,U>&) const ;
	bool operator>=(const superresolution<T,U>&) const ;
	bool operator<=(const superresolution<T,U>&) const ;
	bool operator< (const superresolution<T,U>&) const ;
	bool operator> (const superresolution<T,U>&) const ;
	
	// scalar operation
	superresolution<T,U>& operator= (const T& s) ;
	superresolution<T,U>  operator* (const T& s) const ;
	superresolution<T,U>  operator/ (const T& s) const ;
	superresolution<T,U>& operator*=(const T& s) ;
	superresolution<T,U>& operator/=(const T& s) ;
	
	superresolution<T,U>& abs() ;      
	
	T& operator[](const U& index)
	  ;
	
	const T& operator[](const U& index) const
	  ;	
	
	// superresolution operations
	void basis_scaling(const T& s) ; // uniform
	bool basis_scaling(const std::vector<T>& s) ; // non-uniform scaling
	T measure(const U& s) ; // measures with s-(dimensional) measure-function
	
	
      private:
	
	// superresolution basis
	// (todo: switch to 'sparse' representation with (U,T) pairs)
	// and ordered by U (use own avltree implementation)
	
	std::vector<T> basis;
      };
    
    
  }
}


#include "superresolution.cpp"


#endif
