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
	superresolution<T,U> operator+(const superresolution<T,U>&) const throw(illegal_operation);
	superresolution<T,U> operator-(const superresolution<T,U>&) const throw(illegal_operation);
	superresolution<T,U> operator*(const superresolution<T,U>&) const throw(illegal_operation);
	superresolution<T,U> operator/(const superresolution<T,U>&) const throw(illegal_operation);
	
	// complex conjugate (?)
	superresolution<T,U> operator!() const throw(illegal_operation);
	superresolution<T,U> operator-() const throw(illegal_operation);
	
	superresolution<T,U>& operator+=(const superresolution<T,U>&) throw(illegal_operation);
	superresolution<T,U>& operator-=(const superresolution<T,U>&) throw(illegal_operation);
	superresolution<T,U>& operator*=(const superresolution<T,U>&) throw(illegal_operation);
	superresolution<T,U>& operator/=(const superresolution<T,U>&) throw(illegal_operation);
	
	superresolution<T,U>& operator=(const superresolution<T,U>&) throw(illegal_operation);      
	
	bool operator==(const superresolution<T,U>&) const throw(uncomparable);
	bool operator!=(const superresolution<T,U>&) const throw(uncomparable);
	bool operator>=(const superresolution<T,U>&) const throw(uncomparable);
	bool operator<=(const superresolution<T,U>&) const throw(uncomparable);
	bool operator< (const superresolution<T,U>&) const throw(uncomparable);
	bool operator> (const superresolution<T,U>&) const throw(uncomparable);
	
	// scalar operation
	superresolution<T,U>& operator= (const T& s) throw(illegal_operation);
	superresolution<T,U>  operator* (const T& s) const throw();
	superresolution<T,U>  operator/ (const T& s) const throw(std::invalid_argument);
	superresolution<T,U>& operator*=(const T& s) throw();
	superresolution<T,U>& operator/=(const T& s) throw(std::invalid_argument);
	
	superresolution<T,U>& abs() throw();      
	
	T& operator[](const U& index)
	  throw(std::out_of_range, illegal_operation);
	
	const T& operator[](const U& index) const
	  throw(std::out_of_range, illegal_operation);	
	
	// superresolution operations
	void basis_scaling(const T& s) throw(); // uniform
	bool basis_scaling(const std::vector<T>& s) throw(); // non-uniform scaling
	T measure(const U& s) throw(); // measures with s-(dimensional) measure-function
	
	
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
