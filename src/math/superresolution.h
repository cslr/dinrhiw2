/*
 * superresolutional / dimensional numbers with basis
 * values of type T and exponents/dimensions of type U
 */
#ifndef superresolution_h
#define superresolution_h

#include "number.h"
#include "modular.h"
#include "blade_math.h"
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
	
	// superresolution();
	superresolution(const unsigned int resolution = DEFAULT_MODULAR_BASIS);
	superresolution(const U& resolution);
	superresolution(const superresolution<T,U>& s);
	superresolution(const std::vector<T>& values);
	virtual ~superresolution();
	
	// operators
	virtual superresolution<T,U> operator+(const superresolution<T,U>&) const ;
	virtual superresolution<T,U> operator-(const superresolution<T,U>&) const ;
	virtual superresolution<T,U> operator*(const superresolution<T,U>&) const ;
	virtual superresolution<T,U> operator/(const superresolution<T,U>&) const ;
	
	// complex conjugate (?)
	virtual superresolution<T,U> operator!() const;
	
	virtual superresolution<T,U>& conj();
	
	virtual superresolution<T,U> operator-() const;
	
      
	virtual superresolution<T,U>& operator+=(const superresolution<T,U>&) ;
	virtual superresolution<T,U>& operator-=(const superresolution<T,U>&) ;
	virtual superresolution<T,U>& operator*=(const superresolution<T,U>&) ;
	virtual superresolution<T,U>& operator/=(const superresolution<T,U>&) ;
	
	virtual superresolution<T,U>& operator=(const superresolution<T,U>&) ;      
	
	virtual bool operator==(const superresolution<T,U>&) const ;
	virtual bool operator!=(const superresolution<T,U>&) const ;
	virtual bool operator>=(const superresolution<T,U>&) const ;
	virtual bool operator<=(const superresolution<T,U>&) const ;
	virtual bool operator< (const superresolution<T,U>&) const ;
	virtual bool operator> (const superresolution<T,U>&) const ;
	
	// scalar operation
	virtual superresolution<T,U>& operator= (const T& s) ;
	virtual superresolution<T,U>  operator* (const T& s) const ;
	virtual superresolution<T,U>  operator/ (const T& s) const ;
	virtual superresolution<T,U>& operator*=(const T& s) ;
	virtual  superresolution<T,U>& operator/=(const T& s) ;
	
	virtual superresolution<T,U>& abs();
	virtual superresolution<T,U>& zero();
        virtual bool iszero() const;
	
	virtual T& operator[](const U& index);
	
	virtual const T& operator[](const U& index) const;
	
	// superresolution operations
	virtual superresolution<T,U>& basis_scaling(const T& s) ; // uniform
	virtual superresolution<T,U>& basis_scaling(const std::vector<T>& s) ; // non-uniform scaling
	virtual T measure(const U& s) const; // measures with s-(dimensional) measure-function

	virtual bool comparable(){
	  if(basis.size() == 1) return true;
	  else return false;
	}

	virtual unsigned int size(){ return this->basis.size(); }
	
      private:
	
	// superresolution basis
	// (todo: switch to 'sparse' representation with (U,T) pairs)
	// and ordered by U (use own avltree implementation)
	
	std::vector<T> basis;
      };
    
    
  }
}


// #include "superresolution.cpp"

namespace whiteice{

  namespace math
  {
    extern template class superresolution< whiteice::math::blas_real<float>,
					   whiteice::math::modular<unsigned int> >;
    extern template class superresolution< whiteice::math::blas_real<double>,
					   whiteice::math::modular<unsigned int> >;
    
    extern template class superresolution< whiteice::math::blas_complex<float>,
					   whiteice::math::modular<unsigned int> >;
    extern template class superresolution< whiteice::math::blas_complex<double>,
					   whiteice::math::modular<unsigned int> >;
  }
  
};


#endif
