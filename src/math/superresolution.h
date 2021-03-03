/*
 * superresolutional / dimensional numbers with basis
 * values of type T and exponents/dimensions of type U
 */
#ifndef superresolution_h
#define superresolution_h

namespace whiteice
{
  namespace math
  {
    template <typename T, typename S>
    class superresolution;
    
  };
};

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
    //class superresolution : public number<superresolution<T,U>, T, T, U>
    struct superresolution
    {
      public:
	
        superresolution();
        superresolution(const T value);
        // superresolution(const U& resolution);
	superresolution(const superresolution<T,U>& s);
	superresolution(const std::vector<T>& values);
        ~superresolution();
	
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

        // WARN! comparision operators assume infinite basis and not cyclic exponent basis
        // this means comparision operators don't work properly b > c => a*b > a*c, a > 0
        // don't hold etc. Comparision work with static numbers but not with operations
        // changing multiple basis numbers
        virtual bool operator>=(const superresolution<T,U>&) const ;
	virtual bool operator<=(const superresolution<T,U>&) const ;
	virtual bool operator< (const superresolution<T,U>&) const ;
	virtual bool operator> (const superresolution<T,U>&) const ;
	
	// scalar operation
	virtual superresolution<T,U>& operator= (const T& s) ;
	virtual superresolution<T,U>  operator+ (const T& s) const ;
	virtual superresolution<T,U>  operator- (const T& s) const ;
	virtual superresolution<T,U>  operator* (const T& s) const ;
	virtual superresolution<T,U>  operator/ (const T& s) const ;
	virtual superresolution<T,U>& operator*=(const T& s) ;
	virtual superresolution<T,U>& operator/=(const T& s) ;

	virtual superresolution<T,U>& operator=(const T value) ;
      
#if 0
	virtual superresolution<T,U>  operator* (const T value) const ;
	virtual superresolution<T,U>  operator/ (const T value) const ;
	virtual superresolution<T,U>& operator*=(const T value) ;
	virtual superresolution<T,U>& operator/=(const T value) ;
#endif
	

	
	virtual superresolution<T,U>& abs();
	virtual superresolution<T,U>& zero();
        virtual bool iszero() const;
	
	virtual T& operator[](const U& index);
	
	virtual const T& operator[](const U& index) const;

	inline virtual T& first(){ return basis[0]; }

	inline virtual const T& first() const{ return basis[0]; }
	
	inline virtual T& first(const T value){
	  basis[0] = value;
	  return basis[0];
	}
	
	// superresolution operations
	virtual superresolution<T,U>& basis_scaling(const T& s) ; // uniform
	virtual superresolution<T,U>& basis_scaling(const std::vector<T>& s) ; // non-uniform scaling
	virtual T measure(const U& s) const; // measures with s-(dimensional) measure-function

	inline virtual bool comparable(){
	  return true; // NOT REALLY COMPARABLE IF OVERFLOW HAPPENS (CIRCULAR CONVOLUTION)
	}

      virtual unsigned int size() const;
      
      public:
	
	// superresolution basis
	// (todo: switch to 'sparse' representation with (U,T) pairs)
	// and ordered by U (use own avltree implementation)
	
        T basis[DEFAULT_MODULAR_BASIS] __attribute__ ((packed));;
      
        // std::vector<T> basis;
      
      };
    
    
  }
}


// #include "superresolution.cpp"

namespace whiteice{

  namespace math
  {
    template <typename T>
    std::ostream& operator<<(std::ostream& ios, const superresolution<T, modular<unsigned int> > & m);
			     
    
    // DO NOT USE BLAS_REAL BUT BLAS_COMPLEX
    extern template class superresolution< whiteice::math::blas_real<float>,
					   whiteice::math::modular<unsigned int> >;
    extern template class superresolution< whiteice::math::blas_real<double>,
					   whiteice::math::modular<unsigned int> >;
    
    extern template class superresolution< whiteice::math::blas_complex<float>,
					   whiteice::math::modular<unsigned int> >;
    extern template class superresolution< whiteice::math::blas_complex<double>,
					   whiteice::math::modular<unsigned int> >;

    
    extern template std::ostream& operator<< <whiteice::math::blas_complex<float> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_complex<float>,
     whiteice::math::modular<unsigned int> >&);

    extern template std::ostream& operator<< <whiteice::math::blas_complex<double> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_complex<double>,
     whiteice::math::modular<unsigned int> >&);

  }
  
};


#endif
