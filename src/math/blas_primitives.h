/*
 * atlas real and complex number
 * template data primitives
 */

#ifndef blas_primitives_h
#define blas_primitives_h

#include "ownexception.h"
#include <stdexcept>
#include <exception>
#include <complex>


namespace whiteice
{
  namespace math
  {
    
#ifdef __GNUG__
    // gnu c++ specific hack to keep sizeof(blas_real<T>) == sizeof(T)
    // (so there's no vtable, padding etc. to enlarge the 'struct'/class)
    // -> can malloc and use blas_xxx structures as pure types
    //    -> memcpy((T*)(blas_real<T>_pointer), xxx) etc.
    //    -> especially now ATLAS library can access data directly (as memory)
    
    template <typename T>
      struct blas_complex;

    
    template <typename T=float>
      struct blas_real
      {
	T c[1] __attribute__ ((packed));
	
	inline blas_real(){ c[0] = (T)0; }
	inline blas_real(const T& t){ c[0] = t; }
	inline blas_real(const blas_real<T>& t){ c[0] = t.c[0]; }

	// work arounds stupid compiler..
	// explicit inline blas_real(const blas_complex<float>& t){ c[0] = t.c[0]; } // takes real part
	// explicit inline blas_real(const blas_complex<double>& t){ c[0] = t.c[0]; } // takes real part
	
	inline ~blas_real(){ }
	
	inline blas_real<T> operator++(int n){
	  if(n) c[0] += T(n);
	  else c[0]++;
	  
	  return (*this);
	}
	
	inline blas_real<T> operator+(const blas_real<T>& t) const throw(illegal_operation)
	{ return blas_real<T>(this->c[0] + t.c[0]); }
	
	inline blas_real<T> operator-(const blas_real<T>& t) const throw(illegal_operation)
	{ return blas_real<T>(this->c[0] - t.c[0]); }
	
	inline blas_real<T> operator*(const blas_real<T>& t) const throw(illegal_operation)
	{ return blas_real<T>(this->c[0] * t.c[0]); }
	     
	inline blas_real<T> operator/(const blas_real<T>& t) const throw(illegal_operation)
	{ return blas_real<T>(this->c[0] / t.c[0]); } // no division by zero checks
	  
	inline blas_real<T> operator!() const throw(illegal_operation) // complex conjugate
	{ return *this;}
	
	inline blas_real<T> operator-() const throw(illegal_operation)
	{ return blas_real<T>(-this->c[0]); }
      
	inline blas_real<T>& operator+=(const blas_real<T>& t) throw(illegal_operation)
	{ this->c[0] += t.c[0]; return *this; }
	     
	inline blas_real<T>& operator-=(const blas_real<T>& t) throw(illegal_operation)
	{ this->c[0] -= t.c[0]; return *this; }
	
	inline blas_real<T>& operator*=(const blas_real<T>& t) throw(illegal_operation)
	{ this->c[0] *= t.c[0]; return *this; }
	     
	inline blas_real<T>& operator/=(const blas_real<T>& t) throw(illegal_operation)
	{ this->c[0] /= t.c[0]; return *this; } // no division by zero checks
	  
	inline blas_real<T>& operator=(const blas_real<T>& t) throw(illegal_operation)
	{ this->c[0] = t.c[0]; return *this; }
	
	inline bool operator==(const blas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] == t.c[0]); }
	     
	inline bool operator!=(const blas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] != t.c[0]); }
	     
	inline bool operator>=(const blas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] >= t.c[0]); }
	     
	inline bool operator<=(const blas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] <= t.c[0]); }
	     
	inline bool operator< (const blas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] < t.c[0]); }
	     
	inline bool operator> (const blas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] > t.c[0]); }
	
	inline bool operator==(const T& t) const throw(uncomparable)
	{ return (this->c[0] == t); }
	     
	inline bool operator!=(const T& t) const throw(uncomparable)
	{ return (this->c[0] != t); }
	     
	inline bool operator>=(const T& t) const throw(uncomparable)
	{ return (this->c[0] >= t); }
	     
	inline bool operator<=(const T& t) const throw(uncomparable)
	{ return (this->c[0] <= t); }
	     
	inline bool operator< (const T& t) const throw(uncomparable)
	{ return (this->c[0] < t); }
	     
	inline bool operator> (const T& t) const throw(uncomparable)
	{ return (this->c[0] > t); }
	
	// scalar operation
	inline blas_real<T>& operator= (const T& s) throw(illegal_operation)
	{ this->c[0] = s; return *this; }

	inline blas_real<T> operator+=(const T& s) throw(illegal_operation)
	{ this->c[0] += s; return *this; } 
	  
	inline blas_real<T>& operator-=(const T& s) throw(illegal_operation)
	{ this->c[0] -= s; return *this; }
	     
	inline blas_real<T>  operator* (const T& s) const throw()
	{ blas_real<T> r; r.c[0] = s * this->c[0]; return r; }
	     
	inline blas_real<T>  operator/ (const T& s) const throw(std::invalid_argument)
	{ blas_real<T> r; r.c[0] =  this->c[0] / s; return r; } // no division by zero checks
	  
	inline blas_real<T>& operator*=(const T& s) throw()
	{ this->c[0] *= s; return *this; }
	     
	inline blas_real<T>& operator/=(const T& s) throw(std::invalid_argument)
	{ this->c[0] /= s; return *this; }
	     
	inline blas_real<T> abs() const
	{ return blas_real<T>( T(fabs((double)c[0])) ); }
	
	/*
	 * inline T& operator[](unsigned int index) throw(std::out_of_range, illegal_operation)
	 * { return c[0]; } // doesn't use index!
	 * 
	 * inline const T& operator[](unsigned int index) const throw(std::out_of_range, illegal_operation)
	 * { return c[0]; } // doesn't use index!
	 *
	 */
	
	inline T real() throw(){ return c[0]; }
	inline const T real() const throw(){ return c[0]; }

	inline T imag() throw(){ return T(0.0f); }
	inline const T imag() const throw(){ return T(0.0f); }
	
	// BUG: DOESN'T COMPILE WITH REFERENCE ALTHOUGH IT SHOULD, removed T&
	inline T value() throw(){ return c[0]; }
	inline const T value() const throw(){ return c[0]; }
	
	
	template <typename A>
	friend blas_real<A> operator*(const A& s, const blas_real<A>& r) throw(std::invalid_argument);
	
	template <typename A>
	friend blas_real<A> operator/(const A& s, const blas_real<A>& r) throw(std::invalid_argument);
	
	template <typename A>
	friend bool operator==(const A& t, const blas_real<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator!=(const A& t, const blas_real<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator>=(const A& t, const blas_real<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator<=(const A& t, const blas_real<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator< (const A& t, const blas_real<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator> (const A& t, const blas_real<A>& r) throw(uncomparable);
	
      } __attribute__ ((packed));
    
    
    
    template <typename T>
      inline blas_real<T> operator*(const T& s, const blas_real<T>& r) throw(std::invalid_argument)
      {
	return blas_real<T>(r * s);
      }
    
    template <typename T>
      inline blas_real<T> operator/(const T& s, const blas_real<T>& r) throw(std::invalid_argument)
      {
	return blas_real<T>(blas_real<T>(s) / r);
      }
    
    
    template <typename T>
      inline bool operator==(const T& t, const blas_real<T>& r) throw(uncomparable)
      { return (r.c[0] == t); }
    
    
    template <typename T>
      inline bool operator!=(const T& t, const blas_real<T>& r) throw(uncomparable)
      { return (r.c[0] != t); }
    
    
    template <typename T>
      inline bool operator>=(const T& t, const blas_real<T>& r) throw(uncomparable)
      { return (t >= r.c[0]); }
    
    
    template <typename T>
      inline bool operator<=(const T& t, const blas_real<T>& r) throw(uncomparable)
      { return (t <= r.c[0]); }
    
    
    template <typename T>
      inline bool operator< (const T& t, const blas_real<T>& r) throw(uncomparable)
      { return (t < r.c[0]); }
    
    
    template <typename T>
      inline bool operator> (const T& t, const blas_real<T>& r) throw(uncomparable)
      { return (t > r.c[0]); }
    
    
    
    
    
    
    
    template <typename T=float>
      struct blas_complex
      {
	T c[2] __attribute__ ((packed));
	
	
	inline blas_complex(){ c[0] = T(0); c[1] = T(0); }
	inline blas_complex(const T& r){ c[0] = r; c[1] = T(0); }
	inline blas_complex(const T& r, const T& i){ c[0] = r; c[1] = i; }
	inline blas_complex(const blas_complex<T>& r){ c[0] = r.c[0]; c[1] = r.c[1]; }
	inline blas_complex(const std::complex<T>& z){ c[0] = std::real(z); c[1] = std::imag(z); }
	  
	// work arounds stupid compiler..
	inline blas_complex(const blas_real<float>& r){ c[0] = r.c[0]; c[1] = T(0); }
	inline blas_complex(const blas_real<double>& r){ c[0] = r.c[0]; c[1] = T(0); }
	
	inline ~blas_complex(){ }
	
	
	inline blas_complex<T> operator+(const blas_complex<T>& t) const throw(illegal_operation)
	{ return blas_complex<T>(this->c[0] + t.c[0], this->c[1] + t.c[1]); }
	
	inline blas_complex<T> operator-(const blas_complex<T>& t) const throw(illegal_operation)
	{ return blas_complex<T>(this->c[0] - t.c[0], this->c[1] - t.c[1]); }
	
	inline blas_complex<T> operator*(const blas_complex<T>& t) const throw(illegal_operation)
	{ return blas_complex<T>(this->c[0] * t.c[0] - this->c[1]*t.c[1],
				  this->c[1] * t.c[0] + this->c[0]*t.c[1]); }
	
	// no division by zero checks
	inline blas_complex<T> operator/(const blas_complex<T>& t) const throw(illegal_operation)
	{ blas_complex<T> r; r.c[0] = (c[0]*t.c[0] + c[1]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]);
	  r.c[1] = (c[1]*t.c[0] - c[0]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]); return r;}
	
	inline blas_complex<T> operator!() const throw(illegal_operation)  // complex conjugate
	{ return blas_complex<T>(this->c[0], -this->c[1]); }
	
	inline blas_complex<T> operator-() const throw(illegal_operation)
	{ return blas_complex<T>(-this->c[0], -this->c[1]); }
      
	inline blas_complex<T>& operator+=(const blas_complex<T>& t) throw(illegal_operation)
	{ this->c[0] += t.c[0]; this->c[1] += t.c[1]; return *this; }
	     
	inline blas_complex<T>& operator-=(const blas_complex<T>& t) throw(illegal_operation)
	{ this->c[0] -= t.c[0]; this->c[1] -= t.c[1]; return *this; }
	
	inline blas_complex<T>& operator*=(const blas_complex<T>& t) throw(illegal_operation)
	{ T a = c[0] * t.c[0] - c[1]*t.c[1]; T b = c[1] * t.c[0] + c[0]*t.c[1];
	  this->c[0] = a; this->c[1] = b; return *this; }
	  
	// no division by zero checks
	inline blas_complex<T>& operator/=(const blas_complex<T>& t) throw(illegal_operation)
	{ T a = (c[0]*t.c[0] + c[1]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]);
	  T b = (c[1]*t.c[0] - c[0]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]);
	  this->c[0] = a; this->c[1] = b; return *this; }	  
	  
	inline blas_complex<T>& operator=(const blas_complex<T>& t) throw(illegal_operation)
	{ this->c[0] = t.c[0]; this->c[1] = t.c[1]; return *this; }
	  
	inline blas_complex<T>& operator=(const blas_real<T>& t) throw(illegal_operation)
	{ this->c[0] = t.c[0]; this->c[1] = T(0.0); return *this; }
	
	inline bool operator==(const blas_complex<T>& t) const throw(uncomparable)
	{ return (this->c[0] == t.c[0] && this->c[1] == t.c[1]); }
	     
	inline bool operator!=(const blas_complex<T>& t) const throw(uncomparable)
	{ return (this->c[0] != t.c[0] && this->c[1] != t.c[1]); }
	     
	inline bool operator>=(const blas_complex<T>& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	     
	inline bool operator<=(const blas_complex<T>& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	     
	inline bool operator< (const blas_complex<T>& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	     
	inline bool operator> (const blas_complex<T>& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	
	inline bool operator==(const T& t) const throw(uncomparable)
	{ return (this->c[0] == t && this->c[1] == 0); }
	     
	inline bool operator!=(const T& t) const throw(uncomparable)
	{ return (this->c[0] != t && this->c[1] != 0); }
	     
	inline bool operator>=(const T& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	     
	inline bool operator<=(const T& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	     
	inline bool operator< (const T& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	     
	inline bool operator> (const T& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	
	// scalar operation
	inline blas_complex<T>& operator= (const T& s) throw(illegal_operation)
	{ this->c[0] = s; this->c[1] = T(0); return *this; }
	
	inline blas_real<T> operator+=(const T& s) throw(illegal_operation)
	{ this->c[0] += s; return *this; }
	
	inline blas_real<T>& operator-=(const T& s) throw(illegal_operation)
	{ this->c[0] -= s; return *this; }
	     
	inline blas_complex<T>  operator* (const T& s) const throw()
	{ blas_complex<T> r; r.c[0] = s * this->c[0]; r.c[1] = s * this->c[1]; return r; }
	
	// no division by zero checks
	inline blas_complex<T>  operator/ (const T& s) const throw(std::invalid_argument)
	{ blas_complex<T> r; r.c[0] =  this->c[0] / s; r.c[1] = this->c[1] / s; return r; }
	  
	inline blas_complex<T>& operator*=(const T& s) throw()
	{ this->c[0] *= s; this->c[1] *= s; return *this; }
	     
	inline blas_complex<T>& operator/=(const T& s) throw(std::invalid_argument)
	{ this->c[0] /= s; this->c[1] /= s; return *this; }
	     
	inline blas_real<T> abs() const
	{ blas_real<T> r; r.c[0] = T(sqrt((double)(c[0]*c[0] + c[1]*c[1]))); return r; }
	
	/*
	 * inline T& operator[](unsigned int index) throw(std::out_of_range, illegal_operation)
	 * { return c[index]; }
	 * 
	 * inline const T& operator[](unsigned int index) const throw(std::out_of_range, illegal_operation)
	 * { return c[index]; }
	 * 
	 */
	
	inline T real() throw(){ return c[0]; }
	inline const T real() const throw(){ return c[0]; }
	
	inline T imag() throw(){ return c[1]; }
	inline const T imag() const throw(){ return c[1]; }
	
	
	
	template <typename A>
	friend blas_complex<A> operator*(const A& s, const blas_complex<A>& r) throw(std::invalid_argument);
	
	template <typename A>
	friend blas_complex<A> operator/(const A& s, const blas_complex<A>& r) throw(std::invalid_argument);
	
	template <typename A>
	friend bool operator==(const A& t, const blas_complex<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator!=(const A& t, const blas_complex<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator>=(const A& t, const blas_complex<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator<=(const A& t, const blas_complex<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator< (const A& t, const blas_complex<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator> (const A& t, const blas_complex<A>& r) throw(uncomparable);
	
      } __attribute__ ((packed));
    
    
    

    template <typename T>
      inline blas_complex<T> operator*(const T& s, const blas_complex<T>& r) throw(std::invalid_argument)
      {
	return blas_complex<T>(r * s);
      }
    
    
    template <typename T>
      inline blas_complex<T> operator/(const T& s, const blas_complex<T>& r) throw(std::invalid_argument)
      {
	return blas_complex<T>(blas_complex<T>(s) / r);
      }
    
    
    template <typename T>
      inline bool operator==(const T& t, const blas_complex<T>& r) throw(uncomparable)
      { return (r.c[0] == t && r.c[1] == 0); }
    
    
    template <typename T>
      inline bool operator!=(const T& t, const blas_complex<T>& r) throw(uncomparable)
      { return (r.c[0] != t && r.c[1] != 0); }
    
    
    template <typename T>
      inline bool operator>=(const T& t, const blas_complex<T>& r) throw(uncomparable)
      { throw uncomparable("complex numbers cannot be compared"); }
    
    
    template <typename T>
      inline bool operator<=(const T& t, const blas_complex<T>& r) throw(uncomparable)
      { throw uncomparable("complex numbers cannot be compared"); }
    
    
    template <typename T>
      inline bool operator< (const T& t, const blas_complex<T>& r) throw(uncomparable)
      { throw uncomparable("complex numbers cannot be compared"); }
    
    
    template <typename T>
      inline bool operator> (const T& t, const blas_complex<T>& r) throw(uncomparable)
      { throw uncomparable("complex numbers cannot be compared"); }

    
    
    ////////////////////////////////////////////////////////////////////////////////
    // some basic math functions are in "blade_math.h"
    
    
    template <typename T>
      inline blas_real<T> sqrt(const blas_real<T>& a)
      {
	return blas_real<T>( T(::sqrt(a.c[0])) );
      }
    
    
    template <typename T>
      inline blas_complex<T> sqrt(const blas_complex<T>& a)
      {
	std::complex<T> t(a.c[0], a.c[1]);
	
	return blas_complex<T>( std::sqrt(t) );
      }
    
    
    

    
    template <typename T>
      inline blas_real<T> abs(const blas_real<T>& t) throw()
      {
	blas_real<T> u;
	u.c[0] = T(::fabs((double)t.c[0]));
	return u;
      }
    
    template <typename T>
      inline blas_real<T> abs(const blas_complex<T>& t) throw()
      {
	blas_real<T> u;
	
	u.c[0] = T(::sqrt((double)(t.c[0]*t.c[0] + t.c[1] + t.c[1])));
	return u;
      }
    
    
    // double fabs(const double& t) throw();
    // float fabs(const float& t) throw();
    
    
    //////////////////////////////////////////////////////////////////////
    // conversion functions

    
    // tries to convert blas_real of type S to blas_real of type T (B = A)
    template <typename T, typename S>
      inline bool convert(blas_real<T>& B, const blas_real<S>& A) throw()
      {
	try{ B.c[0] = static_cast<T>(A.c[0]); return true; }
	catch(std::exception& e){ return false; }
      }
    
    // tries to convert blas_complex of type S to blas_complex of type T (B = A)
    template <typename T, typename S>
      inline bool convert(blas_complex<T>& B, const blas_complex<S>& A) throw()
      {
	try{
	  B.c[0] = static_cast<T>(A.c[0]);	
	  B.c[1] = static_cast<T>(A.c[1]);
	  return true;
	}
	catch(std::exception& e){ return false; }
      }
    
    
    // tries to convert blas_real of type S to scalar of type T
    template <typename T, typename S>
      inline bool convert(T& B, const blas_real<S>& A) throw()
      {
	try{ B = (T)(A.c[0]); return true; }
	catch(std::exception& e){ return false; }
      }

    
    

#else

#error "No packed BLAS primitives specified for this compiler."
    
#endif
    
    template <typename T>
      inline std::ostream& operator<<(std::ostream& ios,
				      const blas_real<T>& r){
      ios << r.c[0];
      return ios;
    }
    
  
    template <typename T>
      inline std::ostream& operator<<(std::ostream& ios,
				      const blas_complex<T>& r){
      ios << r.c[0] << " + ";
      ios << r.c[1] << "i";
      return ios;
    }
    
    
  };
};


#endif
