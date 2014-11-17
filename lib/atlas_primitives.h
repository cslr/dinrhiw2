/*
 * atlas real and complex number
 * template data primitives
 */

#ifndef atlas_primitives_h
#define atlas_primitives_h

#include "ownexception.h"
#include <stdexcept>
#include <exception>
#include <complex>


namespace whiteice
{
  namespace math
  {
    
#ifdef __GNUG__
    // gnu c++ specific hack to keep sizeof(atlas_real<T>) == sizeof(T)
    // (so there's no vtable, padding etc. to enlarge the 'struct'/class)
    // -> can malloc and use atlas_xxx structures as pure types
    //    -> memcpy((T*)(atlas_real<T>_pointer), xxx) etc.
    //    -> especially now ATLAS library can access data directly (as memory)
    
    template <typename T>
      struct atlas_complex;

    
    template <typename T=float>
      struct atlas_real
      {
	T c[1] __attribute__ ((packed));
	
	inline atlas_real(){ c[0] = (T)0; }
	inline atlas_real(const T& t){ c[0] = t; }
	inline atlas_real(const atlas_real<T>& t){ c[0] = t.c[0]; }

	// work arounds stupid compiler..
	// explicit inline atlas_real(const atlas_complex<float>& t){ c[0] = t.c[0]; } // takes real part
	// explicit inline atlas_real(const atlas_complex<double>& t){ c[0] = t.c[0]; } // takes real part
	
	inline ~atlas_real(){ }
	
	inline atlas_real<T> operator++(int n){
	  if(n) c[0] += T(n);
	  else c[0]++;
	  
	  return (*this);
	}
	
	inline atlas_real<T> operator+(const atlas_real<T>& t) const throw(illegal_operation)
	{ return atlas_real<T>(this->c[0] + t.c[0]); }
	
	inline atlas_real<T> operator-(const atlas_real<T>& t) const throw(illegal_operation)
	{ return atlas_real<T>(this->c[0] - t.c[0]); }
	
	inline atlas_real<T> operator*(const atlas_real<T>& t) const throw(illegal_operation)
	{ return atlas_real<T>(this->c[0] * t.c[0]); }
	     
	inline atlas_real<T> operator/(const atlas_real<T>& t) const throw(illegal_operation)
	{ return atlas_real<T>(this->c[0] / t.c[0]); } // no division by zero checks
	  
	inline atlas_real<T> operator!() const throw(illegal_operation) // complex conjugate
	{ return *this;}
	
	inline atlas_real<T> operator-() const throw(illegal_operation)
	{ return atlas_real<T>(-this->c[0]); }
      
	inline atlas_real<T>& operator+=(const atlas_real<T>& t) throw(illegal_operation)
	{ this->c[0] += t.c[0]; return *this; }
	     
	inline atlas_real<T>& operator-=(const atlas_real<T>& t) throw(illegal_operation)
	{ this->c[0] -= t.c[0]; return *this; }
	
	inline atlas_real<T>& operator*=(const atlas_real<T>& t) throw(illegal_operation)
	{ this->c[0] *= t.c[0]; return *this; }
	     
	inline atlas_real<T>& operator/=(const atlas_real<T>& t) throw(illegal_operation)
	{ this->c[0] /= t.c[0]; return *this; } // no division by zero checks
	  
	inline atlas_real<T>& operator=(const atlas_real<T>& t) throw(illegal_operation)
	{ this->c[0] = t.c[0]; return *this; }
	
	inline bool operator==(const atlas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] == t.c[0]); }
	     
	inline bool operator!=(const atlas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] != t.c[0]); }
	     
	inline bool operator>=(const atlas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] >= t.c[0]); }
	     
	inline bool operator<=(const atlas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] <= t.c[0]); }
	     
	inline bool operator< (const atlas_real<T>& t) const throw(uncomparable)
	{ return (this->c[0] < t.c[0]); }
	     
	inline bool operator> (const atlas_real<T>& t) const throw(uncomparable)
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
	inline atlas_real<T>& operator= (const T& s) throw(illegal_operation)
	{ this->c[0] = s; return *this; }

	inline atlas_real<T> operator+=(const T& s) throw(illegal_operation)
	{ this->c[0] += s; return *this; } 
	  
	inline atlas_real<T>& operator-=(const T& s) throw(illegal_operation)
	{ this->c[0] -= s; return *this; }
	     
	inline atlas_real<T>  operator* (const T& s) const throw()
	{ atlas_real<T> r; r.c[0] = s * this->c[0]; return r; }
	     
	inline atlas_real<T>  operator/ (const T& s) const throw(std::invalid_argument)
	{ atlas_real<T> r; r.c[0] =  this->c[0] / s; return r; } // no division by zero checks
	  
	inline atlas_real<T>& operator*=(const T& s) throw()
	{ this->c[0] *= s; return *this; }
	     
	inline atlas_real<T>& operator/=(const T& s) throw(std::invalid_argument)
	{ this->c[0] /= s; return *this; }
	     
	inline atlas_real<T> abs() const
	{ return atlas_real<T>( T(fabs((double)c[0])) ); }
	
	/*
	 * inline T& operator[](unsigned int index) throw(std::out_of_range, illegal_operation)
	 * { return c[0]; } // doesn't use index!
	 * 
	 * inline const T& operator[](unsigned int index) const throw(std::out_of_range, illegal_operation)
	 * { return c[0]; } // doesn't use index!
	 *
	 */
	
	inline T& real() throw(){ return c[0]; }
	inline const T& real() const throw(){ return c[0]; }
	
	inline T& value() throw(){ return c[0]; }
	inline const T& value() const throw(){ return c[0]; }
	
	
	template <typename A>
	friend atlas_real<A> operator*(const A& s, const atlas_real<A>& r) throw(std::invalid_argument);
	
	template <typename A>
	friend atlas_real<A> operator/(const A& s, const atlas_real<A>& r) throw(std::invalid_argument);
	
	template <typename A>
	friend bool operator==(const A& t, const atlas_real<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator!=(const A& t, const atlas_real<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator>=(const A& t, const atlas_real<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator<=(const A& t, const atlas_real<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator< (const A& t, const atlas_real<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator> (const A& t, const atlas_real<A>& r) throw(uncomparable);
	
      } __attribute__ ((packed));
    
    
    
    template <typename T>
      inline atlas_real<T> operator*(const T& s, const atlas_real<T>& r) throw(std::invalid_argument)
      {
	return atlas_real<T>(r * s);
      }
    
    template <typename T>
      inline atlas_real<T> operator/(const T& s, const atlas_real<T>& r) throw(std::invalid_argument)
      {
	return atlas_real<T>(atlas_real<T>(s) / r);
      }
    
    
    template <typename T>
      inline bool operator==(const T& t, const atlas_real<T>& r) throw(uncomparable)
      { return (r.c[0] == t); }
    
    
    template <typename T>
      inline bool operator!=(const T& t, const atlas_real<T>& r) throw(uncomparable)
      { return (r.c[0] != t); }
    
    
    template <typename T>
      inline bool operator>=(const T& t, const atlas_real<T>& r) throw(uncomparable)
      { return (t >= r.c[0]); }
    
    
    template <typename T>
      inline bool operator<=(const T& t, const atlas_real<T>& r) throw(uncomparable)
      { return (t <= r.c[0]); }
    
    
    template <typename T>
      inline bool operator< (const T& t, const atlas_real<T>& r) throw(uncomparable)
      { return (t < r.c[0]); }
    
    
    template <typename T>
      inline bool operator> (const T& t, const atlas_real<T>& r) throw(uncomparable)
      { return (t > r.c[0]); }
    
    
    
    
    
    
    
    template <typename T=float>
      struct atlas_complex
      {
	T c[2] __attribute__ ((packed));
	
	
	inline atlas_complex(){ c[0] = T(0); c[1] = T(0); }
	inline atlas_complex(const T& r){ c[0] = r; c[1] = T(0); }
	inline atlas_complex(const T& r, const T& i){ c[0] = r; c[1] = i; }
	inline atlas_complex(const atlas_complex<T>& r){ c[0] = r.c[0]; c[1] = r.c[1]; }
	inline atlas_complex(const std::complex<T>& z){ c[0] = std::real(z); c[1] = std::imag(z); }
	  
	// work arounds stupid compiler..
	inline atlas_complex(const atlas_real<float>& r){ c[0] = r.c[0]; c[1] = T(0); }
	inline atlas_complex(const atlas_real<double>& r){ c[0] = r.c[0]; c[1] = T(0); }
	
	inline ~atlas_complex(){ }
	
	
	inline atlas_complex<T> operator+(const atlas_complex<T>& t) const throw(illegal_operation)
	{ return atlas_complex<T>(this->c[0] + t.c[0], this->c[1] + t.c[1]); }
	
	inline atlas_complex<T> operator-(const atlas_complex<T>& t) const throw(illegal_operation)
	{ return atlas_complex<T>(this->c[0] - t.c[0], this->c[1] - t.c[1]); }
	
	inline atlas_complex<T> operator*(const atlas_complex<T>& t) const throw(illegal_operation)
	{ return atlas_complex<T>(this->c[0] * t.c[0] - this->c[1]*t.c[1],
				  this->c[1] * t.c[0] + this->c[0]*t.c[1]); }
	
	// no division by zero checks
	inline atlas_complex<T> operator/(const atlas_complex<T>& t) const throw(illegal_operation)
	{ atlas_complex<T> r; r.c[0] = (c[0]*t.c[0] + c[1]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]);
	  r.c[1] = (c[1]*t.c[0] - c[0]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]); return r;}
	
	inline atlas_complex<T> operator!() const throw(illegal_operation)  // complex conjugate
	{ return atlas_complex<T>(this->c[0], -this->c[1]); }
	
	inline atlas_complex<T> operator-() const throw(illegal_operation)
	{ return atlas_complex<T>(-this->c[0], -this->c[1]); }
      
	inline atlas_complex<T>& operator+=(const atlas_complex<T>& t) throw(illegal_operation)
	{ this->c[0] += t.c[0]; this->c[1] += t.c[1]; return *this; }
	     
	inline atlas_complex<T>& operator-=(const atlas_complex<T>& t) throw(illegal_operation)
	{ this->c[0] -= t.c[0]; this->c[1] -= t.c[1]; return *this; }
	
	inline atlas_complex<T>& operator*=(const atlas_complex<T>& t) throw(illegal_operation)
	{ T a = c[0] * t.c[0] - c[1]*t.c[1]; T b = c[1] * t.c[0] + c[0]*t.c[1];
	  this->c[0] = a; this->c[1] = b; return *this; }
	  
	// no division by zero checks
	inline atlas_complex<T>& operator/=(const atlas_complex<T>& t) throw(illegal_operation)
	{ T a = (c[0]*t.c[0] + c[1]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]);
	  T b = (c[1]*t.c[0] - c[0]*t.c[1])/(t.c[0]*t.c[0] + t.c[1]*t.c[1]);
	  this->c[0] = a; this->c[1] = b; return *this; }	  
	  
	inline atlas_complex<T>& operator=(const atlas_complex<T>& t) throw(illegal_operation)
	{ this->c[0] = t.c[0]; this->c[1] = t.c[1]; return *this; }
	  
	inline atlas_complex<T>& operator=(const atlas_real<T>& t) throw(illegal_operation)
	{ this->c[0] = t.c[0]; this->c[1] = T(0.0); return *this; }
	
	inline bool operator==(const atlas_complex<T>& t) const throw(uncomparable)
	{ return (this->c[0] == t.c[0] && this->c[1] == t.c[1]); }
	     
	inline bool operator!=(const atlas_complex<T>& t) const throw(uncomparable)
	{ return (this->c[0] != t.c[0] && this->c[1] != t.c[1]); }
	     
	inline bool operator>=(const atlas_complex<T>& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	     
	inline bool operator<=(const atlas_complex<T>& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	     
	inline bool operator< (const atlas_complex<T>& t) const throw(uncomparable)
	{ throw uncomparable("complex numbers cannot be compared"); }
	     
	inline bool operator> (const atlas_complex<T>& t) const throw(uncomparable)
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
	inline atlas_complex<T>& operator= (const T& s) throw(illegal_operation)
	{ this->c[0] = s; this->c[1] = T(0); return *this; }
	
	inline atlas_real<T> operator+=(const T& s) throw(illegal_operation)
	{ this->c[0] += s; return *this; }
	
	inline atlas_real<T>& operator-=(const T& s) throw(illegal_operation)
	{ this->c[0] -= s; return *this; }
	     
	inline atlas_complex<T>  operator* (const T& s) const throw()
	{ atlas_complex<T> r; r.c[0] = s * this->c[0]; r.c[1] = s * this->c[1]; return r; }
	
	// no division by zero checks
	inline atlas_complex<T>  operator/ (const T& s) const throw(std::invalid_argument)
	{ atlas_complex<T> r; r.c[0] =  this->c[0] / s; r.c[1] = this->c[1] / s; return r; }
	  
	inline atlas_complex<T>& operator*=(const T& s) throw()
	{ this->c[0] *= s; this->c[1] *= s; return *this; }
	     
	inline atlas_complex<T>& operator/=(const T& s) throw(std::invalid_argument)
	{ this->c[0] /= s; this->c[1] /= s; return *this; }
	     
	inline atlas_real<T> abs() const
	{ atlas_real<T> r; r.c[0] = T(sqrt((double)(c[0]*c[0] + c[1]*c[1]))); return r; }
	
	/*
	 * inline T& operator[](unsigned int index) throw(std::out_of_range, illegal_operation)
	 * { return c[index]; }
	 * 
	 * inline const T& operator[](unsigned int index) const throw(std::out_of_range, illegal_operation)
	 * { return c[index]; }
	 * 
	 */
	
	inline T& real() throw(){ return c[0]; }
	inline const T& real() const throw(){ return c[0]; }
	
	inline T& imag() throw(){ return c[1]; }
	inline const T& imag() const throw(){ return c[1]; }
	
	
	
	template <typename A>
	friend atlas_complex<A> operator*(const A& s, const atlas_complex<A>& r) throw(std::invalid_argument);
	
	template <typename A>
	friend atlas_complex<A> operator/(const A& s, const atlas_complex<A>& r) throw(std::invalid_argument);
	
	template <typename A>
	friend bool operator==(const A& t, const atlas_complex<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator!=(const A& t, const atlas_complex<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator>=(const A& t, const atlas_complex<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator<=(const A& t, const atlas_complex<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator< (const A& t, const atlas_complex<A>& r) throw(uncomparable);
	
	template <typename A>
	friend bool operator> (const A& t, const atlas_complex<A>& r) throw(uncomparable);
	
      } __attribute__ ((packed));
    
    
    

    template <typename T>
      inline atlas_complex<T> operator*(const T& s, const atlas_complex<T>& r) throw(std::invalid_argument)
      {
	return atlas_complex<T>(r * s);
      }
    
    
    template <typename T>
      inline atlas_complex<T> operator/(const T& s, const atlas_complex<T>& r) throw(std::invalid_argument)
      {
	return atlas_complex<T>(atlas_complex<T>(s) / r);
      }
    
    
    template <typename T>
      inline bool operator==(const T& t, const atlas_complex<T>& r) throw(uncomparable)
      { return (r.c[0] == t && r.c[1] == 0); }
    
    
    template <typename T>
      inline bool operator!=(const T& t, const atlas_complex<T>& r) throw(uncomparable)
      { return (r.c[0] != t && r.c[1] != 0); }
    
    
    template <typename T>
      inline bool operator>=(const T& t, const atlas_complex<T>& r) throw(uncomparable)
      { throw uncomparable("complex numbers cannot be compared"); }
    
    
    template <typename T>
      inline bool operator<=(const T& t, const atlas_complex<T>& r) throw(uncomparable)
      { throw uncomparable("complex numbers cannot be compared"); }
    
    
    template <typename T>
      inline bool operator< (const T& t, const atlas_complex<T>& r) throw(uncomparable)
      { throw uncomparable("complex numbers cannot be compared"); }
    
    
    template <typename T>
      inline bool operator> (const T& t, const atlas_complex<T>& r) throw(uncomparable)
      { throw uncomparable("complex numbers cannot be compared"); }

    
    
    ////////////////////////////////////////////////////////////////////////////////
    // some basic math functions are in "blade_math.h"
    
    
    template <typename T>
      inline atlas_real<T> sqrt(const atlas_real<T>& a)
      {
	return atlas_real<T>( T(::sqrt(a.c[0])) );
      }
    
    
    template <typename T>
      inline atlas_complex<T> sqrt(const atlas_complex<T>& a)
      {
	std::complex<T> t(a.c[0], a.c[1]);
	
	return atlas_complex<T>( std::sqrt(t) );
      }
    
    
    

    
    template <typename T>
      inline atlas_real<T> abs(const atlas_real<T>& t) throw()
      {
	atlas_real<T> u;
	u.c[0] = T(::fabs((double)t.c[0]));
	return u;
      }
    
    template <typename T>
      inline atlas_real<T> abs(const atlas_complex<T>& t) throw()
      {
	atlas_real<T> u;
	
	u.c[0] = T(::sqrt((double)(t.c[0]*t.c[0] + t.c[1] + t.c[1])));
	return u;
      }
    
    
    // double fabs(const double& t) throw();
    // float fabs(const float& t) throw();
    
    
    //////////////////////////////////////////////////////////////////////
    // conversion functions

    
    // tries to convert atlas_real of type S to atlas_real of type T (B = A)
    template <typename T, typename S>
      inline bool convert(atlas_real<T>& B, const atlas_real<S>& A) throw()
      {
	try{ B.c[0] = dynamic_cast<T>(A.c[0]); return true; }
	catch(std::exception& e){ return false; }
      }
    
    // tries to convert atlas_complex of type S to atlas_complex of type T (B = A)
    template <typename T, typename S>
      inline bool convert(atlas_complex<T>& B, const atlas_complex<S>& A) throw()
      {
	try{
	  B.c[0] = dynamic_cast<T>(A.c[0]);	
	  B.c[1] = dynamic_cast<T>(A.c[1]);
	  return true;
	}
	catch(std::exception& e){ return false; }
      }
    
    
    // tries to convert atlas_real of type S to scalar of type T
    template <typename T, typename S>
      inline bool convert(T& B, const atlas_real<S>& A) throw()
      {
	try{ B = (T)(A.c[0]); return true; }
	catch(std::exception& e){ return false; }
      }

    
    

#else

#error "No packed ATLAS primitives specified for this compiler."
    
#endif
    
    template <typename T>
      inline std::ostream& operator<<(std::ostream& ios,
				      const atlas_real<T>& r){
      ios << r.c[0];
      return ios;
    }
    
  
    template <typename T>
      inline std::ostream& operator<<(std::ostream& ios,
				      const atlas_complex<T>& r){
      ios << r.c[0] << " + ";
      ios << r.c[1] << "i";
      return ios;
    }
    
    
  };
};


#endif
