
#ifndef superresolution_cpp
#define superresolution_cpp

#include "superresolution.h"
#include "blade_math.h"


namespace whiteice
{
  namespace math
  {
    
    template <typename T, typename U>
    superresolution<T,U>::superresolution()
    {
      // initializes zero dimensional scaling basis with zero value
      this->basis.resize(0);
      this->basis[U(0)] = T(0);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>::superresolution(const U& resolution)
    {
      this->basis.resize(resolution);      
      for(U u=U(0);u<U(this->basis.size());u++)
	basis[u] = T(0);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>::superresolution(const superresolution<T,U>& s)
    {
      this->basis = s.basis;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>::superresolution(const std::vector<T>& values)
    {
      // initializes values.size() - dimensional basis
      this->basis = values;
    }   
    
    
    template <typename T, typename U>
    superresolution<T,U>::~superresolution()
    {
    }
    
    
    // operators
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator+(const superresolution<T,U>& s)
      const 
    {
      superresolution<T,U> t(*this);
      
      for(U u=U(0);u<U(s.basis.size());u++)
	t.basis[u] += s.basis[u];
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator-(const superresolution<T,U>& s) const
      
    {
      superresolution<T,U> t(*this);
      
      for(U u=U(0);u<U(s.basis.size());u++)
	t.basis[u] -= s.basis[u];
      
      return t;      
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator*(const superresolution<T,U>& s) const
      
    {
      superresolution<T,U> t(U(this->basis.size()));
      
      // note if basis is non-sparse and finite this would be faster
      // with discrete fourier transform (convolution)
      
      for(U u=U(0);u<U(s.basis.size());u++){
	for(U v=U(0);v<U(this->basis.size());v++){
	  t.basis[u+v] += (this->basis[v])*(t.basis[u]);
	}
      }
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator/(const superresolution<T,U>& s) const
      
    {
      {
	U u;
	if(u.comparable() == false){
	  throw illegal_operation("impossible: would need infinite basis");
	}
      }
      
#if 0
      superesolution<T,U> t(U(this->basis.size()));
      
      // possible implementation: fourier transform, inverse
      // (if possible) and inverse fouerier transform
      // (convolution)
#endif
    }
    
    
    // complex conjugate (?)
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator!() const 
      
    {
      // conjugates both numbers and basises            
      
      superresolution<T,U> s(U(this->basis.size()));
      
      // numbers + inits basis
      for(U u=U(0);u<U(this->basis.size());u++){
	this->basis[u] = conj(this->basis[u]);
      }
      
      // basis
      for(U u=U(0);u<U(this->basis.size());u++){
	s.basis[conj(u)] += this->basis[u];
      }
      
      return s;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator-() const
      
    {
      superresolution<T,U> s(*this);
      
      for(U u=U(0);u<U(this->basis.size());u++){
	s.basis[u] = -(s.basis[u]);
      }
      
      return s;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator+=(const superresolution<T,U>& s)
      
    {
      for(U u=U(0);u<U(this->basis.size());u++){
	this->basis[u] += s.basis[u];
      }
      
      return *this;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator-=(const superresolution<T,U>& s)
      
    {
      for(U u=U(0);u<U(this->basis.size());u++){
	this->basis[u] -= s.basis[u];
      }
      
      return *this;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator*=(const superresolution<T,U>& s)
      
    {
      superresolution<T,U> t(*this);
      
      // note if basis is non-sparse and finite this would be faster
      // with discrete fourier transform (convolution)
      
      for(U u=U(0);u<U(t.basis.size());u++){
	for(U v=U(0);v<U(s.basis.size());v++){
	  this->basis[u+v] += (s.basis[v])*(t.basis[u]);
	}
      }
      
      return *this;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator/=(const superresolution<T,U>& s)
      
    {
      // TODO
      
      return *this;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator=(const superresolution<T,U>& s)
      
    {
      this->basis.resize(s.basis.size());
      
      for(U u=U(0);u<U(s.basis.size());u++){
	this->basis[u] = s.basis[u];
      }
      
      return *this;
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator==(const superresolution<T,U>& s) const 
      
    
    {
      if(this->size() != s.size()){		
	U u = U(this->basis.size());
	U v = U(s.basis.size());
	
	if(u > v){
	  std::swap<U>(u,v);
	  
	  for(;u<v;u++)
	    if(s.basis[u] != T(0)) return false;
	}
	else{
	  for(;u<v;u++)
	    if(this->basis[u] != T(0)) return false;
	}
      }
      
      for(U u=U(0);u<U(s.basis.size());u++){
	if(this->basis[u] != s.basis[u])
	  return false;
      }
      
      return true;
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator!=(const superresolution<T,U>& s) const 
      
    {
      
      if(this->size() != s.size()){		
	U u = U(this->basis.size());
	U v = U(s.basis.size());
	
	if(u > v){
	  std::swap<U>(u,v);
	  
	  for(;u<v;u++)
	    if(s.basis[u] != T(0)) return true;
	}
	else{
	  for(;u<v;u++)
	    if(this->basis[u] != T(0)) return true;
	}
      }
      
      for(U u=U(0);u<U(s.basis.size());u++){
	if(this->basis[u] == s.basis[u])
	  return false;
      }
      
      return true;      
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator>=(const superresolution<T,U>& s) const
      
    {
      {
	U u;
	if(u.uncomparable())
	  throw uncomparable("basis exponents are cannot be compared");
	T t;
      }
      
    }
    
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator<=(const superresolution<T,U>& s) const 
      
    {
      // not implemented
      return false;
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator< (const superresolution<T,U>& s) const 
      
    {
      // not implemented
      return false;
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator> (const superresolution<T,U>& s) const 
      
    {
      // not implemented
      return false;
    }
    
    
    
    // scalar operation
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator= (const T& s) 
      
    {
      // not implemented
      return (*this);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator* (const T& s) const 
      
    {
      // not implemented
      return *this;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator/ (const T& s) const 
      
    {
      // not implemented
      return *this;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator*=(const T& s) 
    {
      // not implemented
      return (*this);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator/=(const T& s)
      
    {
      // not implemented
      return (*this);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::abs() 
    {
      // not implemented
      return (*this);
    }
    
    
    template <typename T, typename U>
    T& superresolution<T,U>::operator[](const U& index)
      
    {
      return basis[index];
    }
    
    
    template <typename T, typename U>
    const T& superresolution<T,U>::operator[](const U& index) const
      
    {
      return basis[index];
    }
    
    
    // scales basis - not numbers
    template <typename T, typename U>
    void superresolution<T,U>::basis_scaling(const T& s) 
    {
      // not implemented
    }
    
    
    template <typename T, typename U>
    bool basis_scaling(const std::vector<T>& s)   // non-uniform scaling
    {
      // not implemented
      return false;
    }
    
    
    // measures with s-(dimensional) measure-function
    template <typename T, typename U>
    T superresolution<T,U>::measure(const U& s) 
    {
      // not implemented
      return T(0.0f);
    }
    
  }
}


#endif
