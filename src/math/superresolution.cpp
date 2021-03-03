
#ifndef superresolution_cpp
#define superresolution_cpp

#include "superresolution.h"
#include "modular.h"
#include "blade_math.h"


namespace whiteice
{
  namespace math
  {

    template <typename T, typename U>
    superresolution<T,U>::superresolution()
    {
      // initializes one dimensional scaling basis with zero value
      
      for(unsigned int i=0;i<this->size();i++)
	basis[i] = T(0);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>::superresolution(const T value)
    {
      for(unsigned int i=1;i<this->size();i++)
	basis[i] = T(0);

      basis[0] = value;
    }

    template <typename T, typename U>
    superresolution<T,U>::superresolution(const superresolution<T,U>& s)
    {
      for(unsigned int i=0;i<size();i++)
	this->basis[i] = s.basis[i];
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>::superresolution(const std::vector<T>& values)
    {
      // initializes values.size() - dimensional basis
      if(values.size() != this->size())
	throw illegal_operation("Incorrect basis size in input vector");

      for(unsigned int i=0;i<size();i++)
	basis[i] = values[i];
      
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

      U u = U(0);
      t.basis[u[0]] += s.basis[U(0)[0]];

      for(U u=U(1);u != U(0);u++)
	t.basis[u[0]] += s.basis[u[0]];

      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator-(const superresolution<T,U>& s) const
      
    {
      superresolution<T,U> t(*this);
      
      U u = U(0);
      t.basis[u[0]] -= s.basis[u[0]];
      
      for(U u=U(1);u != U(0);u++)
	t.basis[u[0]] -= s.basis[u[0]];
      
      return t;      
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator*(const superresolution<T,U>& s) const
      
    {
      {
	if(this->size() != s.size())
	  throw illegal_operation("Not same basis");

	// don't check basis dimension is PRIME
	
	if(s.iszero())
	  throw illegal_operation("division by zero");
      }

      // z = convolution(x, y)
      // z = InvFFT(FFT(x)*FFT(y))
      
      whiteice::math::vertex< whiteice::math::blas_complex<float> > b1, b2;
      b1.resize(this->size());
      b2.resize(this->size());

      for(unsigned int i=0;i<b1.size();i++){
	//b1[i] = this->basis[i];
	//b2[i] = s.basis[i];

	whiteice::math::convert(b1[i], this->basis[i]);
	whiteice::math::convert(b2[i], s.basis[i]);
      }

      const unsigned int K = DEFAULT_MODULAR_EXP;

      // calculates FFT of convolutions
      if(whiteice::math::fft<K, float >(b1) == false ||
	 whiteice::math::fft<K, float >(b2) == false)
	throw illegal_operation("FFT failed");

      // inverse computation of convolution
      for(unsigned int i=0;i<DEFAULT_MODULAR_BASIS;i++){
	b1[i] *= b2[i];
      }

      // inverse FFT
      if(whiteice::math::ifft<K, float >(b1) == false)
	throw illegal_operation("Inverse FFT failed");

      superresolution<T,U> result(T(0));
      
      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(result.basis[i], b1[i]);
	// result.basis[i] = T(b1[i]);
      }

      return result;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator/(const superresolution<T,U>& s) const
      
    {
      {
	if(this->size() != s.size())
	  throw illegal_operation("Not same basis");

	// don't check basis dimension is PRIME
	
	if(s.iszero())
	  throw illegal_operation("division by zero");
      }

      // z = convolution(x, (1/y))
      // z = InvFFT(FFT(x)/FFT(y))
      
      whiteice::math::vertex< whiteice::math::blas_complex<float> > b1, b2;
      b1.resize(this->size());
      b2.resize(this->size());

      for(unsigned int i=0;i<b1.size();i++){
	//b1[i] = this->basis[i];
	//b2[i] = s.basis[i];

	whiteice::math::convert(b1[i], this->basis[i]);
	whiteice::math::convert(b2[i], s.basis[i]);
      }

      const unsigned int K = DEFAULT_MODULAR_EXP;

      // calculates FFT of convolutions
      if(whiteice::math::fft<K, float >(b1) == false ||
	 whiteice::math::fft<K, float >(b2) == false)
	throw illegal_operation("FFT failed");

      // inverse computation of convolution
      for(unsigned int i=0;i<b1.size();i++){
	if(b2[i] != whiteice::math::blas_complex<float>(0.0f)){
	  b1[i] = b1[i] / b2[i];
	}
      }

      // inverse FFT
      if(whiteice::math::ifft<K, float >(b1) == false)
	throw illegal_operation("Inverse FFT failed");

      superresolution<T,U> result(T(0));
      
      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(result.basis[i], b1[i]);
	// result.basis[i] = T(b1[i]);
      }

      return result;
    }
    
    
    // complex conjugate
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator!() const 
      
    {
      // conjugates both numbers and basises            
      
      superresolution<T,U> s(T(0)); // s = 0 vector initially
      superresolution<T,U> t(T(0));

      t.basis[U(0)[0]] = whiteice::math::conj(this->basis[U(0)[0]]);
      
      // numbers + inits basis
      for(U u=U(1);u!=U(0);u++){
	t.basis[u[0]] = whiteice::math::conj(this->basis[u[0]]);
      }
      
      // basis
      s.basis[whiteice::math::conj(U(0))[0]] = t.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	s.basis[whiteice::math::conj(u)[0]] += t.basis[u[0]];
      }
      
      return s;
    }


    // complex conjugate
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::conj()
    {
      // conjugates both numbers and basises            
      
      superresolution<T,U> t(T(0));

      t.basis[U(0)[0]] = whiteice::math::conj(this->basis[U(0)[0]]);
      
      // numbers + inits basis
      for(U u=U(1);u!=U(0);u++){
	t.basis[u[0]] = whiteice::math::conj(this->basis[u[0]]);
      }
      
      // basis
      this->basis[whiteice::math::conj(U(0)[0])] = t.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	this->basis[whiteice::math::conj(u[0])] += t.basis[u[0]];
      }
      
      return (*this);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U> superresolution<T,U>::operator-() const
      
    {
      superresolution<T,U> s(*this);

      s.basis[U(0)[0]] = -s.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	s.basis[u[0]] = -(s.basis[u[0]]);
      }
      
      return s;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator+=(const superresolution<T,U>& s)
      
    {
      this->basis[U(0)[0]] += s.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	this->basis[u[0]] += s.basis[u[0]];
      }
      
      return *this;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator-=(const superresolution<T,U>& s)
      
    {
      this->basis[U(0)[0]] -= s.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	this->basis[u[0]] -= s.basis[u[0]];
      }
      
      return *this;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator*=(const superresolution<T,U>& s)
      
    {
      {
	if(this->size() != s.size())
	  throw illegal_operation("Not same basis");
	
	// don't check basis dimension is PRIME
	
	if(s.iszero())
	  throw illegal_operation("division by zero");
      }

      // z = convolution(x, y)
      // z = InvFFT(FFT(x)*FFT(y))
      
      whiteice::math::vertex< whiteice::math::blas_complex<float> > b1, b2;
      b1.resize(this->size());
      b2.resize(this->size());

      for(unsigned int i=0;i<b1.size();i++){
	//b1[i] = this->basis[i];
	//b2[i] = s.basis[i];

	whiteice::math::convert(b1[i], this->basis[i]);
	whiteice::math::convert(b2[i], s.basis[i]);
      }

      const unsigned int K = DEFAULT_MODULAR_EXP;

      // calculates FFT of convolutions
      if(whiteice::math::fft<K, float >(b1) == false ||
	 whiteice::math::fft<K, float >(b2) == false)
	throw illegal_operation("FFT failed");

      // inverse computation of convolution
      for(unsigned int i=0;i<DEFAULT_MODULAR_BASIS;i++){
	b1[i] *= b2[i];
      }

      // inverse FFT
      if(whiteice::math::ifft<K, float >(b1) == false)
	throw illegal_operation("Inverse FFT failed");

      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(this->basis[i], b1[i]);
	// result.basis[i] = T(b1[i]);
      }

      return (*this);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator/=(const superresolution<T,U>& s)
      
    {
      {
	if(this->size() != s.size())
	  throw illegal_operation("Not same basis");

	// don't check basis dimension is PRIME
	
	if(s.iszero())
	  throw illegal_operation("division by zero");
      }

      // z = convolution(x, (1/y))
      // z = InvFFT(FFT(x)/FFT(y))
      
      whiteice::math::vertex< whiteice::math::blas_complex<float> > b1, b2;
      b1.resize(this->size());
      b2.resize(this->size());

      for(unsigned int i=0;i<b1.size();i++){
	//b1[i] = this->basis[i];
	//b2[i] = s.basis[i];

	whiteice::math::convert(b1[i], this->basis[i]);
	whiteice::math::convert(b2[i], s.basis[i]);
      }

      const unsigned int K = DEFAULT_MODULAR_EXP;

      // calculates FFT of convolutions
      if(whiteice::math::fft<K, float >(b1) == false ||
	 whiteice::math::fft<K, float >(b2) == false)
	throw illegal_operation("FFT failed");

      // inverse computation of convolution
      for(unsigned int i=0;i<b1.size();i++){
	if(b2[i] != whiteice::math::blas_complex<float>(0))
	  b1[i] /= b2[i];
      }

      // inverse FFT
      if(whiteice::math::ifft<K, float >(b1) == false)
	throw illegal_operation("Inverse FFT failed");

      for(unsigned int i=0;i<b1.size();i++){
	whiteice::math::convert(this->basis[i], b1[i]);
      }

      return (*this);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator=(const superresolution<T,U>& s)
      
    {
      this->basis[U(0)[0]] = s.basis[U(0)[0]];
      
      for(U u=U(1);u!=U(0);u++){
	this->basis[u[0]] = s.basis[u[0]];
      }
      
      return *this;
    }

    
    template <typename T, typename U>
    bool superresolution<T,U>::operator==(const superresolution<T,U>& s) const 
      
    
    {
      if(this->size() != s.size()){
	return false;
      }

      if(this->basis[U(0)[0]] != s.basis[U(0)[0]])
	return false;
      
      for(U u=U(1);u!=U(0);u++){
	if(this->basis[u[0]] != s.basis[u[0]])
	  return false;
      }
      
      return true;
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator!=(const superresolution<T,U>& s) const 
      
    {
      return (!((*this) == s));
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator>=(const superresolution<T,U>& s) const
      
    {
      if(s.size() != this->size())
	throw uncomparable("Non same baisis size numbers are uncomparable");

      std::cout << "sr:>=" << std::endl;

      U u = U(this->size()-1);

      do{
	if(basis[u[0]] > s.basis[u[0]]){
	  return true;
	}
	else if(basis[u[0]] < s.basis[u[0]]){
	  return false;
	}
	
	u--;
      }
      while(u != U(0));

      return (this->basis[0] >= s.basis[0]);
    }
    
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator<=(const superresolution<T,U>& s) const 
    {
      if(s.size() != this->size())
	throw uncomparable("Non same basis size numbers are uncomparable");

      std::cout << "sr:<=" << std::endl;

      U u = U(this->size()-1);

      do{
	if(basis[u[0]] < s.basis[u[0]]){
	  return true;
	}
	else if(basis[u[0]] > s.basis[u[0]]){
	  return false;
	}
	
	u--;
      }
      while(u != U(0));
      
      return (this->basis[0] <= s.basis[0]);
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator< (const superresolution<T,U>& s) const 
      
    {
      if(s.size() != this->size())
	throw uncomparable("Non same basis basis size numbers are uncomparable");

      std::cout << "sr:<" << std::endl;
      
      U u = U(this->size()-1);

      do{
	if(basis[u[0]] < s.basis[u[0]]){
	  return true;
	}
	else if(basis[u[0]] > s.basis[u[0]]){
	  return false;
	}
	
	u--;
      }
      while(u != U(0));

      return (this->basis[0] < s.basis[0]);
    }
    
    
    template <typename T, typename U>
    bool superresolution<T,U>::operator> (const superresolution<T,U>& s) const 
      
    {
      if(s.size() != this->size())
	throw uncomparable("Non same basis size numbers are uncomparable");

      std::cout << "sr:>" << std::endl;

      U u = U(this->size()-1);
      
      do{
	if(basis[u[0]] > s.basis[u[0]]){
	  return true;
	}
	else if(basis[u[0]] < s.basis[u[0]]){
	  return false;
	}
	
	u--;
      }
      while(u != U(0));


      return (this->basis[0] > s.basis[0]);
    }
    
    
    
    // scalar operation
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator= (const T& s) 
      
    {
      this->zero();

      this->basis[0] = s;

      return (*this);
    }

    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator+ (const T& s) const
    {
      superresolution<T,U> t(*this);
      
      U i = U(0);
      
      t.basis[i[0]] += s;
      
      return t;
    }

    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator- (const T& s) const
    {
      superresolution<T,U> t(*this);
      
      U i = U(0);
      
      t.basis[i[0]] -= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator* (const T& s) const 
      
    {
      superresolution<T,U> t(*this);

      U i = U(0);
      
      t.basis[i[0]] *= s;

      for(U i=U(1);i != U(0);i++)
	t.basis[i[0]] *= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator/ (const T& s) const 
    {
      superresolution<T,U> t(*this);

      t.basis[U(0)[0]] /= s;
      
      for(U i=U(1);i!=U(0);i++)
	t.basis[i[0]] /= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator*=(const T& s) 
    {
      superresolution<T,U>& t = (*this);

      t.basis[U(0)[0]] *= s;
      
      for(U i=U(1);i!=U(0);i++)
	t.basis[i[0]] *= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator/=(const T& s)
      
    {
      superresolution<T,U>& t = (*this);

      t.basis[U(0)[0]] /= s;
      
      for(U i=U(1);i!=U(0);i++)
	t.basis[i[0]] /= s;
      
      return t;
    }


    // scalar operation
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator= (const T s) 
      
    {
      this->zero();

      this->basis[0] = s;

      return (*this);
    }
    

#if 0
    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator* (const T s) const 
      
    {
      superresolution<T,U> t(*this);

      U i = U(0);
      
      t.basis[i[0]] *= s;

      for(U i=U(1);i != U(0);i++)
	t.basis[i[0]] *= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>  superresolution<T,U>::operator/ (const T s) const 
    {
      superresolution<T,U> t(*this);

      t.basis[U(0)[0]] /= s;
      
      for(U i=U(1);i!=U(0);i++)
	t.basis[i[0]] /= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator*=(const T s) 
    {
      superresolution<T,U>& t = (*this);

      t.basis[U(0)[0]] *= s;
      
      for(U i=U(1);i!=U(0);i++)
	t.basis[i[0]] *= s;
      
      return t;
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::operator/=(const T s)
      
    {
      superresolution<T,U>& t = (*this);

      t.basis[U(0)[0]] /= s;
      
      for(U i=U(1);i!=U(0);i++)
	t.basis[i[0]] /= s;
      
      return t;
    }
#endif
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::abs()
    {
      class whiteice::math::superresolution<T,U>& z = (*this);
      class whiteice::math::superresolution<T,U> zc = (*this);

      {
	// conjugates both numbers and basises            
	
	zc.basis[U(0)[0]] = whiteice::math::conj(zc.basis[U(0)[0]]);
	
	// numbers + inits basis
	for(U u=U(1);u!=U(0);u++){
	  zc.basis[u[0]] = whiteice::math::conj(zc.basis[u[0]]);
	}

	// DO NOT CALCULATE COMPLEX CONJUGATE OF BASIS FUNCTION!
	
      }

      // zc.conj;

      z *= zc;
	
      return z;
    }


    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::zero()
    {
      for(unsigned int i=0;i<this->size();i++)
	basis[i] = T(0);
      
      return (*this);
    }


    template <typename T, typename U>
    bool superresolution<T,U>::iszero() const
    {
      if(whiteice::math::abs(this->basis[U(0)[0]]) != T(0)) return false;
      
      for(U i=U(1);i != U(0);i++)
	if(whiteice::math::abs(this->basis[i[0]]) != T(0)) return false;
      
      return true;
    }
    
    
    template <typename T, typename U>
    T& superresolution<T,U>::operator[](const U& index)
      
    {
      return basis[index[0]];
    }
    
    
    template <typename T, typename U>
    const T& superresolution<T,U>::operator[](const U& index) const
      
    {
      return basis[index[0]];
    }
    
    
    // scales basis - not numbers
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::basis_scaling(const T& s) 
    {
      for(U i=U(1);i != U(0);i++){
	T scaling = whiteice::math::pow(s, T(i[0]));
	this->basis[i[0]] *= scaling;
      }

      return (*this);
    }
    
    
    template <typename T, typename U>
    superresolution<T,U>& superresolution<T,U>::basis_scaling(const std::vector<T>& s)   // non-uniform scaling
    {
      if(this->size() != s.size())
	throw uncomparable("Number basises don't match");

      for(U i=U(1);i != U(0);i++){
	T scaling = whiteice::math::pow(s[i[0]], T(i[0]));
	this->basis[i[0]] *= scaling;
      }
      
      return (*this);
    }
    
    
    // measures with s-(dimensional) measure-function
    template <typename T, typename U>
    T superresolution<T,U>::measure(const U& s) const
    {
      if(s[0] == this->size()-1){
	return T(basis[s[0]]);
      }
      else{
	for(U i=U(this->size()-1);i != U(s);i--)
	  if(basis[i[0]] != T(0)){
	    if(basis[i[0]] < T(0)) return T(-INFINITY);
	    else return T(INFINITY);
	  }

	return basis[s[0]];
      }
    }

    
    template <typename T, typename U>
    unsigned int superresolution<T,U>::size() const {
      return DEFAULT_MODULAR_BASIS;
    }
    
  }
}

namespace whiteice
{
  namespace math
  {
    template <typename T>
    std::ostream& operator<<(std::ostream& ios,
			     const superresolution<T, modular<unsigned int> > & m)
    {
      if(m.size() == 1){
	ios << m[0];
      }
      else if(m.size() > 1){
	ios << "(";

	for(unsigned int i=0;i<m.size();i++){
	  if(i == 0) ios << m[i];
	  else ios << " " << m[i];
	}

	ios << ")";
      }
      
      return ios;
    }

    
    // DO NOT USE BLAS_REAL BUT BLAS_COMPLEX
    template class superresolution< whiteice::math::blas_real<float>,
				    whiteice::math::modular<unsigned int> >;
    template class superresolution< whiteice::math::blas_real<double>,
				    whiteice::math::modular<unsigned int> >;
    
    template class superresolution< whiteice::math::blas_complex<float>,
				    whiteice::math::modular<unsigned int> >;
    template class superresolution< whiteice::math::blas_complex<double>,
				    whiteice::math::modular<unsigned int> >;

    template std::ostream& operator<< <whiteice::math::blas_complex<float> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_complex<float>,
     whiteice::math::modular<unsigned int> >&);

    template std::ostream& operator<< <whiteice::math::blas_complex<double> >
    (std::ostream& ios,
     const whiteice::math::superresolution< whiteice::math::blas_complex<double>,
     whiteice::math::modular<unsigned int> >&);
    
  };
};


#endif
