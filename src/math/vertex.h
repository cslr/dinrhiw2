/*
 * atlas vertex
 *  implementation of vertex class
 *  using ATLAS library
 */

#ifndef vertex_h
#define vertex_h

#include "dinrhiw_blas.h"

#include "ownexception.h"
#include "number.h"

#include "compressable.h"
#include "MemoryCompressor.h"

#include <stdexcept>
#include <exception>
#include <iostream>
#include <vector>



namespace whiteice
{
  class SOM2D;
  
  template <typename T> class neuronlayer;
  template <typename T> class backpropagation;
  
  
  namespace math
  {
    template <typename T> class vertex;
    template <typename T> class matrix;
    template <typename T> class quaternion;
    
    
    
    template <typename T> bool gramschmidt
      (matrix<T>& B,const unsigned int i,const unsigned int j);
        
    // correlation
    template <typename T> bool autocorrelation
      (matrix<T>& R, const std::vector< vertex<T> >& data);
    template <typename T> bool autocorrelation
      (matrix<T>& R, const matrix<T>& W);
    template <typename T> bool mean_covariance_estimate
      (vertex<T>& m, matrix<T>& R, const std::vector< vertex<T> >& data);
    
    // rotations
    template <typename T> bool rhouseholder_leftrot
      (matrix<T>& A, const unsigned int i, const unsigned int M, const unsigned int k, vertex<T>& v);
    
    template <typename T> bool rhouseholder_rightrot
      (matrix<T>& A, const unsigned int i, const unsigned int M, const unsigned int k, vertex<T>& v);
    
    
    
    template <typename T=blas_real<float> >
      class vertex : public number<vertex<T>, T, T, unsigned int>
      // multiple-inheritance doesn't work with neuralnetwork/tst/test.cpp
      //   simple_dataset_test() etc.  ..  it's probably compiler bug
      //,public compressable
      
    {
      public:
      
      vertex();
      explicit vertex(unsigned int i);
      vertex(const vertex<T>& v);
      vertex(const std::vector<T>& v);
      virtual ~vertex();
      
#if 0
      typedef typename std::vector<T>::iterator iterator;
      typedef typename std::vector<T>::const_iterator const_iterator;
#endif
      
      unsigned int size() const throw();
      unsigned int resize(unsigned int d) throw();
      
      // TODO: move norm() to generic number class
      T norm() const throw();
      
      // calculates partial norm for vertex(i:j)
      T norm(unsigned int i, unsigned int j) const throw();      
      
      bool normalize() throw(); // length = 1
      void zero() throw(); // vertex = 0;
      
      vertex<T> operator+(const vertex<T>& v) const throw(illegal_operation);
      vertex<T> operator-(const vertex<T>& v) const throw(illegal_operation);
      
      // if v.size() != 1 this is inner product - returns vertex with size 1
      // if v.size() == 1 this is scalar multiplication
      vertex<T> operator*(const vertex<T>& v) const throw(illegal_operation);
      
      vertex<T> operator/(const vertex<T>& v) const throw(illegal_operation);
      vertex<T> operator!() const throw(illegal_operation);
      vertex<T> operator-() const throw(illegal_operation);
      
      // cross product
      vertex<T> operator^(const vertex<T>& v) const throw(illegal_operation);
      
      vertex<T>& operator+=(const vertex<T>& v) throw(illegal_operation);
      vertex<T>& operator-=(const vertex<T>& v) throw(illegal_operation);
      vertex<T>& operator*=(const vertex<T>& v) throw(illegal_operation);
      vertex<T>& operator/=(const vertex<T>& v) throw(illegal_operation);
      
      vertex<T>& operator=(const vertex<T>& v) throw(illegal_operation);      
      
      bool operator==(const vertex<T>& v) const throw(uncomparable);
      bool operator!=(const vertex<T>& v) const throw(uncomparable);
      bool operator>=(const vertex<T>& v) const throw(uncomparable);
      bool operator<=(const vertex<T>& v) const throw(uncomparable);
      bool operator< (const vertex<T>& v) const throw(uncomparable);
      bool operator> (const vertex<T>& v) const throw(uncomparable);
      
      vertex<T>& operator=(const quaternion<T>&) throw(std::domain_error);
      
      vertex<T>& abs() throw();
      
      /* scalars */
      vertex<T>& operator= (const T& s) throw(illegal_operation);
      vertex<T>  operator* (const T& s) const throw();
      vertex<T>  operator/ (const T& s) const throw(std::invalid_argument);
      vertex<T>& operator*=(const T& s) throw();
      vertex<T>& operator/=(const T& s) throw(std::invalid_argument);
    
      template <typename TT>
      friend vertex<TT> operator*(const TT& s, const vertex<TT>& v); // multi from right
      
      
      // multiply from left
      vertex<T>  operator* (const matrix<T>& m) const throw(std::invalid_argument);    

      // outer product
      matrix<T> outerproduct() const throw(std::domain_error);
      matrix<T> outerproduct(const vertex<T>& v) const throw(std::domain_error);
      matrix<T> outerproduct(const vertex<T>& v0,
			     const vertex<T>& v1) const throw(std::domain_error);
      
      // element-wise multiplication of vector elements
      vertex<T>& dotmulti(const vertex<T>& v) throw(illegal_operation);
      
      T& operator[](const unsigned int& index) throw(std::out_of_range, illegal_operation);
      const T& operator[](const unsigned int& index) const throw(std::out_of_range, illegal_operation);
      
      bool subvertex(vertex<T>& v, unsigned int x0, unsigned int len) const throw();
      bool write_subvertex(vertex<T>& v, unsigned int x0) throw();
      
#if 0
      iterator begin() throw(); // iterators
      iterator end() throw();
      const_iterator begin() const throw(); // iterators
      const_iterator end() const throw();
#endif
      
      bool comparable() throw();
      
      
      //////////////////////////////////////////////////
      // vertex data compression
      
      bool compress() throw();
      bool decompress() throw();
      bool iscompressed() const throw();
      float ratio() const throw();
      
      //////////////////////////////////////////////////
      // direct memory access
      
      // copies vertex[start:(start+len-1)] = data[0:(len-1)]
      bool importData(const T* data, unsigned int len=0, unsigned int start=0) throw();
      
      // copies data[0:(len-1)] = vertex[start:(start+len-1)]
      bool exportData(T* data, unsigned int len=0, unsigned int start=0) const throw();
      
      //////////////////////////////////////////////////
      
      // friend list
      friend class matrix<T>;
      
      friend class whiteice::neuronlayer<T>;
      friend class whiteice::backpropagation<T>;
      friend class whiteice::SOM2D;
      
      friend bool gramschmidt<T>(matrix<T>& B,
				 const unsigned int i,
				 const unsigned int j);
      
      // correlation
      friend bool autocorrelation<T>(matrix<T>& R, const std::vector< vertex<T> >& data);
      friend bool autocorrelation<T>(matrix<T>& R, const matrix<T>& W);
      friend bool mean_covariance_estimate<T>(vertex<T>& m, matrix<T>& R, const std::vector< vertex<T> >& data);
      
      
      // rotations
      friend bool rhouseholder_leftrot<T> (matrix<T>& A,
					   const unsigned int i,
					   const unsigned int M,
					   const unsigned int k,
					   vertex<T>& v);
      
      friend bool rhouseholder_rightrot<T>(matrix<T>& A,
					   const unsigned int i,
					   const unsigned int M,
					   const unsigned int k,
					   vertex<T>& v);

      private:
      
      T* data;      
      unsigned int dataSize;
      
      MemoryCompressor* compressor;
      
    };
    
    
    template <typename T>
      vertex<T> operator*(const T& s, const vertex<T>& v);
    
    template <typename T>
      std::ostream& operator<<(std::ostream& ios,
			       const whiteice::math::vertex<T>&);
    
    
    // tries to convert vertex of type S to vertex of type T (B = A)
    template <typename T, typename S>
      bool convert(vertex<T>& B, const vertex<S>& A) throw()
      {
	try{
	  if(B.resize(A.size()) == false)
	    return false;
	  
	  for(unsigned int i=0;i<B.size();i++)
	    B[i] = static_cast<T>(A[i]);
	  
	  return true;
	}
	catch(std::exception& e){
	  return false;
	}
      }
  };
};


#include "backpropagation.h"
#include "neuronlayer.h"

#include "linear_algebra.h"
#include "correlation.h"
#include "matrix_rotations.h"


namespace whiteice{
  namespace math{
    
    extern template class vertex<float>;
    extern template class vertex<double>;
    extern template class vertex<complex<float> >;
    extern template class vertex<complex<double> >;
    
    extern template class vertex<int>;
    extern template class vertex<char>;
    extern template class vertex<unsigned int>;
    extern template class vertex<unsigned char>;
        
    extern template class vertex< blas_real<float> >;
    extern template class vertex< blas_real<double> >;
    extern template class vertex< blas_complex<float> >;
    extern template class vertex< blas_complex<double> >;
    
    extern template vertex<float> operator*<float>(const float& s, const vertex<float>& v);
    extern template vertex<double> operator*<double>(const double& s, const vertex<double>& v);
    
    extern template vertex<complex<float> > operator*<complex<float> >
      (const complex<float>& s, const vertex<complex<float> >& v);
    
    extern template vertex<complex<double> > operator*<complex<double> >
      (const complex<double>& s, const vertex<complex<double> >& v);
    
    extern template vertex<int> operator*<int>(const int& s, const vertex<int>& v);
    extern template vertex<char> operator*<char>(const char& s, const vertex<char>& v);
    extern template vertex<unsigned int> operator*<unsigned int>(const unsigned int& s, const vertex<unsigned int>& v);
    extern template vertex<unsigned char> operator*<unsigned char>(const unsigned char& s, const vertex<unsigned char>& v);
      
    extern template vertex<blas_real<float> > operator*<blas_real<float> >
      (const blas_real<float>& s, const vertex<blas_real<float> >& v);
    
    extern template vertex<blas_real<double> > operator*<blas_real<double> >
      (const blas_real<double>& s, const vertex<blas_real<double> >& v);
									     
    extern template vertex<blas_complex<float> > operator*<blas_complex<float> >
      (const blas_complex<float>& s, const vertex<blas_complex<float> >& v);
    
    extern template vertex<blas_complex<double> > operator*<blas_complex<double> >
      (const blas_complex<double>& s, const vertex<blas_complex<double> >& v);
       
    
    
    extern template std::ostream& operator<< <float>(std::ostream& ios, const vertex<float>&);
    extern template std::ostream& operator<< <double>(std::ostream& ios, const vertex<double>&);
    extern template std::ostream& operator<< <complex<float> >(std::ostream& ios, const vertex<complex<float> >&);
    extern template std::ostream& operator<< <complex<double> >(std::ostream& ios, const vertex<complex<double> >&);
    
    extern template std::ostream& operator<< <int>(std::ostream& ios, const vertex<int>&);
    extern template std::ostream& operator<< <char>(std::ostream& ios, const vertex<char>&);
    extern template std::ostream& operator<< <unsigned int>(std::ostream& ios, const vertex<unsigned int>&);
    extern template std::ostream& operator<< <unsigned char>(std::ostream& ios, const vertex<unsigned char>&);
    
    extern template std::ostream& operator<< <blas_real<float> >(std::ostream& ios, const vertex<blas_real<float> >&);
    extern template std::ostream& operator<< <blas_real<double> >(std::ostream& ios, const vertex<blas_real<double> >&);
    extern template std::ostream& operator<< <blas_complex<float> >(std::ostream& ios, const vertex<blas_complex<float> >&);
    extern template std::ostream& operator<< <blas_complex<double> >(std::ostream& ios, const vertex<blas_complex<double> >&);
    
    
  };
};



#endif
