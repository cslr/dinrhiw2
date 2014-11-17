/*
 * generic matrix class
 */

#ifndef gmatrix_h
#define gmatrix_h

#include "ownexception.h"
#include "number.h"

#include <stdexcept>
#include <exception>


namespace whiteice
{
  namespace math
  {

    template <typename T, typename S>
      class gvertex;
    
    // S must be scalar which can operate with T
    // T is type of the scalar
    
    template <typename T=float, typename S=T>
      class gmatrix : public whiteice::number< gmatrix<T,S>, S, gvertex<T,S>, unsigned int >
    {
      public:
      
      explicit gmatrix(const unsigned int size_y = 4,
		      const unsigned int size_x = 4);
      gmatrix(const gmatrix<T,S>& M);
      gmatrix(const gvertex<T,S>& diagonal);
      virtual ~gmatrix();
      
      
      gmatrix<T,S> operator+(const gmatrix<T,S>&) const throw(illegal_operation);
      gmatrix<T,S> operator-(const gmatrix<T,S>&) const throw(illegal_operation);
      gmatrix<T,S> operator*(const gmatrix<T,S>&) const throw(illegal_operation);
      gmatrix<T,S> operator/(const gmatrix<T,S>&) const throw(illegal_operation);
      gmatrix<T,S> operator!() const throw(illegal_operation);
      gmatrix<T,S> operator-() const throw(illegal_operation);
      
      gmatrix<T,S>& operator+=(const gmatrix<T,S>&) throw(illegal_operation);
      gmatrix<T,S>& operator-=(const gmatrix<T,S>&) throw(illegal_operation);
      gmatrix<T,S>& operator*=(const gmatrix<T,S>&) throw(illegal_operation);
      gmatrix<T,S>& operator/=(const gmatrix<T,S>&) throw(illegal_operation);
      
      gmatrix<T,S>& operator=(const gmatrix<T,S>&) throw(illegal_operation);
                  
      bool operator==(const gmatrix<T,S>&) const throw(uncomparable);
      bool operator!=(const gmatrix<T,S>&) const throw(uncomparable);
      bool operator>=(const gmatrix<T,S>&) const throw(uncomparable);
      bool operator<=(const gmatrix<T,S>&) const throw(uncomparable);
      bool operator< (const gmatrix<T,S>&) const throw(uncomparable);
      bool operator> (const gmatrix<T,S>&) const throw(uncomparable);

      // scalars and gmatrix interaction
      gmatrix<T,S>& operator= (const S&) throw(illegal_operation);

      gmatrix<T,S>  operator* (const S&) const throw();
      
      template <typename TT, typename SS>
      friend gmatrix<TT,SS> operator*(const SS&, const gmatrix<TT,SS>&)
        throw(std::invalid_argument);
      
      gmatrix<T,S>  operator/ (const S&) const throw(std::invalid_argument);

      gmatrix<T,S>& operator*=(const S&) throw();
      gmatrix<T,S>& operator/=(const S&) throw(std::invalid_argument);

      gvertex<T,S> operator*(const gvertex<T,S>&) const throw(std::invalid_argument);      
      
      gvertex<T,S>& operator[](const unsigned int& index)
        throw(std::out_of_range, illegal_operation);
      
      const gvertex<T,S>& operator[](const unsigned int& index)
        const throw(std::out_of_range, illegal_operation);
      
      T& operator()(unsigned int y, unsigned int x) throw(std::out_of_range, illegal_operation);
      const T& operator()(unsigned int y, unsigned int x) const throw(std::out_of_range, illegal_operation);
      
      
      gmatrix<T,S>& zero(); // zeroes gmatrix
      gmatrix<T,S>& identity();
      
      gmatrix<T,S>& crossproduct(const gvertex<T,S>& v) throw(std::domain_error);
      
      // euclidean rotation
      gmatrix<T,S>& rotation(const S& xr, const S& yr, const S& zr) throw();
      
      // translation
      gmatrix<T,S>& translation(const S& dx, const S& dy, const S& dz) throw();
      
      
      gmatrix<T,S>& abs() throw();
      gmatrix<T,S>& conj() throw();
      gmatrix<T,S>& transpose() throw();
      gmatrix<T,S>& hermite() throw();
      
      gmatrix<T,S>& inv() throw(std::logic_error); // inverse
      
      T det() const throw(std::logic_error); // determinate
      T trace() const throw(std::logic_error);
      
      unsigned int size() const throw();
      unsigned int ysize() const throw();      
      unsigned int xsize() const throw();
      
      bool resize_x(unsigned int d) throw();
      bool resize_y(unsigned int d) throw();
      bool resize(unsigned int y, unsigned int x) throw();
      
      
      // normalizes each row vector of matrix to have unit length
      void normalize() throw();
      
      
      bool comparable() throw(){ return false; }
      
      // TODO: add dummy    static const gmatrix<T,S> null; for empty/null gvertex
      
      private:
      
      gvertex< gvertex<T, S>, S > data;
    };
    
    
    template <typename T, typename S>
      std::ostream& operator<<(std::ostream& ios, const gmatrix<T,S>& M);
    
    // tries to convert gmatrix of type L to gmatrix of type T (B = A)
    template <typename T, typename S, typename L, typename M>
      bool convert(gmatrix<T,S>& B, const gmatrix<L,M>& A) throw();

  }
}

  
#include "gvertex.h"
#include "gmatrix.cpp"
  
#endif

