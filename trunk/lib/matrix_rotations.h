/*
 * matrix rotations for eigenvalue etc. solving
 * (householder and givens)
 */


#ifndef math_matrix_rotations_h
#define math_matrix_rotations_h

#include "atlas.h"

namespace whiteice
{
  namespace math
  {
    template <typename T> class vertex;
    template <typename T> class matrix;
    
    // implementations for real value data
    
    // calculates householder rotation vector to v
    // if(rowdir == false) uses M(y:n, x)
    // if(rowdir == true)  uses M(y, x:n)
    // lenght of v will be |n - y| or |n - x|
    template <typename T>
      bool rhouseholder_vector(vertex<T>& v, const matrix<T>& M,
			       unsigned int y, unsigned int x,
			       bool rowdir = false);
    
    
    // rotates from left (rows, A = PA)
    // P(v(1:N-1)) * A(k:k+N-1, i:i+(M-1)), N = length(v)
    template <typename T>
      bool rhouseholder_leftrot(matrix<T>& A,
				const unsigned int i,
				const unsigned int M,
				const unsigned int k,
				vertex<T>& v);
    
    
    // rotates from right (columns, A = AP)
    // A(i:i+M-1,k:k+N-1) * P(v(1:N-1)), N = length(v)
    template <typename T>
      bool rhouseholder_rightrot(matrix<T>& A,
				 const unsigned int i,
				 const unsigned int M,
				 const unsigned int k,
				 vertex<T>& v);
    
    
    
    // calculates givens rotation parameters
    // a is upper/left one and b is more below/right
    template <typename T>
      void rgivens(const T& a, const T& b, vertex<T>& p);
    
    // givens row rotation of A(k:k+1,i:j-1)
    template <typename T>
      void rgivens_leftrot(matrix<T>& A,
			   const vertex<T>& p,
			   const unsigned int i,
			   const unsigned int j,
			   const unsigned int k);
    
    // givens column rotation of A(i:j-1,k:k+1))
    template <typename T>
      void rgivens_rightrot(matrix<T>& A,
			    const vertex<T>& p,
			    const unsigned int i,
			    const unsigned int j,
			    const unsigned int k);
    

#if 0
    
    // calculates householder rotation
    // vector from v(i:N) to x(j:N), N = length(v)
    template <typename T>
      void householder_vector(const vertex<T>& v,
			      const unsigned int i,
			      vertex<T>& x,
			      const unsigned int j,
			      bool iscomplex=false);
    
    // performs householder rotations
    
    // A = AP (cols)
    template <typename T>
      void householder_rightrot(matrix<T>& A,
				const unsigned int i, const unsigned int j, const unsigned int k,
				vertex<T>& x, unsigned int l);
    
    // A = PA (rows)
    template <typename T>
      void householder_leftrot(matrix<T>& A,
			       const unsigned int i, const unsigned int j, const unsigned int k,
			       vertex<T>& x, unsigned int l);
    
    
    
    // calculates fast givens rotation parameters
    template <typename T>
      void fastgivens(const vertex<T>& x,
		      const vertex<T>& d,
		      vertex<T>& p);
    
    // fastgivens row rotation of A(k:k+1,i:j-1)
    template <typename T>
      void fastgivens_leftrot(matrix<T>& A,
			      const vertex<T>& p,
			      const unsigned int i, const unsigned int j,
			      const unsigned int k);
    
    // fastgivens col rotation of A(i:j-1,k:k+1))
    template <typename T>
      void fastgivens_rightrot(matrix<T>& A,
			       const vertex<T>& p,
			       const unsigned int i, const unsigned int j,
			       const unsigned int k);
    
#endif    
    

    
    extern template bool rhouseholder_vector< atlas_real<float> >
      (vertex< atlas_real<float> >& v,
       const matrix< atlas_real<float> >& M,
       unsigned int y, unsigned int x,
       bool rowdir);
    extern template bool rhouseholder_vector< atlas_real<double> >
      (vertex< atlas_real<double> >& v,
       const matrix< atlas_real<double> >& M,
       unsigned int y, unsigned int x,
       bool rowdir);
    extern template bool rhouseholder_vector< float >
      (vertex< float >& v,
       const matrix< float >& M,
       unsigned int y, unsigned int x,
       bool rowdir);
    extern template bool rhouseholder_vector< double >
      (vertex< double >& v,
       const matrix< double >& M,
       unsigned int y, unsigned int x,
       bool rowdir);
    
    
    extern template bool rhouseholder_leftrot< atlas_real<float> > 
      (matrix< atlas_real<float> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< atlas_real<float> >& v);
    extern template bool rhouseholder_leftrot< atlas_real<double> >
      (matrix< atlas_real<double> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< atlas_real<double> >& v);
    extern template bool rhouseholder_leftrot< float > 
      (matrix< float >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< float >& v);
    extern template bool rhouseholder_leftrot< double >
      (matrix< double >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< double >& v);
    
    
    extern template bool rhouseholder_rightrot< atlas_real<float> >
      (matrix< atlas_real<float> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< atlas_real<float> >& v);
    extern template bool rhouseholder_rightrot< atlas_real<double> >
      (matrix< atlas_real<double> >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< atlas_real<double> >& v);
    extern template bool rhouseholder_rightrot< float >
      (matrix< float >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< float >& v);
    extern template bool rhouseholder_rightrot< double >
      (matrix< double >& A,
       const unsigned int i,
       const unsigned int M,
       const unsigned int k,
       vertex< double >& v);

    
    
    extern template void rgivens< atlas_real<float> >
      (const atlas_real<float>& a, const atlas_real<float>& b, vertex< atlas_real<float> >& p);
    extern template void rgivens< atlas_real<double> >
      (const atlas_real<double>& a, const atlas_real<double>& b, vertex< atlas_real<double> >& p);
    extern template void rgivens< float >
      (const float& a, const float& b, vertex< float >& p);
    extern template void rgivens< double >
      (const double& a, const double& b, vertex< double >& p);
    
    
    extern template void rgivens_rightrot< atlas_real<float> >
      (matrix< atlas_real<float> >& A, const vertex< atlas_real<float> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    extern template void rgivens_rightrot< atlas_real<double> >
      (matrix< atlas_real<double> >& A, const vertex< atlas_real<double> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    extern template void rgivens_rightrot< float >
      (matrix< float >& A, const vertex< float >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    extern template void rgivens_rightrot< double >
      (matrix< double >& A, const vertex< double >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    
    
    extern template void rgivens_leftrot< atlas_real<float> >
      (matrix< atlas_real<float> >& A, const vertex< atlas_real<float> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    extern template void rgivens_leftrot< atlas_real<double> >
      (matrix< atlas_real<double> >& A, const vertex< atlas_real<double> >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    extern template void rgivens_leftrot< float >
      (matrix< float >& A, const vertex< float >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    extern template void rgivens_leftrot< double >
      (matrix< double >& A, const vertex< double >& p,
       const unsigned int i, const unsigned int j, const unsigned int k);
    
  }
}


#include "vertex.h"
#include "matrix.h"

#endif

