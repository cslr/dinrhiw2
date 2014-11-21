/*
 * eigenvalue and singular value decomposition
 *
 * - at least currently these things work only with real valued data
 */

// TODO: move sylvester equation solver (when works) to linear_equations.h/cpp
// keep here only EVD and SVD  related code
// (eigenvalue/singular value decomposition)

#ifndef eig_h
#define eig_h

#include "blade_math.h"
#include "atlas.h"


namespace whiteice
{
  namespace math
  {
    template <typename T> class vertex;
    template <typename T> class matrix;

    
    // calculates eigenvalues d and eigenvector matrix X of 2x2 A matrix
    // returns false if eigenvalues are complex and complex_ok == false
    template <typename T>
      bool eig2x2matrix(const matrix<T>& A,
			vertex<T>& d, matrix<T>& X, bool complex_ok = false);
    
    // solves eigenvectors for eigenvalues
    // with inverse power method
    // (works only if A's eigenvalues aren't close to zero)
    // e contains different eigenvectors and
    // n tells repeations of different eigenvectors
    template <typename T>
      void invpowermethod(const matrix<T>& A,
			  const vertex<T>& e,
			  const vertex<T>& n,
			  matrix<T>& X);
    
    
    
    // reduces A to hessenberg form: A = Q*H*Q', overwrites A with H
    template <typename T>
      bool hessenberg_reduction(matrix<T>& A, matrix<T>& Q);
    
    
    // calculates Q*R decomposition of A(i:i+N,j:j+N). A will be replated with R
    template <typename T>
      bool qr(matrix<T>& A, matrix<T>& Q);
    
    
    // does symmetric qr step with wilkinson shift to square (sub)matrix
    // A(e1:e1+N-1,e1:e1+N-1) and makes same colgivens also to X(e1:e1+N-1,e1:e1+N-1)
    template <typename T>
      bool implicit_symmetric_qrstep_wilkinson(matrix<T>& A, matrix<T>& X,
					       unsigned int e1, unsigned int N);
    
    
    // solves eigenvalue problem (EVD) for symmetric real matrices
    // A = X * D * X^t , A = A^t. A will be overwritten with D.
    template <typename T>
      bool symmetric_eig(matrix<T>& A, matrix<T>& X);
    
    // calculates real singular value decomposition (SVD) A = U*S*V^t.
    // A will be replaced with matrix S containing A's
    // singular values along its diagonal.
    template <typename T>
      bool svd(matrix<T>& A, matrix<T>& U, matrix<T>& V);
    
    
    // calculates francis implicit double shift
    // reduction for H(i:j-1,i:j-1) and restores
    // matrix back to hessenberg form with Z (i.e. unsymmetric qr-step)
    // [NOT TESTED/NOT USED/NEEDED RIGHT NOW]
    template <typename T>
      void francis_qr_step(matrix<T>& H,
			   matrix<T>& Z,
			   const unsigned int i,
			   const unsigned int j,
			   bool iscomplex);
    
    
    // calculates (real) schur form, overwrites A.
    // A_orig = X*A*X'
    // (TODO check/fix if this works with
    //  complex numbers)
    template <typename T>
      void schur(matrix<T>& A,
		 matrix<T>& X,
		 bool iscomplex);
    
    
    
    
    extern template bool eig2x2matrix< atlas_real<float> >
      (const matrix< atlas_real<float> >& A, vertex< atlas_real<float> >& d, 
       matrix< atlas_real<float> >& X, bool complex_ok);
    extern template bool eig2x2matrix< atlas_real<double> >
      (const matrix< atlas_real<double> >& A, vertex< atlas_real<double> >& d,
       matrix< atlas_real<double> >& X, bool complex_ok);
    extern template bool eig2x2matrix< atlas_complex<float> >
      (const matrix< atlas_complex<float> >& A, vertex< atlas_complex<float> >& d,
       matrix< atlas_complex<float> >& X, bool complex_ok);
    extern template bool eig2x2matrix< atlas_complex<double> >
      (const matrix< atlas_complex<double> >& A, vertex< atlas_complex<double> >& d,
       matrix< atlas_complex<double> >& X, bool complex_ok);
    extern template bool eig2x2matrix<float>
      (const matrix<float>& A, vertex<float>& d, matrix<float>& X, bool complex_ok);
    extern template bool eig2x2matrix<double>
      (const matrix<double>& A, vertex<double>& d, matrix<double>& X, bool complex_ok);
    extern template bool eig2x2matrix< complex<float> >
      (const matrix< complex<float> >& A, vertex< complex<float> >& d,
       matrix< complex<float> >& X, bool complex_ok);
    extern template bool eig2x2matrix< complex<double> >
      (const matrix< complex<double> >& A, vertex< complex<double> >& d,
       matrix< complex<double> >& X, bool complex_ok);
       
    
    
    extern template bool hessenberg_reduction< atlas_real<float> >
      (matrix< atlas_real<float> >& A, matrix< atlas_real<float> >& Q);
    extern template bool hessenberg_reduction< atlas_real<double> >
      (matrix< atlas_real<double> >& A, matrix< atlas_real<double> >& Q);
    extern template bool hessenberg_reduction<float>
      (matrix<float>& A, matrix<float>& Q);
    extern template bool hessenberg_reduction<double>
      (matrix<double>& A, matrix<double>& Q);
    
    
    
    extern template bool qr< atlas_real<float> > (matrix< atlas_real<float> >&  A,
						  matrix< atlas_real<float> >&  Q);
    extern template bool qr< atlas_real<double> >(matrix< atlas_real<double> >& A,
						  matrix< atlas_real<double> >& Q);
    extern template bool qr<float> (matrix<float>&  A, matrix<float>&  Q);
    extern template bool qr<double>(matrix<double>& A, matrix<double>& Q);
    
    
    extern template bool implicit_symmetric_qrstep_wilkinson< atlas_real<float> >
      (matrix< atlas_real<float> >& A, matrix< atlas_real<float> >& X,
       unsigned int e1, unsigned int N);
    extern template bool implicit_symmetric_qrstep_wilkinson< atlas_real<double> >
      (matrix< atlas_real<double> >& A, matrix< atlas_real<double> >& X,
       unsigned int e1, unsigned int N);
    extern template bool implicit_symmetric_qrstep_wilkinson<float>
      (matrix<float>& A, matrix<float>& X,
       unsigned int e1, unsigned int N);
    extern template bool implicit_symmetric_qrstep_wilkinson<double>
      (matrix<double>& A, matrix<double>& X,
       unsigned int e1, unsigned int N);
    
    
    
    extern template bool symmetric_eig< atlas_real<float> >
      (matrix< atlas_real<float> >& A, matrix< atlas_real<float> >& D);
    extern template bool symmetric_eig< atlas_real<double> >
      (matrix< atlas_real<double> >& A, matrix< atlas_real<double> >& D);
    extern template bool symmetric_eig<float>(matrix<float>& A, matrix<float>& D);
    extern template bool symmetric_eig<double>(matrix<double>& A, matrix<double>& D);
    
    
    extern template bool svd< atlas_real<float> >
      (matrix< atlas_real<float> >& A,
       matrix< atlas_real<float> >& U, matrix< atlas_real<float> >& V);
    extern template bool svd< atlas_real<double> >
      (matrix< atlas_real<double> >& A,
       matrix< atlas_real<double> >& U, matrix< atlas_real<double> >& V);
    extern template bool svd<float>
      (matrix<float>& A, matrix<float>& U, matrix<float>& V);
    extern template bool svd<double>
      (matrix<double>& A, matrix<double>& U, matrix<double>& V);
      
    
    
  };
};


#include "vertex.h"
#include "matrix.h"


#endif

