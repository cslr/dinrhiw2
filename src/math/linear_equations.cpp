/*
 * see implementation notes from the "linear_equations.h"
 */

#include <iostream>
#include <algorithm>
#include <exception>
#include <stdexcept>

#include "linear_equations.h"
#include "matrix.h"
#include "vertex.h"
#include "dinrhiw_blas.h"


namespace whiteice
{
  namespace math
  {
    
    // explicit template instantations
    
    template bool linsolve< blas_real<float> >
    (matrix< blas_real<float> >& A, 
     vertex< blas_real<float> >& x, 
     const vertex< blas_real<float> >& b) throw();
    
    template bool linsolve< blas_real<double> >
    (matrix< blas_real<double> >& A, 
     vertex< blas_real<double> >& x, 
     const vertex< blas_real<double> >& b) throw();
    
    template bool linsolve<float>
    (matrix<float>& A, 
     vertex<float>& x, 
     const vertex<float>& b) throw();
    
    template bool linsolve<double>
    (matrix<double>& A, 
     vertex<double>& x, 
     const vertex<double>& b) throw();
    
    
    
    template bool linlsqsolve< blas_real<float> >
    (matrix< blas_real<float> >& A, 
     const vertex< blas_real<float> >& b, 
     vertex< blas_real<float> >& x) throw();
    
    template bool linlsqsolve< blas_real<double> >
    (matrix< blas_real<double> >& A, 
     const vertex< blas_real<double> >& b, 
     vertex< blas_real<double> >& x) throw();
    
    template bool linlsqsolve<float>
    (matrix<float>& A, 
     const vertex<float>& b, 
     vertex<float>& x) throw();
    
    template bool linlsqsolve<double>
    (matrix<double>& A, 
     const vertex<double>& b, 
     vertex<double>& x) throw();
    
    
    
    template bool cholesky_factorization< blas_real<float> >
      (matrix< blas_real<float> >& A) throw();
    template bool cholesky_factorization< blas_real<double> >
      (matrix< blas_real<double> >& A) throw();
    template bool cholesky_factorization<float>
      (matrix<float>& A) throw();
    template bool cholesky_factorization<double>
      (matrix<double>& A) throw();    
    
    
    template bool solvegg< blas_real<float> >
    (matrix< blas_real<float> >& C, 
     vertex< blas_real<float> >& x) throw();
    
    template bool solvegg< blas_real<double> >
    (matrix< blas_real<double> >& C, 
     vertex< blas_real<double> >& x) throw();
    
    template bool solvegg<float>
      (matrix<float>& C, vertex<float>& x) throw();
    template bool solvegg<double>
      (matrix<double>& C, vertex<double>& x) throw();
    
    
    template void solve_sylvester< blas_real<float> >
    (const matrix< blas_real<float> >& A,
     const matrix< blas_real<float> >& B,
     matrix< blas_real<float> >& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    template void solve_sylvester< blas_real<double> >
    (const matrix< blas_real<double> >& A,
     const matrix< blas_real<double> >& B,
     matrix< blas_real<double> >& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    template void solve_sylvester< blas_complex<float> >
    (const matrix< blas_complex<float> >& A,
     const matrix< blas_complex<float> >& B,
     matrix< blas_complex<float> >& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    template void solve_sylvester< blas_complex<double> >
    (const matrix< blas_complex<double> >& A,
     const matrix< blas_complex<double> >& B,
     matrix< blas_complex<double> >& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    template void solve_sylvester<float>
    (const matrix<float>& A,
     const matrix<float>& B,
     matrix<float>& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    template void solve_sylvester<double>
    (const matrix<double>& A,
     const matrix<double>& B,
     matrix<double>& C,
     const unsigned int i, const unsigned int j,
     const unsigned int a, const unsigned int k,
     const unsigned int l, const unsigned int b);
    
    
    
    /*
     * solves linear Ax = b problem
     * by gaussian elimination with partial pivoting
     * returns false if A is (numerically) singular
     * A is destroyed in a process.
     * (caller may need to make local copies for the call)
     */
    template <typename T>
      bool linsolve(matrix<T>& A, vertex<T>& x, const vertex<T>& b) throw()
      {
	// initial conditions and initializations
	if(A.xsize() != A.ysize()) return false; // not a square (TODO: calculate pseudoinverse with svd then)
	if(b.size() != A.xsize()) return false;  // b not large enough	
	x = b;
		
	vertex<unsigned int> permutation(A.ysize()); // permutation of rows
	// permutation p: p(i) = j <-> permutation[i] = j
	
	for(unsigned int i=0;i<permutation.size();i++)
	  permutation[i] = i; // identity permutation		
	
	const unsigned int N = A.ysize();
	
	// calculates upper triangular A 
	// (doesn't really really/numerically
	//  bother to zero lower trianglular part)
	for(unsigned int i=0;i<N;i++) // i:th row
	{
	  // finds biggest value from the column
	  unsigned int index = i;
	  for(unsigned int j=index;j<N;j++)
	    if(A(i,j) > A(index,j)) index = i;
	  
	  if(index != i){ // changes rows
	    // swaps rows
	    // (optimization: should only copy A[i][i..N-1] !)
	    
	    // std::swap< vertex<T> >(A[i], A[index]);
	    // slow swap..
	    for(unsigned int k=0;k<A.xsize();k++){
	      T temp = A(index,k);
	      A(i,k) = temp;
	    }
	    
	    
	    std::swap< T > (x[i], x[index]);
	    
	    // updates permutation
	    std::swap< unsigned int >(permutation[i], permutation[index]);
	  }

	  if(A(i,i) == 0) // TODO: or close to zero! (calculate good tolerance)
	    return false;  // singular matrix
	    
	  // divides i:th row/equation
	  x[i] /= A(i,i);

	  // divides a_ii to be one
	  // (reverse order to set a_ii = 1 last)
	  for(unsigned int j=N;j>i;j--){
	    A(i,j-1) /= A(i,i);
	  }
	  
	  // 'zeroes' bottom rows
	  
	  for(unsigned int j=i+1;j<N;j++){ // j:th row
	    // only alters absolutely needed parts of A's
	    // lower triangular values
	    // -> lower triangular values = (more or less) garbage
	    // (not used anymore)
	    
	    x[j] -= A(j,i)*x[i]; // updates result vector
	    
	    // updates row
	    for(unsigned int k=i+1;k<N;k++) // here: skips actual zeroing of A[j][i]
	      A(j,k) -= A(j,i)*A(i,k);
	  }
	  
	}
	
	// A is upper triangular, solves it iteratively
	// only 'virtually' zeroes upper triangular values
	// as a linear sum of already solved variables
	// (doesn't solve values/variables separately
	//  -> faster + more numerical accuracy)
	
	T sum;
	
	if(N >= 2){
	  for(unsigned int i=N;i>0;i--){ // i:th row
	    
	    // calculates sum of already solved values
	    // for this row and subtracts it
	    // from the other side -> left side variables
	    // are 'virtually' zeroed
	    sum = 0.0f;
	    for(unsigned int j=i;j<N;j++)
	      sum += A(i - 1,j)*x[j];
	    
	    x[i-1] -= sum;
	    x[i-1] /= A(i-1,i-1); // not necessary(?)
	  }
	}
	
	return true;
      }
   
    
    /**************************************************/
    
    /* least squares solution */
    template <typename T>
      bool linlsqsolve(matrix<T>& A, const vertex<T>& b, vertex<T>& x) throw()
      {
	// A = X^(m x n) , m >= n

	if(!(A.ysize() >= A.xsize()))
	  return false;
	
	// calculates first C = A*A^t (note: should only calculate lower/upper triangular part)
	
	// 1. C = A^t * A (slow implementation)
	matrix<T> At = A;
	At.transpose();	
	matrix<T> C = At * A; // needs only really calculate lower triangular part
	
	// 2. d = A^t * b (reasonable fast implementation possible)
	x = At * b;
	
	// 3. cholesky factorization (quite fast, 
	//    use gaxpy + atlas + partial saving of data (lower part only) for fast impl.)
	if(!cholesky_factorization(C)) return false;
	
	// 4. solve Gy = d  (G is lower triangular)
	// 5. solve G^t*x = y (G^t is upper triangular)
	
	if(!solvegg(C, x)) // solves G*G^t*z = x, saves z in x
	  return false;
	
	return true;
      }
    
    
    // calculates cholesky factorization of symmetric
    // positive definite matrix, A = G*G^t (G is lower triangular)
    // implementation only uses lower triangular part of A and calculates
    // result to lower triangular part of A (overwrites input data).
    template <typename T>
      bool cholesky_factorization(matrix<T>& A) throw()
      {
	if(A.xsize() != A.ysize()) return false;
	
	const unsigned int N = A.xsize();
	T scale;
	
	// processes first column
	scale = T(1.0) / sqrt(A(0,0));
	for(unsigned int k=0;k<N;k++)
	  A(k,0) *= scale;
	
	
	for(unsigned int i=1;i<N;i++){
	  for(unsigned int j=0;j<=(i - 1);j++){
	    for(unsigned int k=j;k<N;k++){
	      A(k,i) -= A(i,j)*A(k,j);
	    }
	  }
	  
	  if(A(i,i) <= 0) // not positive definite
	    return false;
	  
	  // normalizes then length of A[j..N][j]
	  scale = T(1.0) / sqrt(A(i,i));
	  
	  for(unsigned int k=i;k<N;k++) // was k=j -> k=i (correct?)
	    A(k,i) *= scale;
	}
	
	return true;
      }
    
    
    template <typename T>
      bool solvegg(matrix<T>& C, vertex<T>& x) throw()
      {
	// C's lower triangular part = G;
	
	// 4. solves Gy = d  (G is lower triangular)

	const unsigned int N = C.ysize();
	T sum;
	
	for(unsigned int i=0;i<N;i++){ // i:th row
	    
	  // calculates sum of already solved values
	  // for this row and subtracts it
	  // from the other side -> left side variables
	  // are 'virtually' zeroed
	  sum = 0.0f;
	  for(unsigned int j=0;j<i;j++)
	    sum += C(i,j)*x[j];
	  
	  x[i] -= sum;
	  
	  if(C(i,i) <= 0) return false;
	  else x[i] /= C(i,i);
	}
	
	
	// 5. solves G^t*x = y (G^t is upper triangular)
    
	for(unsigned int i=N;i>0;i--){ // i:th row
	  
	  // calculates sum of already solved values
	  // for this row and subtracts it
	  // from the other side -> left side variables
	  // are 'virtually' zeroed
	  sum = 0.0f;
	  for(unsigned int j=i;j<N;j++)
	    // C[i][j].transpose() * x[j] (because need to get values from lower triangular part)
	    sum += C(j,i - 1)*x[j];
	  
	  x[i - 1] -= sum;
	  
	  if(C(i - 1, i - 1) <= 0) return false;
	  else x[i - 1] /= C(i - 1, i - 1);
	}	
	
	return true;
      }
    
    
    /**************************************************/
    
    
    // FIXME: SYLVESTER EQUATION SOLVER IS CURRENT BROKEN
    // solves sylvester eq and saves result in C
    // A(i:i+a:j:j+a)*X - X*B(k:k+b,l:l+b) = C
    template <typename T>
    void solve_sylvester(const matrix<T>& A,
			 const matrix<T>& B,
			 matrix<T>& C,
			 const unsigned int i,
			 const unsigned int j,
			 const unsigned int a,
			 const unsigned int k,
			 const unsigned int l,
			 const unsigned int b)
    {
      // first iteration, p = 0
      unsigned int p = 0;
      
      if(B(k+p+1,l+p) != T(0)){ // must solve two vectors at once (2x2 block)
	vertex<T> h(2*a);
	
	for(unsigned int m=0;m<a;m++) h[m] = C(m, p);
	for(unsigned int m=a;m<2*a;m++) h[m] = C(m-a, p+1);
	
	// slow: shouldn't generate separated vectors and matrix for
	// linear equations, should write linear equation solver for
	// given data structures instead
	matrix<T> AA(2*a,2*a); // initial matrix is zero matrix (right?)
	
	for(unsigned int m=0;m<a;m++){
	  for(unsigned int n=0;n<a;n++){
	    AA(m,n)     = A(i+m,j+n);
	    AA(a+m,a+n) = A(i+m,j+n);
	  }
	  
	  AA(m,m)     -= B(k+p,l+p);
	  AA(m,a+m)   -= B(k+p+1,l+p);
	  AA(a+m,m)   -= B(k+p+1,l+p);
	  AA(a+m,a+m) -= B(k+p+1,l+p+1);
	}
	
	// slow (optimize: start to use linear equations solvers:
	// AAx = h when they are known to work correctly)
	AA.inv();
	h = AA*h;
	
	for(unsigned int m=0;m<a;m++)   C(m,p)   = h[m];
	for(unsigned int m=a;m<2*a;m++) C(m-a,p+1) = h[m];
	
	p += 2;
      }
      else{ // need to solve only single vector (1x1 block)
	
	// solves (AA - b(k,k)*I)*b = z
	vertex<T> z(a);
	matrix<T> AA(A);
	for(unsigned int m=0;m<a;m++){
	  AA(m,m) -= B(p,p);
	  z[m] = C(m,p);
	}
	
	AA.inv();
	z = AA*z;
	
	for(unsigned int m=0;m<a;m++){
	  C(m,p) = z[m];
	}
	
	p++;
      }
      
      
      // inner parts of the solution matrix
      while(p < b-1){
	
	
	if(B(k+p+1,l+p) != T(0)){ // must solve two vectors at once (2x2 block)
	  vertex<T> h(2*a);
	  
	  for(unsigned int m=0;m<a;m++){
	    h[m] = C(m,p);
	    
	    for(unsigned int n=0;n<p;n++)
	      h[m] += C(m,n)*B(k+n,l+p);
	  }
	  
	  for(unsigned int m=a;m<2*a;m++){
	    h[m] = C(m-a,p+1);
	    
	    for(unsigned int n=0;n<p;n++)
	      h[m] += C(m,n)*B(k+n,l+p+1);
	  }
	  
	  // slow: shouldn't generate separated vectors and matrix for
	  // linear equations, should write linear equation solver for
	  // given data structures instead
	  matrix<T> AA(2*a,2*a); // initial matrix is zero matrix (right?)
	  
	  for(unsigned int m=0;m<a;m++){
	    for(unsigned int n=0;n<a;n++){
	      AA(m,n)     = A(i+m,j+n);
	      AA(a+m,a+n) = A(i+m,j+n);
	    }
	    
	    AA(m,m)     -= B(k+p,l+p);
	    AA(m,a+m)   -= B(k+p+1,l+p);
	    AA(a+m,m)   -= B(k+p+1,l+p);
	    AA(a+m,a+m) -= B(k+p+1,l+p+1);
	  }
	
	  // slow (optimize: start to use linear equations solvers:
	  // AAx = h when they are known to work correctly)
	  AA.inv();
	  h = AA*h;
	  
	  for(unsigned int m=0;m<a;m++)   C(m,p)   = h[m];
	  for(unsigned int m=a;m<2*a;m++) C(m-a,p+1) = h[m];
	  
	  p += 2;
	}
	else{ // need to solve only single vector (1x1 block)
	  
	  for(unsigned int m=0;m<a;m++){
	    for(unsigned int n=0;n<p;n++){
	      C(m,p) += C(m,n)*B(k+n,l+p);
	    }
	  }
	  
	  
	  // solves (AA - b[k][k]*I)*b = z
	  vertex<T> z(a);
	  matrix<T> AA(A);
	  for(unsigned int m=0;m<a;m++){
	    AA(m,m) -= B(p,p);
	    z[m] = C(m,p);
	  }
	  
	  AA.inv();
	  z = AA*z;
	  
	  for(unsigned int m=0;m<a;m++){
	    C(m,p) = z[m];
	  }
	  
	  p++;
	}
	
      }
      
      // last iteration (if needed)
      
      if(p < b){
	
	for(unsigned int m=0;m<a;m++){
	  for(unsigned int n=0;n<p;n++){
	    C(m,p) += C(m,n)*B(k+n,l+p);
	  }
	}
	  
	
	// solves (AA - b[k][k]*I)*b = z
	vertex<T> z(a);
	matrix<T> AA(A);
	for(unsigned int m=0;m<a;m++){
	  AA(m,m) -= B(p,p);
	  z[m] = C(m,p);
	}
	
	AA.inv();
	z = AA*z;
	
	for(unsigned int m=0;m<a;m++){
	  C(m,p) = z[m];
	}
	
	p++;
	
      }
      
    }
    
    
    
  }
}




















