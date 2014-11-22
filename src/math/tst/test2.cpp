/*
 * these tests tests code which assumes
 * that basic algorithms (matrices etc.)
 * in part1 worked.
 */

#include <iostream>
#include <exception>
#include <stdexcept>
#include <complex>
#include <memory>
#include <string>
#include <cmath>
#include <new>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "simplex.h"
#include "number.h"
#include "vertex.h"
#include "quaternion.h"
#include "matrix.h"
#include "hermite.h"
#include "bezier.h"
#include "pdftree.h"
#include "norms.h"
#include "linear_equations.h"
#include "matrix_rotations.h"
#include "conffile.h"
#include "eig.h"
#include "ica.h"



using namespace whiteice;
using namespace whiteice::math;

void test_matrix_rotations();

void test_basic_linear();
void test_rotations();
void test_eigenproblem_tests();
void test_ica();



int main(int argc, char **argv, char **envp)
{
  srand(time(0));
  
  test_matrix_rotations();
  std::cout << std::endl;
  
  test_basic_linear();
  std::cout << std::endl;
  
  test_eigenproblem_tests();
  std::cout << std::endl;
  
  test_ica();
  std::cout << std::endl;
  
  return 0;
}



void test_matrix_rotations()
{
  std::cout << "BASIC MATRIX ROTATIONS TESTS" << std::endl;  
  
  try{
  
    /////////////////////////////////////////////////////////
    // householder rotation calculation, right/left rotation
    
    // basic functionality test
    {
      matrix< blas_real<float> > A, B;
      vertex< blas_real<float> > x, y, z;
      
      A.resize(4,4);
      B.resize(4,4);
      
      for(unsigned int j=0;j<A.ysize();j++)
	for(unsigned int i=0;i<A.xsize();i++)	
	  A(j,i) = ((float)rand()/((float)RAND_MAX) - 0.5);
      
      B = A;
      
      
      // calculates householder rotation for the first column of A
      if(rhouseholder_vector(x, A, 0, 0) == false){
	std::cout << "ERROR: calculating householder vector failed"
		  << std::endl;
	return;
      }
      
      // calculates householder rotation for the first row of A
      if(rhouseholder_vector(y, B, 0, 0, true) == false){
	std::cout << "ERROR: calculating householder vector failed"
		  << std::endl;
	return;
      }

      
      if(rhouseholder_leftrot (A, 0, A.xsize(), 0, x) == false){
	std::cout << "ERROR: left rotation failed"
		  << std::endl;
	return;
      }
      
      
      if(rhouseholder_rightrot(B, 0, B.ysize(), 0, y) == false){
	std::cout << "ERROR: right rotation failed"
		  << std::endl;
	return;
      }
      
      
      
      // tests householder rotation to triangelization problem
      
      A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4;
      A(1,0) =  1; A(1,1) =  0; A(1,2) =  0; A(1,3) =  1;
      A(2,0) =  0; A(2,1) =  1; A(2,2) =  1; A(2,3) =  0;
      A(3,0) =  1; A(3,1) =  3; A(3,2) =  2; A(3,3) =  1;
      
      B = A;
      unsigned int N = A.xsize();
      
      std::cout << "Starting lower triangelization of A" << std::endl;
      
      for(unsigned int i=0;i<N;i++){
	rhouseholder_vector(x, A, i, i, false);	
	rhouseholder_leftrot(A, i, A.xsize() - i, i, x);
      }
      
      std::cout << "Result of lower triangelization : " << std::endl
		<< A << std::endl;
      
      A(0,0) -= -1.7321; A(0,1) -= -2.8868; A(0,2) -= -2.8868; A(0,3) -= -3.4641;
      A(1,0) -=  0.0000; A(1,1) -=  2.3805; A(1,2) -=  1.9604; A(1,3) -=  0.4201;
      A(2,0) -=  0.0000; A(2,1) -=  0.0000; A(2,2) -= -1.3504; A(2,3) -= -2.3523;
      A(3,0) -=  0.0000; A(3,1) -=  0.0000; A(3,2) -=  0.0000; A(3,3) -=  0.5388;
      
      if(norm_inf(A) > 0.01)
	std::cout << "BAD RESULT. A - CORRECT = " << A << std::endl;
      
      
      std::cout << "Starting upper triangelization of A" << std::endl;
      
      A = B;
      for(unsigned int i=0;i<N;i++){
	rhouseholder_vector(x, A, i, i, true);	
	rhouseholder_rightrot(A, i, A.xsize() - i, i, x);
      }

      std::cout << "Result of upper triangelization : " << std::endl
		<< A << std::endl;
      
      A(0,0) -= -5.4772; A(0,1) -=  0.0000; A(0,2) -=  0.0000; A(0,3) -=  0.0000;
      A(1,0) -= -0.9129; A(1,1) -=  1.0801; A(1,2) -=  0.0000; A(1,3) -=  0.0000;
      A(2,0) -= -0.9129; A(2,1) -= -0.7715; A(2,2) -=  0.7559; A(2,3) -=  0.0000;
      A(3,0) -= -3.1038; A(3,1) -= -0.7715; A(3,2) -=  2.0788; A(3,3) -= -0.6708;
      
      if(norm_inf(A) > 0.01)
	std::cout << "BAD RESULT. A - CORRECT = " << A << std::endl;
      
    }
        
  }
  catch(std::exception& e){
    std::cout << "Error in householder rotation tests" << std::endl;
    std::cout << "Unexpected exception: " << e.what() << std::endl;
    return;
  }
  
  
  
  try{
    
    /////////////////////////////////////////////////////////
    // givens rotation calculation, right/left rotation
    
    // basic functionality test
    {
      matrix< blas_real<float> > A, B;
      vertex< blas_real<float> > x, y, z;
      
      A.resize(4,4);
      B.resize(4,4);
      
      A(0,0) =  1; A(0,1) =  2; A(0,2) =  3; A(0,3) =  4;
      A(1,0) =  1; A(1,1) =  0; A(1,2) =  0; A(1,3) =  1;
      A(2,0) =  0; A(2,1) =  1; A(2,2) =  1; A(2,3) =  0;
      A(3,0) =  1; A(3,1) =  3; A(3,2) =  2; A(3,3) =  1;

      // tests givens rotations to triangelization problem
      
      B = A;
      unsigned int N = A.xsize();
      x.resize(2);
           
      std::cout << "Starting lower triangelization of A" << std::endl;
      
      for(unsigned int i=0;i<(N-1);i++){
	for(unsigned int j=(N-1);j>i;j--){	  
	  rgivens(A(j-1,i),A(j,i),x); // x = [c,s]
	  rgivens_leftrot(A, x, i, N, j-1); // rotates from the left
	}
      }
      
      std::cout << "Results of lower triangularization: "
		<< std::endl << A << std::endl;
      
      A(0,0) -= -1.7321; A(0,1) -= -2.8868; A(0,2) -= -2.8868; A(0,3) -= -3.4641;
      A(1,0) -=  0.0000; A(1,1) -=  2.3805; A(1,2) -=  1.9604; A(1,3) -=  0.4201;
      A(2,0) -=  0;      A(2,1) -=  0.0000; A(2,2) -=  1.3504; A(2,3) -=  2.3523;
      A(3,0) -=  0;      A(3,1) -=  0;      A(3,2) -=  0;      A(3,3) -= -0.5388;
      
      if(norm_inf(A) > 0.01)
	std::cout << "BAD RESULT. A - CORRECT = " << A << std::endl;
      
      
      A = B;
      x.resize(2);
      
      std::cout << "Starting upper triangelization of A" << std::endl;
      
      for(unsigned int i=0;i<(N-1);i++){
	for(unsigned int j=(N-1);j>i;j--){	  
	  rgivens(A(i,j-1),A(i,j),x); // x = [c,s]
	  rgivens_rightrot(A, x, i, N, j-1); // rotates from the right
	}
      }
      
      std::cout << "Results of upper triangularization: "
		<< std::endl << A << std::endl;
      
    }

  }
  catch(std::exception& e){
    std::cout << "Error in givens rotation tests" << std::endl;
    std::cout << "Unexpected exception: " << e.what() << std::endl;
    return;
  }
  
  
    try{
    
    /////////////////////////////////////////////////////////
    // qr step with implicit wilkinson shift algorithm test
    
    matrix< blas_real<float> > A, I;
    vertex< blas_real<float> > x, y, z;
    
    std::cout << "Implicit wilkinson shift test" << std::endl;
    
    A.resize(4,4);
    I.resize(4,4);
    I.identity();
    
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3; A(0,3) = 4;
    A(1,0) = 1; A(1,1) = 0; A(1,2) = 1; A(1,3) = 0;
    A(2,0) = 4; A(2,1) = 3; A(2,2) = 2; A(2,3) = 1;
    A(3,0) = 1; A(3,1) = 1; A(3,2) = 2; A(3,3) = 2;
    
    std::cout << "A = " << A << std::endl;
    std::cout << "I = " << I << std::endl;
    
    if(implicit_symmetric_qrstep_wilkinson(A, I, 1, 3) == false){
      std::cout << "failed" << std::endl;
    }
    
    std::cout << "A = " << A << std::endl;
    std::cout << "I = " << I << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "Error in wilkinson qr rotation tests" << std::endl;
    std::cout << "Unexpected exception: " << e.what() << std::endl;
    return;
  }
    
  
  std::cout << "BASIC ROTATION TESTS: PASSED" << std::endl;
}




void test_basic_linear()
{
  std::cout << "BASIC LINEAR ALGEBRA ALGORITHMS TESTS"
	    << std::endl;
  
  //////////////////////////////////////////////////
  // gramschmidt orthonormalization tests
  
  try
  {
    // TEST CASE 1: "trivial" diagonal matrix test
    matrix<double>  A, B;
    unsigned int k;
    
    k = rand() % 30;
    
    A.resize(k+3,k+3); B.resize(k+3,k+3);
    A.identity();
    B = A;
    
    gramschmidt(B,0,k+3);
    
    A -= B;
    
    
    if(norm_inf(A) > 0.001){
      std::cout << "ERROR: gramschmidt orthonormalization failed test 1."
		<< std::endl;
    }
    
    // TEST CASE 2: random vectors
    k = rand() % 10;
    
    A.resize(k+3, k + (rand() % 10) + 3);
    B.resize(A.ysize(),A.xsize());
    
    for(unsigned int j=0;j<A.ysize();j++){
      for(unsigned int i=0;i<A.xsize();i++){
	A(j,i) = ((double)rand()) / ((double)RAND_MAX) - 0.5;
      }
    }
    
    B = A;
    
    gramschmidt(B,0,A.ysize());
    
    A = B;
    A.transpose(); // A = B';
    A = B * A; // B * B' should be I
    
    bool ok = true;
    
    for(unsigned int j=0;j<A.ysize() && ok;j++){
      for(unsigned int i=0;i<A.xsize() && ok;i++){
	
	if(i == j){ // tests for unit vector
	  if(whiteice::math::abs(A(j,i) - 1) > 0.001){
	    std::cout << "ERROR: gram-schmidt failed: non unit vectors"
		      << std::endl;
	    std::cout << "B = " << B << std::endl;
	    std::cout << "B * B^t = " << A << std::endl;	    
	    ok = false;
	  }
	}
	else{ // tests for orthogonality
	  if(whiteice::math::abs(A(j,i)) > 0.001){
	    std::cout << "ERROR: gram-schmidt failed: not orthogonal vectors" 
		      << std::endl;
	    std::cout << "B = " << B << std::endl;
	    std::cout << "B * B^t = " << A << std::endl;	    
	    ok = false;
	  }
	 
	}
      }
    }

    if(ok)
      std::cout << "GRAM-SCHMIDT ORTHOGONALIZATION TESTS: PASSED" << std::endl;
      
  }
  catch(std::exception& e){
    std::cout << "ERROR: gram-schmidt tests (1). unexcepted exception: " 
	      << e.what() << std::endl;
  }
      

  //////////////////////////////////////////////////
  // eig2x2matrix
    
  try
  {
    // TEST CASE 1: trivial case - identity matrix
    
    matrix<float> A, X;
    vertex<float> d;
    
    A.resize(2,2);
    A.identity();
    
    if(eig2x2matrix(A, d, X, false) == false){
      std::cout << "ERROR: eig2x2matrix() failed with identity matrix (1)"
		<< std::endl;
    }
    
    if(whiteice::math::abs(d[0] - 1) > 0.001 ||
       whiteice::math::abs(d[1] - 1) > 0.001){
      std::cout << "ERROR: eig2x2matrix() failed with identity matrix (2)"
		<< std::endl;
    }
    
    if(whiteice::math::abs(X(0,0) - 1) > 0.001 ||
       whiteice::math::abs(X(1,1) - 1) > 0.001 ||
       whiteice::math::abs(X(0,1)) > 0.001 ||
       whiteice::math::abs(X(1,0)) > 0.001){
      std::cout << "ERROR: eig2x2matrix() failed with identity matrix (3)"
		<< std::endl;
    }
  }
  catch(std::exception& e){
    std::cout << "ERROR: eig2x2matrix() tests (1). unexcepted exception: " 
	      << e.what() << std::endl;
  }
  
    
  // TEST CASE 2 - random matrix this have
  // very small possiblity of being singular (tested against)
  // calculated with complex numbers if needed
  try{
    matrix<float> A, X;
    vertex<float> d;
    
    A.resize(2,2);
    
    A(0,0) = ((double)rand()/((double)RAND_MAX) - 0.5);
    A(0,1) = ((double)rand()/((double)RAND_MAX) - 0.5);
    A(1,0) = ((double)rand()/((double)RAND_MAX) - 0.5);
    A(1,1) = ((double)rand()/((double)RAND_MAX) - 0.5);
    
    if(eig2x2matrix(A, d, X, false) == false){
      
      matrix< whiteice::math::complex<float> > C, Z;
      vertex< whiteice::math::complex<float> > w;
      
      if(convert(C, A) == false){
	std::cout << "ERROR: problem related conversion to complex values failed (4)"
		  << std::endl;
      }
      
      if(eig2x2matrix(C,w,Z,true) == false){
	std::cout << "ERROR: eig2x2matrix() failed with complex numbers (5)"
		  << std::endl;
      }
      
      if((std::real(whiteice::math::abs(w[0])) +
	  std::imag(whiteice::math::abs(w[0]))) < 0.00001 ||
	 (std::real(whiteice::math::abs(w[1])) +
	  std::imag(whiteice::math::abs(w[1]))) < 0.00001){
	
	std::cout << "ERROR: solved eigenvalues are probably wrong (6)"
		  << std::endl;
      }
    }
    else{
      if(whiteice::math::abs(d[0]) < 0.00001 ||
	 whiteice::math::abs(d[1]) < 0.00001)
	std::cout << "ERROR: solved eigenvalues are probably wrong (7)"
		  << std::endl;
    }
  }
  catch(std::exception& e){
    std::cout << "ERROR: eig2x2matrix() tests (2). unexcepted exception: " 
	      << e.what() << std::endl;
  }
  
  
  // TEST CASE 3 - tests with articially generated real matrix
  try
  {
    matrix<float> A, D, X, XX, XXX;
    vertex<float> d;
    
    A.resize(2,2);
    X.resize(2,2);
    
    D.resize(2,2);
    D(0,0) = ((float)rand()/((float)RAND_MAX) - 0.5);
    D(1,1) = ((float)rand()/((float)RAND_MAX) - 0.5);
    
    XX.resize(2,2);
    XXX.resize(2,2);
    XX(0,0) = ((float)rand()/((float)RAND_MAX) - 0.5);
    XX(0,1) = ((float)rand()/((float)RAND_MAX) - 0.5);
    XX(1,0) = ((float)rand()/((float)RAND_MAX) - 0.5);
    XX(1,1) = ((float)rand()/((float)RAND_MAX) - 0.5);    
    
    // 
    // XX.transpose();
    // XX[0].normalize(); // normalize row 0
    // XX[1].normalize(); // normalize row 1
    // XX.transpose();
    // 
    
    // normalizes each column to have unit length
    XX.transpose(); // columns -> rows
    XX.normalize(); // normalizes each row
    XX.transpose(); // rows -> columns
    
    
    XXX = XX;
    XXX.inv();
    
    A = XX*D*XXX;
    
    if(eig2x2matrix(A, d, X, false) == false){
      std::cout << "ERROR: (real) non-complex eig2x2matrix() failed "
		<< "with real matrix with real eigenvalues." << std::endl;
    }
    else{
      // compares eigenvalues
      if((whiteice::math::abs(D(0,0) - d[0]) + 
	  whiteice::math::abs(D(1,1) - d[1])) > 0.0001 &&
	 (whiteice::math::abs(D(0,0) - d[1]) + 
	  whiteice::math::abs(D(1,1) - d[0])) > 0.0001){
	std::cout << "ERROR: (real) eig2x2matrix() returned bad eigenvalues:\n"
		  << "correct: " << d << std::endl
		  << "calculated diagonal matrix: " << D << std::endl;
      }
      
      
#if 0
      // compares eigenvectors
      
      // creates copy of correct ones with columns/vectors swapped
      XXX = XX;
      XXX.transpose();
      
      // std::swap< vertex<float> >(XXX[0],XXX[1]);
      for(unsigned int i=0;i<2;i++)
	std::swap< float >(XXX(0, i), XXX(1, i));
      
      XXX.transpose();

      XXX.transpose();
      XX.transpose();
      X.transpose();
      
      if( (whiteice::math::real(norm_inf(XXX[0] - X[0])) + 
	   whiteice::math::real(norm_inf(XXX[1] - X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XXX[0] + X[0])) + 
	   whiteice::math::real(norm_inf(XXX[1] - X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XXX[0] + X[0])) - 
	   whiteice::math::real(norm_inf(XXX[1] + X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XXX[0] + X[0])) + 
	   whiteice::math::real(norm_inf(XXX[1] + X[1]))) > 0.0001 &&

	  (whiteice::math::real(norm_inf(XX[0] - X[0])) + 
	   whiteice::math::real(norm_inf(XX[1] - X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XX[0] + X[0])) + 
	   whiteice::math::real(norm_inf(XX[1] - X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XX[0] + X[0])) - 
	   whiteice::math::real(norm_inf(XX[1] + X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XX[0] + X[0])) + 
	   whiteice::math::real(norm_inf(XX[1] + X[1]))) > 0.0001 )
	{
	  std::cout << "ERROR: (real) eig2x2matrix() returned bad eigenvectors:\n"
		    << "correct: " << XX << std::endl
		    << "calculated: " << X << std::endl;
	  
	  std::cout << "correct eigevalues: " 
		    << D(0,0) << " " << D(1,1) << std::endl;
	  std::cout << "solved eigenvalues: " 
		    << d << std::endl;
	  
	}
#endif
      
    }
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: eig2x2matrix() tests (3). unexcepted exception: " 
	      << e.what() << std::endl;
  }
  
  
  // TEST CASE 4 - tests with articially generated complex matrix
  try
  {
    matrix< whiteice::math::complex<float> > A, D, X, XX, XXX;
    vertex< whiteice::math::complex<float> > d;
    
    A.resize(2,2);
    X.resize(2,2);
    
    D.resize(2,2);
    D(0,0) = whiteice::math::complex<float>( ((float)rand()/((float)RAND_MAX) - 0.5),
					     ((float)rand()/((float)RAND_MAX) - 0.5) );
    
    D(1,1) = whiteice::math::complex<float>( ((float)rand()/((float)RAND_MAX) - 0.5),
					     ((float)rand()/((float)RAND_MAX) - 0.5) );
    XX.resize(2,2);
    XXX.resize(2,2);
    
    XX(0,0) = whiteice::math::complex<float>( ((float)rand()/((float)RAND_MAX) - 0.5),
					      ((float)rand()/((float)RAND_MAX) - 0.5) );
      
    XX(0,1) = whiteice::math::complex<float>( ((float)rand()/((float)RAND_MAX) - 0.5),
					      ((float)rand()/((float)RAND_MAX) - 0.5) );
    XX(1,0) = whiteice::math::complex<float>( ((float)rand()/((float)RAND_MAX) - 0.5),
					      ((float)rand()/((float)RAND_MAX) - 0.5) );
    XX(1,1) = whiteice::math::complex<float>( ((float)rand()/((float)RAND_MAX) - 0.5),
					      ((float)rand()/((float)RAND_MAX) - 0.5) );
    XX.transpose();
    XX.normalize(); // normalizes XX(0,:), X(1,:)
    XX.transpose();
    
    XXX = XX;
    XXX.inv();
    
    A = XX*D*XXX;
    
    if(eig2x2matrix(A, d, X, true) == false){
      std::cout << "ERROR: complex eig2x2matrix() failed\n";
    }
    else{
      // compares eigenvalues
      if((std::real(whiteice::math::abs(D(0,0) - d[0])) + 
	  std::real(whiteice::math::abs(D(1,1) - d[1]))) > 0.0001 &&
	 (std::real(whiteice::math::abs(D(0,0) - d[1])) + 
	  std::real(whiteice::math::abs(D(1,1) - d[0]))) > 0.0001){
	
	std::cout << "ERROR eig2x2matrix() returned bad eigenvalues:\n"
		  << "correct: " << d << std::endl
		  << "calculated diagonal matrix: " << D << std::endl;
      }

#if 0
      // compares eigenvectors
      
      // creates copy of correct ones with columns/vectors swapped
      XXX = XX;
      XXX.transpose();
      std::swap< vertex< whiteice::math::complex<float> > >(XXX[0], XXX[1]);
      XXX.transpose();
      
      XXX.transpose();
      XX.transpose();
      X.transpose();

      
      if( (whiteice::math::real(norm_inf(XXX[0] - X[0])) + 
	   whiteice::math::real(norm_inf(XXX[1] - X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XXX[0] + X[0])) + 
	   whiteice::math::real(norm_inf(XXX[1] - X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XXX[0] + X[0])) - 
	   whiteice::math::real(norm_inf(XXX[1] + X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XXX[0] + X[0])) + 
	   whiteice::math::real(norm_inf(XXX[1] + X[1]))) > 0.0001 &&

	  (whiteice::math::real(norm_inf(XX[0] - X[0])) + 
	   whiteice::math::real(norm_inf(XX[1] - X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XX[0] + X[0])) + 
	   whiteice::math::real(norm_inf(XX[1] - X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XX[0] + X[0])) - 
	   whiteice::math::real(norm_inf(XX[1] + X[1]))) > 0.0001 &&
	  
	  (whiteice::math::real(norm_inf(XX[0] + X[0])) + 
	   whiteice::math::real(norm_inf(XX[1] + X[1]))) > 0.0001 )
	{
	  std::cout << "ERROR eig2x2matrix() returned bad eigenvectors:\n"
		    << "correct: " << XX << std::endl
		    << "calculated: " << X << std::endl;

	  std::cout << "correct eigevalues: " 
		    << D(0,0) << " " << D(1,1) << std::endl;
	  std::cout << "solved eigenvalues: " 
		    << d << std::endl;
	}
#endif

    }
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: eig2x2matrix() tests (4). unexcepted exception: " 
	      << e.what() << std::endl;
  }

  
  std::cout << "WARNING: eig2x2matrix() special singular values are not tested."
	    << std::endl;
  
  
  
  /*
  std::cout << "ERROR: SYLVESTER EQUATION SOLVER WERE BROKEN DURING THE LATEST MATRIX CODE UPDATE"
	    << std::endl;
  */

  
  //////////////////////////////////////////////////
  // sylvester equation
  
  // TEST CASE 1 - trivial case
  // AX - XB = C , where A, B are identity matrices and C = zero
  try
  {
    unsigned int k = 3 + (rand() % 30); // random matrix size
    
    matrix<double> A,B,C,X;
    A.resize(k,k);
    B.resize(k,k);
    C.resize(k,k);
    X.resize(k,k);
    
    A.identity();
    B.identity();
    C.zero();
    X = C;
    
    solve_sylvester(A,B,C,0,0,k,0,0,k);
    
    X = A*C - C*B;
    
    if(norm_inf(X) > 0.0001){
      std::cout << "ERROR: sylvester equation solver didn't found correct solution (1)"
		<< std::endl;
    }    
  }
  catch(std::exception& e){
    std::cout << "ERROR: sylvester equation solver tests (1). unexcepted exception: " 
	      << e.what() << std::endl;
  }
  
  
  // TEST CASE 2 - no solution case
  // AX - XB = C, where A, B and C are identity matrices, so => 0 = I
  try
  {
    unsigned int k = 3 + (rand() % 30); // random matrix size
    
    matrix<double> A,B,C,X;
    A.resize(k,k);
    B.resize(k,k);
    C.resize(k,k);
    X.resize(k,k);
    
    A.identity();
    B.identity();
    C.identity();
    X = C;
    
    solve_sylvester(A,B,C,0,0,k,0,0,k);
    
    std::cout << "TESTCASE2: unsolvable equation - eq. solver gave:\n"
	      << C << " as a solution" << std::endl;
    
    // TODO: change eq solver to return false if there's no solution
  }
  catch(std::exception& e){
    std::cout << "ERROR: sylvester equation solver tests (2). unexcepted exception: " 
	      << e.what() << std::endl;
  }
  
  
  // TEST CASE 3 - create random real A, B and X and calculate C (solution exists),
  // calculate and check that found solution is ok. (there can be many)
  // AX - XB = C
  try
  {
    unsigned int k = 3 + (rand() % 30); // random matrix size
    unsigned int l = 3 + (rand() % 30);
    
    matrix<double> A,B,C,X;
    A.resize(k,k);
    B.resize(l,l);
    C.resize(k,l);
    X.resize(k,l);
 
    for(unsigned int j=0;j<A.ysize();j++)
      for(unsigned int i=0;i<A.xsize();i++)
	A(j,i) = ((float)rand()/((float)RAND_MAX) - 0.5);
    
    for(unsigned int j=0;j<B.ysize();j++)
      for(unsigned int i=0;i<B.xsize();i++)
	B(j,i) = ((float)rand()/((float)RAND_MAX) - 0.5);

    for(unsigned int j=0;j<X.ysize();j++)
      for(unsigned int i=0;i<X.xsize();i++)
	X(j,i) = ((float)rand()/((float)RAND_MAX) - 0.5);
    
    C = A*X - X*B;
    X = C;
    
    solve_sylvester(A,B,X,0,0,k,0,0,l);
    
    if(norm_inf(A*X - X*B - C) > 0.0001){
      std::cout << "ERROR: sylvester equation solver failed to give good enough result"
		<< std::endl;
    }
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: sylvester equation solver tests (3). unexcepted exception: " 
	      << e.what() << std::endl;
  }
  
  

  // TEST CASE 4 - create random *complex* A, B and X and calculate C
  // (solution exists) and calculate and check that found solution is ok.
  // (there can be many solutions to the equation)
  try
  {
    unsigned int k = 3 + (rand() % 30); // random matrix size
    unsigned int l = 3 + (rand() % 30);
    
    matrix< whiteice::math::blas_complex<double> > A,B,C,X;
    A.resize(k,k);
    B.resize(l,l);
    C.resize(k,l);
    X.resize(k,l);
 
    for(unsigned int j=0;j<A.ysize();j++)
      for(unsigned int i=0;i<A.xsize();i++)
	A(j,i) = whiteice::math::blas_complex<double>( ((float)rand()/((float)RAND_MAX) - 0.5),
							((float)rand()/((float)RAND_MAX) - 0.5) );
    
    for(unsigned int j=0;j<B.ysize();j++)
      for(unsigned int i=0;i<B.xsize();i++)
	B(j,i) = whiteice::math::blas_complex<double>( ((float)rand()/((float)RAND_MAX) - 0.5),
							((float)rand()/((float)RAND_MAX) - 0.5) );

    for(unsigned int j=0;j<X.ysize();j++)
      for(unsigned int i=0;i<X.xsize();i++)
	X(j,i) = whiteice::math::blas_complex<double>( ((float)rand()/((float)RAND_MAX) - 0.5),
							((float)rand()/((float)RAND_MAX) - 0.5) );
    
    C = A*X - X*B;
    X = C;
    
    solve_sylvester(A,B,X,0,0,k,0,0,l);
    
    if(norm_inf(A*X - X*B - C) > 0.0001){
      std::cout << "ERROR: sylvester equation solver failed to give good enough result"
		<< std::endl;
    }
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: sylvester equation solver tests (4). unexcepted exception: " 
	      << e.what() << std::endl;
  }
  
  
}



void test_eigenproblem_tests()
{
  std::cout << "TESTS FOR EIGENVALUE PROBLEM ALGORITHMS" << std::endl;
  
  try{
    matrix< blas_real<float> > A, R, Q;
    bool ok = true;

    A.resize(10,10);
    
    for(unsigned int index=0;index<A.xsize()*A.ysize();index++)
      A[index] = ((float)rand())/(float)RAND_MAX;
    
    R = A;
    
    if(qr(R,Q) == false){
      ok = false;
      std::cout << "ERROR: CALCULATION OF QR DECOMPOSITION FAILED." 
		<< std::endl;
    }
    else{
      R = Q*R;
      
      R -= A;
      
      blas_real<float> error = 0.0f;
      
      for(unsigned int index=0;index<A.xsize()*A.ysize();index++){
	error += whiteice::math::abs(R[index]);
      }
      
      if(error > 0.001f){
	ok = false;
	std::cout << "ERROR: incorrect QR-decompostion, A != Q*R" << std::endl;
      }
      
      
      R = Q;
      R.transpose();
      Q *= R;
      
      error = 0.0f;
      
      for(unsigned int j=0;j<Q.ysize();j++){
	for(unsigned int i=0;i<Q.xsize();i++){
	  if(i == j){
	    error += whiteice::math::abs(Q(j,i) - 1.0f);
	  }
	  else{
	    error += whiteice::math::abs(Q(j,i));
	  }
	}
      }
      
      if(error > 0.001f){
	ok = false;
	std::cout << "ERROR: incorrect QR-decompostion, Q*Q^t != I" << std::endl;
	std::cout << "Q*Q^t = " << Q << std::endl;
      }
      
      
    }

    if(ok)
      std::cout << "QR DECOMPOSITION: PASSED ALL TESTS." << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: QR decomposition tests failed: "
	      << e.what() << std::endl;
  }
  
  // hessenberg reduction
  try{
    matrix< blas_real<double> > A,Q, AA, QQ;
    A.resize(4,4); Q.resize(4,4);
    
    A(0,0) =  1; A(0,1) = -1; A(0,2) =  0; A(0,3) =  0;
    A(1,0) =  5; A(1,1) =  3; A(1,2) =  8; A(1,3) =  1;
    A(2,0) =  9; A(2,1) =  1; A(2,2) =  5; A(2,3) =  2;
    A(3,0) = -2; A(3,1) =  4; A(3,2) =  6; A(3,3) =  8;
    
    
    if(hessenberg_reduction(A, Q) == false){
      std::cout << "ERROR: HESSENBERG REDUCTION FAILED" << std::endl;
      
    }
    else{ // checks for correctness
      
      AA = A;
      QQ = Q;
      
      A(0,0) -=  1.0000;  A(0,1) -=  0.4767; A(0,2) -= -0.5846; A(0,3) -=  0.6565;
      A(1,0) -= -10.4881; A(1,1) -=  6.5727; A(1,2) -= -0.3976; A(1,3) -= -2.8034;
      A(2,0) -=  0.0;     A(2,1) -= -8.5029; A(2,2) -=  6.8403; A(2,3) -=  4.9585;
      A(3,0) -=  0.0;     A(3,1) -=     0.0; A(3,2) -=  4.2911; A(3,3) -=  2.5870;
      
      Q(0,0) -=  1.0000;  Q(0,1) -=  0;      Q(0,2) -=  0;      Q(0,3) -=  0;
      Q(1,0) -=  0;       Q(1,1) -= -0.4767; Q(1,2) -=  0.5846; Q(1,3) -= -0.6565;
      Q(2,0) -=  0;       Q(2,1) -= -0.8581; Q(2,2) -= -0.1475; Q(2,3) -=  0.4918;
      Q(3,0) -=  0;       Q(3,1) -=  0.1907; Q(3,2) -=  0.7978; Q(3,3) -=  0.5720;
      
      if(norm_inf(A) > 0.01 || norm_inf(Q) > 0.01){
	std::cout << "ERROR: hessenberg reduction gave wrong results" << std::endl;
	std::cout << "ERROR-A: " << A << std::endl;
	std::cout << "ERROR-Q: " << Q << std::endl;
	std::cout << "RESULT-A: " << AA << std::endl;
	std::cout << "RESULT-Q: " << QQ << std::endl;
      }
      else{
	std::cout << "HESSENBERG REDUCTION TEST: PASSED" << std::endl;
      }
    }
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: hessenberg reduction tests failed: "
	      << e.what() << std::endl;
  }
  
  
  
  // symmetric eigenproblem solver
  try{
    std::cout << "SYMMETRIC EIGENVALUE SOLVER TESTS" << std::endl;
    
    bool ok = true;
    matrix< blas_real<double> > A, X, D;
    A.resize(4,4);        
    
    A(0,0) =  22.000; A(0,1) =  38.000; A(0,2) = -2.0000; A(0,3) =  46.000;
    A(1,0) =  38.000; A(1,1) =  115.00; A(1,2) = -7.0000; A(1,3) =  133.00;
    A(2,0) = -2.0000; A(2,1) = -7.0000; A(2,2) =  28.000; A(2,3) =  0.0000;
    A(3,0) =  46.000; A(3,1) =  133.00; A(3,2) =  0.0000; A(3,3) =  162.00;
    
    D = A;
    
    if(symmetric_eig(D, X) == false){
      ok = false;
      std::cout << "ERROR: SYMMETRIC EIG FAILED" << std::endl;
      
    }
    else{ // checks for correctness
      
      // std::cout << "symmetric evd algorithm: " << std::endl;
      // std::cout << "compare these values manually" << std::endl;
      // std::cout << "D = " << D << std::endl;
      // std::cout << "X = " << X << std::endl;
      
      // this isn't that easy because EVD isn't unique
      
      matrix< blas_real<double> > CD, CX;
      CD.resize(4,4); CX.resize(4,4);
      
      CD(0,0)  =  12.6467; CD(0,1)  =  0.0000; CD(0,2)  =  0.0000; CD(0,3)  =  0.0000;
      CD(1,0)  =  0.00000; CD(1,1)  = -3.4616; CD(1,2)  =  0.0000; CD(1,3)  =  0.0000;
      CD(2,0)  =  0.00000; CD(2,1)  =  0.0000; CD(2,2)  = -4.9657; CD(2,3)  =  0.0000;
      CD(3,0)  =  0.00000; CD(3,1)  =  0.0000; CD(3,2)  =  0.0000; CD(3,3)  =  1.8492;
      
      CX(0,0)  = -0.2117; CX(0,1)  = -0.6850; CX(0,2)  = -0.2362; CX(0,3)  =  0.3191;
      CX(1,0)  = -0.5375; CX(1,1)  =  0.4393; CX(1,2)  =  0.1424; CX(1,3)  = -0.3791;
      CX(2,0)  = -0.2331; CX(2,1)  = -0.5129; CX(2,2)  =  0.9092; CX(2,3)  = -0.5630;
      CX(3,0)  = -0.7823; CX(3,1)  =  0.2734; CX(3,2)  =  0.3120; CX(3,3)  =  0.6614;
      
      // not done yet:
      // the way to do this: find out correct permutation for eigenvalues and
      // check error is small enough. check also X * X^t = I and that
      // A = X*D*X^t
      
      CX = X;
      CX.transpose();
      CX = CX * X;
      
      
      blas_real<double> error = 0.0;
      
      for(unsigned int j=0;j<CX.ysize();j++){
	for(unsigned int i=0;i<CX.xsize();i++){
	  if(i == j)
	    error += whiteice::math::abs(CX(j,i) - 1.0);
	  else
	    error += whiteice::math::abs(CX(j,i));
	}
      }
      
      if(error > 0.001){
	ok = false;
	std::cout << "ERROR: EVD: X^t * X != I" << std::endl;
	std::cout << "X * X^t = " << CX << std::endl;
      }
      
      CX = X; CX.transpose();
      CX = X * D * CX; // CX = X * D * X^t
      
      error = 0.0;
      
      for(unsigned int j=0;j<A.ysize();j++){
	for(unsigned int i=0;i<A.xsize();i++){
	  error += whiteice::math::abs(A(j,i) - CX(j,i));
	}
      }
      
      if(error > 0.001){
	ok = false;
	std::cout << "ERROR: EVD: A != X * D * X^t" << std::endl;
	std::cout << "A = " << A << std::endl;
	std::cout << "X * D * X^t = " << CX << std::endl;
      }                  
      
    }
    
    
    // BIG A TEST
    {
      A.resize(100,100);
      
      for(unsigned int j=0;j<A.ysize();j++)
	for(unsigned int i=0;i<A.xsize();i++)
	  A(j,i) = 1.0f + ((double)rand()/(double)RAND_MAX);
      
      D = A;
      D.transpose();
      A *= D; // A is symmetric now
      
      D = A;
      
      if(symmetric_eig(D, X) == false){
	ok = false;
	std::cout << "ERROR: SYMMETRIC EIG FAILED" << std::endl;
	
      }
      else{ // checks for correctness
	
	matrix< blas_real<double> > C;
	
	C = X*D;
	X.transpose();
	C = C*X;
	
	C -= A;
	
	blas_real<double> error = 0.0;
	
	for(unsigned int j=0;j<C.ysize();j++)
	  for(unsigned int i=0;i<C.xsize();i++)
	    error += whiteice::math::abs(C(j,i));
	
	
	if(error > 0.001){
	  ok = false;	  
	  std::cout << "ERROR: EVD: X^t * D * X != A" << std::endl;
	  std::cout << "error = " << error << std::endl;
	}	
	
      }
    }
    
    
    std::cout << "WARNING: EVD TESTS ARE INCOMPLETE" << std::endl;
    
    if(ok)
      std::cout << "SYMMETRIC EIGENVALUE SOLVER TESTS: PASSED" << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: symmetric eigenproblem solver tests failed: "
	      << e.what() << std::endl;
  }
  
  
  
  // francis qr step
  // schur form calculation
  
  // invpowermethod eigenvector
  //  solver: separated eigenvalues, multiple same, singular
  
  // (not yet implemented):
  // fast_sylvester_solve (recursive sylvester equation solver)
  // block_diagonalization
  
  // eig - generic eigenproblem solver (above + solve 2x2 blocks with eig2x2matrix)
  // + inverse multiply block diagonalized rotation matrix P with
  // 2x2 blocks eigenvector (X) matrix (handle special cases where X is singular
  // correctly)

  
  
  // PCA calculation test
  try{
    matrix< blas_real<float> > R, V;
    std::vector< vertex< blas_real<float> > > list;
    unsigned int DIM = 5, SIZE = 1000;
    list.resize(SIZE);
    R.resize(DIM,DIM);
    
    std::cout << "WARNING: NO PROPER PCA TESTS" << std::endl;
    
    for(unsigned int i=0;i<SIZE;i++){
      list[i].resize(DIM);
      list[i][0] = (rand() / ((float)RAND_MAX)) - 0.5;
      list[i][1] = (rand() / ((float)RAND_MAX)) - 0.5;
      list[i][2] = (rand() / ((float)RAND_MAX)) - 0.5;
      list[i][3] = (rand() / ((float)RAND_MAX)) - 0.5;
      list[i][4] = (rand() / ((float)RAND_MAX)) - 0.5;
    }
    
    if(autocorrelation(R, list) == false){
      std::cout << "ERROR: autocorrelation calculation failed" << std::endl;
    }
    
    // std::cout << "R = " << R << std::endl;
    
    if(symmetric_eig(R, V) == false){
      std::cout << "ERROR: eigenvalue calculation failed" << std::endl;
    }
    
    // std::cout << "D = " << R << std::endl;
    // std::cout << "V = " << V << std::endl;
    
    V.transpose();
    
    for(unsigned int i=0;i<SIZE;i++)
      list[i] = V * list[i];
    
    if(autocorrelation(R, list) == false){
      std::cout << "ERROR: autocorrelation calculation failed" << std::endl;
    }
    
    // std::cout << "R = " << R << std::endl;
    
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: PCA tests failed: "
	      << e.what() << std::endl;
  }

  
  // SVD calculation test
  try{
    std::cout << "SVD CALCULATION TESTS" << std::endl;
    bool ok = true;
    
    matrix< blas_real<float> > A, B;
    matrix< blas_real<float> > AS, AU, AV, BS, BU, BV;
    matrix< blas_real<float> > T;
    
    A.resize(3,5);
    
    A(0,0) = 0.9228; A(0,1) = 0.0226; A(0,2) = 0.6514; A(0,3) = 0.3667; A(0,4) = 0.4829;
    A(1,0) = 0.6568; A(1,1) = 0.0972; A(1,2) = 0.4556; A(1,3) = 0.4237; A(1,4) = 0.0697;
    A(2,0) = 0.9928; A(2,1) = 0.8410; A(2,2) = 0.7007; A(2,3) = 0.6735; A(2,4) = 0.8934;
    
    B = A;
    B.transpose();
    
    AS = A;
    BS = B;
    
    if(svd(AS, AU, AV) == false){
      std::cout << "ERROR: svd of A failed" << std::endl;
      ok = false;
    }
    
    if(svd(BS, BU, BV) == false){
      std::cout << "ERROR: svd of B failed" << std::endl;
      ok = false;
    }
    
    // checks three first eigenvalues are same (fails if values are in different order)
    if(norm_inf(AS - BS) > 0.001){
      std::cout << "ERROR: singular value mismatch" << std::endl;
      std::cout << "AS = " << AS << std::endl;
      std::cout << "BS = " << BS << std::endl;
      ok = false;
    }

    AV.transpose();
    T = AU * AS * AV;
    if(norm_inf(T - A) > 0.001){
      std::cout << "ERROR: bad singular value decomposition" << std::endl;
      std::cout << "A           = " << A << std::endl;
      std::cout << "U * S * V^t = " << T << std::endl;
      ok = false;
    }
    
    BV.transpose();
    T = BU * BS * BV;
    if(norm_inf(T - B) > 0.001){
      std::cout << "ERROR: bad singular value decomposition" << std::endl;
      std::cout << "B           = " << B << std::endl;
      std::cout << "U * S * V^t = " << T << std::endl;
      ok = false;
    }
    
    if(ok == true)
      std::cout << "SINGULAR VALUE DECOMPOSITION TESTS: PASSED" << std::endl;
  }
  catch(std::exception& e){
    std::cout << "ERROR: SVD tests failed: "
	      << e.what() << std::endl;
  }
  
  std::cout << std::endl;
}



void test_ica()
{
  std::cout << "TESTS FOR INFORMATION THEORETIC ALGORITHMS" << std::endl;
  
  // ICA + ADD BSC TESTS LATER
  try{
    std::cout << "NON-ITERATIVE ICA TEST" << std::endl;
    
    const unsigned int NUM = 5000;
    matrix< blas_real<float> > SXDATA, SYDATA, XDATA, YDATA;
    SXDATA.resize(NUM, 3);
    SYDATA.resize(NUM, 3);
    XDATA.resize(NUM, 3);
    YDATA.resize(NUM, 3);
    float t;
    
    // generates data (data from the japanese paper)
    
    std::cout << "Generating test data ..." << std::endl;
    
    for(unsigned int i=0;i<NUM;i++){
      t = (i/10000.0f);
      SXDATA(i,0) = 2.0f*((rand()/((float)RAND_MAX)) - 0.5f);
      SXDATA(i,1) = whiteice::math::sin(2.0*M_PI*800.0*t + 
					6.0*whiteice::math::cos(2.0*M_PI*60.0*t));
      SXDATA(i,2) = whiteice::math::sin(2.0*M_PI*90.0*t);
    }
    
    // removes mean and non unit variance from x data
    {
      vertex< blas_real<float> > mean(3);
      float scaling = (1.0f/((float)NUM));
      
      for(unsigned int i=0;i<NUM;i++){
	mean[0] += SXDATA(i,0) * scaling;
	mean[1] += SXDATA(i,1) * scaling;
	mean[2] += SXDATA(i,2) * scaling;
      }
      
      for(unsigned int i=0;i<NUM;i++){
	SXDATA(i,0) -= mean[0];
	SXDATA(i,1) -= mean[1];
	SXDATA(i,2) -= mean[2];
      }
      
      vertex< blas_real<float> > var(3);
      
      for(unsigned int i=0;i<NUM;i++){
	var[0] += SXDATA(i,0)*SXDATA(i,0);
	var[1] += SXDATA(i,1)*SXDATA(i,1);
	var[2] += SXDATA(i,2)*SXDATA(i,2);
      }
      
      var[0] = blas_real<float>(1.0) / whiteice::math::sqrt(var[0]);
      var[1] = blas_real<float>(1.0) / whiteice::math::sqrt(var[1]);
      var[2] = blas_real<float>(1.0) / whiteice::math::sqrt(var[2]);
      
      for(unsigned int i=0;i<NUM;i++){
	SXDATA(i,0) = SXDATA(i,0) * var[0];
	SXDATA(i,1) = SXDATA(i,1) * var[1];
	SXDATA(i,2) = SXDATA(i,2) * var[2];
      }
    }
    
    // calculates SYs
    for(unsigned int i=0;i<NUM;i++){
      SYDATA(i,0) = whiteice::math::abs(SXDATA(i,0));
      SYDATA(i,1) = whiteice::math::abs(SXDATA(i,1));
      SYDATA(i,2) = whiteice::math::abs(SXDATA(i,2));
    }
    
    // removes mean and non unit variance
    {
      vertex< blas_real<float> > mean(3);
      float scaling = (1.0f/((float)NUM));
      
      for(unsigned int i=0;i<NUM;i++){
	mean[0] += SYDATA(i,0) * scaling;
	mean[1] += SYDATA(i,1) * scaling;
	mean[2] += SYDATA(i,2) * scaling;
      }
      
      for(unsigned int i=0;i<NUM;i++){
	SYDATA(i,0) -= mean[0];
	SYDATA(i,1) -= mean[1];
	SYDATA(i,2) -= mean[2];
      }
      
      vertex< blas_real<float> > var(3);
      
      for(unsigned int i=0;i<NUM;i++){
	var[0] += SYDATA(i,0)*SYDATA(i,0);
	var[1] += SYDATA(i,1)*SYDATA(i,1);
	var[2] += SYDATA(i,2)*SYDATA(i,2);
      }
      
      var[0] = blas_real<float>(1.0) / whiteice::math::sqrt(var[0]);
      var[1] = blas_real<float>(1.0) / whiteice::math::sqrt(var[1]);
      var[2] = blas_real<float>(1.0) / whiteice::math::sqrt(var[2]);
      
      for(unsigned int i=0;i<NUM;i++){
	SYDATA(i,0) = SYDATA(i,0) * var[0];
	SYDATA(i,1) = SYDATA(i,1) * var[1];
	SYDATA(i,2) = SYDATA(i,2) * var[2];
      }
    }
    
    // linear mixing matrices
    matrix< blas_real<float> > AX, AY;
    matrix< blas_real<float> > AXt, AYt;
    AX.resize(3,3);
    AY.resize(3,3);
    
    AX(0,0) = +0.0f; AX(0,1) = -1.0f; AX(0,2) = +1.0f;
    AX(1,0) = +1.0f; AX(1,1) = +1.0f; AX(1,2) = +0.0f;
    AX(2,0) = +1.0f; AX(2,1) = +0.0f; AX(2,2) = -1.0f;
    
    AY(0,0) = +2.0f; AY(0,1) = -2.0f; AY(0,2) = +3.0f;
    AY(1,0) = +2.0f; AY(1,1) = +1.0f; AY(1,2) = +0.0f;
    AY(2,0) = +1.0f; AY(2,1) = +2.0f; AY(2,2) = +6.0f;
    
    AXt = AX;
    AYt = AY;
    AXt.transpose();
    AYt.transpose();
    
    // calculates observed data
    XDATA = SXDATA * AXt;
    YDATA = SYDATA * AYt;
    
    // calculates ICA solution for XDATA
    
    matrix< blas_real<float> > W;
    if(ica(XDATA,W, true) == false){
      std::cout << "ERROR:  ICA failed" << std::endl;
    }
    else{
      std::cout << "ICA solved." << std::endl;
    }
    
    //////////////////////////////////////////////////
    // saves data to file
    {
      conffile datafile;
      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      
      // saves matrix SX
      
      ints.clear();
      ints.push_back(SXDATA.ysize());
      ints.push_back(SXDATA.xsize());
      datafile.set("SX_SIZE", ints);
      
      for(unsigned int i=0;i<NUM;i++){
	char buf[50];
	sprintf(buf,"SX_ROW%d", i);

	floats.clear();
	floats.push_back(SXDATA(0,1).value());
	floats.push_back(SXDATA(1,i).value());
	floats.push_back(SXDATA(2,i).value());
	
	datafile.set(buf, floats);
      }
      
      //////////////////////////////////////////////////
      // saves mixing matrix AX

      ints.clear();
      ints.push_back(AX.ysize());
      ints.push_back(AX.xsize());
      datafile.set("AX_SIZE", ints);
      
      for(unsigned int i=0;i<3;i++){
	char buf[50];
	sprintf(buf,"AX_ROW%d", i);
	
	floats.clear();
	for(unsigned int j=0;j<3;j++)
	  floats.push_back(AX(i,j).value());
	
	datafile.set(buf, floats);
      }
      
      //////////////////////////////////////////////////
      // saves matrix X
      
      ints.clear();
      ints.push_back(XDATA.ysize());
      ints.push_back(XDATA.xsize());
      datafile.set("X_SIZE", ints);

      for(unsigned int i=0;i<NUM;i++){
	char buf[50];
	sprintf(buf,"X_ROW%d", i);
	
	floats.clear();
	floats.push_back(XDATA(0,i).value());
	floats.push_back(XDATA(1,i).value());
	floats.push_back(XDATA(2,i).value());
	
	datafile.set(buf, floats);
      }
      
      //////////////////////////////////////////////////
      // saves demixing matrix W
      
      ints.clear();
      ints.push_back(W.ysize());
      ints.push_back(W.xsize());
      datafile.set("W_SIZE", ints);
      
      for(unsigned int i=0;i<3;i++){
	char buf[50];
	sprintf(buf,"W_ROW%d", i);
	
	floats.clear();
	for(unsigned int j=0;j<3;j++)
	  floats.push_back(W(i,j).value());
	
	datafile.set(buf, floats);
      }

      if(datafile.save("icaresults.data") == false){
	std::cout << "SAVING icaresults.data failed"
		  << std::endl;
      }
      else{
	// at least for now:
	std::cout << "SAVED ICA RESULTS TO FILE icaresults.data "
		  << "TEST / VISUALIZE MANUALLY THAT SOLUTION IS "
		  << " CORRECT."
		  << std::endl;
      }
    }
        
  }
  catch(std::exception& e){
    std::cout << "ERROR: ICA tests failed: "
	      << e.what() << std::endl;
  }
  
  // LINEAR (SVD/BATCH) BSC
  
  // LINEAR ITERATIVE BSC
  
  // UNPARAMETRIC COMPOMENTS ANALYSIS (UCA) (SIGNAL SPECIFICATION BASED BSC)
  // (LINEAR INFORMATION THEORETIC NEURAL NETWORK)
  
  // NON LINEAR BSC (INFORMATION THEORETIC NEURAL NETWORK)
  // - this should be good because it maximizes information
  //   between I(NN(X), Y) in a way that f(X) has also
  //   same representation as Y. => in theory perfect
  //   neuralnetwork, in practice (current) non-linear BSC
  //   calculation approach is somewhat (but only a little)
  //   heuristic
  
  
}


