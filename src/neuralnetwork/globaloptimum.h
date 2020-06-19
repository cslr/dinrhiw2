/*
 * Implements approximately global optimizer pretraining of neural network code.
 * 
 * Tomas Ukkonen, tomas.ukkonen@iki.fi, 2020
 *
 */
#ifndef globaloptimum_nn_h
#define globaloptimum_nn_h

#include "matrix.h"
#include "vertex.h"
#include "dataset.h"
#include "nnetwork.h"
#include <vector>

namespace whiteice
{
  // learns N global linear optimizers (parameter K) u=vec(A,b) parameters u using randomized
  // neural network weights w and randomly N(0,I) generated training data (M training datapoints)
  // learn globally optimal linear optimum mapping from f(u)-> w and trains original problem using
  // globally optimal solution u and gets preset neural network weights w using learnt mapping
  // assumes input data is N(0,I) distributed + assumes output data is close to N(0,I) too.
  template <typename T>
    bool global_optimizer_pretraining(nnetwork<T>& net,
				      const dataset<T>& data, 
				      const unsigned int N = 10000,
				      const unsigned int M = 10000,
				      const unsigned int K = 16);
				      

  
  // solves global linear optimum by discretizing data to K bins per dimension
  // it is assumed that input data x is normalized to have zero mean = 0 and
  // unit variance/standard deviation = 1
  // solves global optimum of pseudolinear "y = Ax + b" by solving least squares problem
  template <typename T>
    bool global_linear_optimizer(const whiteice::dataset<T>& data,
				 const unsigned int K,
				 whiteice::math::matrix<T>& A,
				 whiteice::math::vertex<T>& b);
				 

  // discretizes each variable to K bins: [-4*stdev, 4*stdev]/K
  // discretization is calculated (K/2)*((x-mean)/4*stdev)+K/2 => [0,K[ (clipped to interval)
  // it is recommended that K is even number
  template <typename T>
    bool discretize_problem(const unsigned int K,
			    const std::vector< math::vertex<T> >& input,
			    std::vector< math::vertex<T> >& inputDiscrete,
			    whiteice::math::vertex<T>& mean,
			    whiteice::math::vertex<T>& stdev);
  
  // discretizes each variable to K bins: [-4*stdev, 4*stdev]/K
  // discretization is calculated (K/2)*((x-mean)/4*stdev)+K/2 => [0,K[ (clipped to interval)
  // it is recommended that K is even number
  // returns value k E [0,K[ or negative number in the case of error
  template <typename T>
    int discretize(const T& x, const unsigned int K, const T& mean, const T& stdev);

  

  //////////////////////////////////////////////////////////////////////

  extern template bool global_optimizer_pretraining
    (nnetwork< math::blas_real<float> >& net,
     const dataset< math::blas_real<float> >& data, 
     const unsigned int N,
     const unsigned int M,
     const unsigned int K);

  extern template bool global_optimizer_pretraining
    (nnetwork< math::blas_real<double> >& net,
     const dataset< math::blas_real<double> >& data, 
     const unsigned int N,
     const unsigned int M,
     const unsigned int K);

  extern template bool global_optimizer_pretraining
    (nnetwork< float >& net,
     const dataset<float>& data, 
     const unsigned int N,
     const unsigned int M,
     const unsigned int K);

  extern template bool global_optimizer_pretraining
    (nnetwork<double>& net,
     const dataset<double>& data, 
     const unsigned int N,
     const unsigned int M,
     const unsigned int K);

  

  extern template bool global_linear_optimizer
    (const whiteice::dataset< math::blas_real<float> >& data,
     const unsigned int K,
     whiteice::math::matrix< math::blas_real<float> >& A,
     whiteice::math::vertex< math::blas_real<float> >& b);
  
  extern template bool global_linear_optimizer
    (const whiteice::dataset< math::blas_real<double> >& data,
     const unsigned int K,
     whiteice::math::matrix< math::blas_real<double> >& A,
     whiteice::math::vertex< math::blas_real<double> >& b);
  
  extern template bool global_linear_optimizer
    (const whiteice::dataset<float>& data,
     const unsigned int K,
     whiteice::math::matrix<float>& A,
     whiteice::math::vertex<float>& b);
  
  extern template bool global_linear_optimizer
    (const whiteice::dataset<double>& data,
     const unsigned int K,
     whiteice::math::matrix<double>& A,
     whiteice::math::vertex<double>& b);
  
  
  extern template bool discretize_problem
    (const unsigned int K,
     const std::vector< math::vertex< math::blas_real<float> > >& input,
     std::vector< math::vertex< math::blas_real<float> > >& inputDiscrete,
     whiteice::math::vertex< math::blas_real<float> >& mean,
     whiteice::math::vertex< math::blas_real<float> >& stdev);
  
  extern template bool discretize_problem
    (const unsigned int K,
     const std::vector< math::vertex< math::blas_real<double> > >& input,
     std::vector< math::vertex< math::blas_real<double> > >& inputDiscrete,
     whiteice::math::vertex< math::blas_real<double> >& mean,
     whiteice::math::vertex< math::blas_real<double> >& stdev);
  
  extern template bool discretize_problem
    (const unsigned int K,
     const std::vector< math::vertex<float> >& input,
     std::vector< math::vertex<float> >& inputDiscrete,
     whiteice::math::vertex<float>& mean,
     whiteice::math::vertex<float>& stdev);     
  
  extern template bool discretize_problem
    (const unsigned int K,
     const std::vector< math::vertex<double> >& input,
     std::vector< math::vertex<double> >& inputDiscrete,
     whiteice::math::vertex<double>& mean,
     whiteice::math::vertex<double>& stdev);

  
  extern template
    int discretize(const math::blas_real<float>& x,
		   const unsigned int K,
		   const math::blas_real<float>& mean,
		   const math::blas_real<float>& stdev);


  extern template
    int discretize(const math::blas_real<double>& x,
		   const unsigned int K,
		   const math::blas_real<double>& mean,
		   const math::blas_real<double>& stdev);
		   

  extern template
    int discretize(const float& x,
		   const unsigned int K,
		   const float& mean,
		   const float& stdev);
  

  extern template
    int discretize(const double& x,
		   const unsigned int K,
		   const double& mean,
		   const double& stdev);
  
};


#endif
