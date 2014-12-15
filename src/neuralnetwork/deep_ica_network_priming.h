/*
 * Heuristics for initializing nnetwork weights.
 * 
 * TESTING THIS IDEA:
 * 
 * used to test idea where we use aprox. 
 * "non-linear" ICA calculations to initialize nnetwork 
 * weights with coefficients that calculated non-linear
 * ICA approximatedly (N-N-N-X-OUTPUT) and then use those
 * coefficients to find high-quality nnetwork paramters
 * 
 */

#ifndef deep_ica_network_priming
#define deep_ica_network_priming

#include <vector>
#include "vertex.h"
#include "matrix.h"
#include "nnetwork.h"
#include "dataset.h"

namespace whiteice
{

  /*
   * single 2-layer network parameters that are used
   * to constructed 
   * z = g(W_ica*g(W_pca + b_pca) + b_ica)
   *
   * network layers.
   *
   */
  

  struct deep_ica_parameters
  {
    math::matrix<> W_pca;
    math::vertex<> b_pca;

    math::matrix<> W_ica;
    math::vertex<> b_ica;
  };
  
  
  /**
   * calculates 2*deepness layer "neural network" weights and
   * stores them to parameters. calculations are done recursively
   * by first calculating PCA and one-layer non-linear ICA and
   * then calling this function again with deepness-1.
   */
  bool deep_nonlin_ica(std::vector< math::vertex<> >& data,
		       std::vector<deep_ica_parameters>&
		       parameters, unsigned int deepness);

  /**
   * calculates 2*deepness layer "neural network" weights and
   * stores them to parameters. calculations are done recursively
   * by first calculating PCA and one-layer non-linear ICA and
   * then calling this function again with deepness-1.
   */
  bool deep_nonlin_ica_sinh(std::vector< math::vertex<> >& data,
			    std::vector<deep_ica_parameters>&
			    parameters, unsigned int deepness);
  
  /**
   * constructs nnetwork from deep_nonlin_ica solved parameters  
   * by adding rest_of_arch randomly initiaized extra layers
   * after the deep ica layers. The idea is that we construct a network
   * that extract high-quality (non-linear) ICA solution features and
   * then use gradient descent to get better solutions.
   *
   * This way we may construct 10-50 layer neural network that
   * then uses gradient descent to go to the nearest local minimum.
   *
   */
  bool initialize_nnetwork(const std::vector<deep_ica_parameters>& parameters,
			   nnetwork<>& nnet);
  
  /**
   * function to cause negative feedback between neurons of 
   * each network layer (except the last one). This means that neurons
   * do differentiate to different inputs. This can be used as a training heuristic
   * during learning.
   */
  template <typename T>
    bool negative_feedback_between_neurons(nnetwork<T>& nnet, 
					   const dataset<T>& data,
					   const T& alpha, bool processLastLayer = false);
  
  
  /**
   * calculates ica for each layer and moves weight vectors towards "ICA-subspace".
   * this will mean that neurons do differentiate maximally to different inputs.
   */
  template <typename T>
    bool neuronlayerwise_ica(nnetwork<T>& nnet, const T& alpha, unsigned int layer);
  
  /**
   * calculates linear MSE error solution from last layer to dataset data (cluster 1)
   */
  template <typename T>
    bool neuronlast_layer_mse(nnetwork<T>& nnet, const dataset<T>& data, unsigned int layer);


  /**
   * helper function to normalize neural network weight vectors 
   * ||w|| = 1 and ||b|| = 1 for each layer
   * 
   * (forcing this between every gradient descent steps in directly
   *  forces neural network weights to COMPETE against each other,
   *  this can be also useful in random search and in initialization
   *  step of the neural network weights)
   *
   * NOTE: if input x ~ N(0, I) then 
   *          Var[w^t x] = w^t COV(x) * w = ||w|| = 1 and variance
   *          of the output layer is fixed to be 1, which forces
   *          the problem to be "nice"
   */
  template <typename T>
    bool normalize_weights_to_unity(nnetwork<T>& nnet, bool normalizeLastLayer = false);
  
  
  extern template bool neuronlast_layer_mse<float>(nnetwork<float>& nnet, const dataset<float>& data, unsigned int layer);
  extern template bool neuronlast_layer_mse<double>(nnetwork<double>& nnet, const dataset<double>& data, unsigned int layer);
  extern template bool neuronlast_layer_mse< math::blas_real<float> >(nnetwork< math::blas_real<float> >& nnet,  const dataset< math::blas_real<float> >& data, unsigned int layer);
  extern template bool neuronlast_layer_mse< math::blas_real<double> >(nnetwork< math::blas_real<double> >& nnet, const dataset< math::blas_real<double> >& data, unsigned int layer);
  
  
  extern template bool neuronlayerwise_ica<float>(nnetwork<float>& nnet, const float& alpha, unsigned int layer);
  extern template bool neuronlayerwise_ica<double>(nnetwork<double>& nnet, const double& alpha, unsigned int layer);
  extern template bool neuronlayerwise_ica< math::blas_real<float> >(nnetwork< math::blas_real<float> >& nnet, const math::blas_real<float>& alpha, unsigned int layer);
  extern template bool neuronlayerwise_ica< math::blas_real<double> >(nnetwork< math::blas_real<double> >& nnet, const math::blas_real<double>& alpha, unsigned int layer);
  
  extern template bool negative_feedback_between_neurons<float>(nnetwork<float>& nnet, const dataset<float>& data, const float& alpha, bool processLastLayer);
  extern template bool negative_feedback_between_neurons<double>(nnetwork<double>& nnet, const dataset<double>& data, const double& alpha, bool processLastLayer);
  extern template bool negative_feedback_between_neurons< math::blas_real<float> >(nnetwork< math::blas_real<float> >& nnet, const dataset< math::blas_real<float> >& data, const math::blas_real<float>& alpha, bool processLastLayer);
  extern template bool negative_feedback_between_neurons< math::blas_real<double> >(nnetwork< math::blas_real<double> >& nnet, const dataset< math::blas_real<double> >& data, const math::blas_real<double>& alpha, bool processLastLayer);
  
  extern template bool normalize_weights_to_unity<float>(nnetwork<float>& nnet, bool normalizeLastLayer);
  extern template bool normalize_weights_to_unity<double>(nnetwork<double>& nnet, bool normalizeLastLayer);
  extern template bool normalize_weights_to_unity< math::blas_real<float> >(nnetwork< math::blas_real<float> >& nnet, bool normalizeLastLayer);
  extern template bool normalize_weights_to_unity< math::blas_real<double> >(nnetwork< math::blas_real<double> >& nnet, bool normalizeLastLayer);

};

#endif
