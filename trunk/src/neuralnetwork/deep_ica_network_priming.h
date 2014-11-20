/*
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

};

#endif
