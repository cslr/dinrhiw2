/*
 * TESTING:
 * used to test idea where we use aprox. 
 * "non-linear" ICA calculations to initialize nnetwork 
 * weights with coefficients that calculated non-linear
 * ICA approximatedly (N-N-N-X-OUTPUT) and then use those
 * coefficients to find high-quality nnetwork paramters
 * 
 * (here we simply "drop" the non-linearities away when
 *  we do the priming as the non-linearities are nearly
 *  linear close to zeros where most of the data is. We
 *  then hope that gradient descent and optimize the solution
 *  to work properly with values that are not close to zero)
 */

#include "deep_ica_network_priming.h"
#include "eig.h"
#include "correlation.h"
#include <math.h>



namespace whiteice
{
  /**
   * calculates 2*deepness layer "neural network" weights and
   * stores them to parameters. calculations are done recursively
   * by first calculating PCA and one-layer non-linear ICA and
   * then calling this function again with deepness-1.
   *
   * NOTE: currently using tanh(x) non-linearity, sinh(x) would
   *       be better but DO NOT work for nnetwork initialization
   *       which uses tanh(x) 
   */
  bool deep_nonlin_ica(std::vector< math::vertex<> >& data,
		       std::vector<deep_ica_parameters>&
		       parameters, unsigned int deepness)
  {
    if(data.size() <= 0)
      return false;
    
    if(deepness <= 0){ // time to stop
      return (parameters.size() > 0);
    }

    struct deep_ica_parameters p;
    
    // initially we calculate PCA for this layer's data
    {
      // 1. removes mean value
      // 2. calculates covariance matrix Rxx and whitens the data
      
      math::vertex<> mean;
      math::matrix<> Rxx;
      math::matrix<> V;

      if(math::mean_covariance_estimate<>(mean, Rxx, data) == false)
	return false;

      if(math::symmetric_eig<>(Rxx, V) == false)
	return false;

      math::matrix<>& D = Rxx; // Rxx contains now diagnonal matrix

      for(unsigned int i=0;i<Rxx.ysize();i++){
	if(D(i,i) < 0.0)
	  D(i,i) = math::abs(D(i,i));
	if(D(i,i) <= 10e-8){
	  D(i,i) = 0.0; // dirty way to handle singular matrixes (just force the data to zero)
	}
	else{
	  // sets diagonal variances to 0.5
	  math::blas_real<float> d = D(i,i);
	  math::blas_real<float> s = 0.5f;
	  
	  D(i,i) =s/sqrt(d);
	}
      }

      p.W_pca = V * D * V.transpose();
      p.b_pca = - p.W_pca*mean;

      // actual processes the data (PCA)
      for(unsigned int i=0;i<data.size();i++){
	data[i] = p.W_pca * data[i] + p.b_pca;
      }

      if(math::mean_covariance_estimate<>(mean, Rxx, data) == false)
	return false;

      // std::cout << "mean = " << mean << std::endl;
      // std::cout << "Rxx = " << Rxx << std::endl;
    }

    
    /*
     * next we do diagnalize E[sinh(x)sinh(x)^t] matrix 
     * where aim is to remove higher order correlations 
     * from the data
     */
    {
      std::vector< math::vertex<> > sinh_data;
      math::vertex<> sinh_mean;

      for(unsigned int i=0;i<data.size();i++){
	math::vertex<> d1 = data[i];

	for(unsigned int ii=0;ii<d1.size();ii++){
	  // d1[ii] = sinhf(d1[ii].c[0]);
	  d1[ii] = tanhf(d1[ii].c[0]);
	}

	sinh_data.push_back(d1);
      }

      math::vertex<> mean;
      math::matrix<> Rxx;
      math::matrix<> V;

      if(math::mean_covariance_estimate<>(mean, Rxx, sinh_data) == false)
	return false;

      if(math::symmetric_eig<>(Rxx, V) == false)
	return false;
      
      p.W_ica = V.transpose();

      Rxx.identity();
      Rxx = Rxx - p.W_ica;
      
      p.b_ica = Rxx*mean;

      // now we transform data through g^-1(W*g(x)) transform
      // g_new(x) = g(x) - E[g(x)] in order to keep E[g(x)] = 0
      
      for(unsigned int i=0;i<sinh_data.size();i++){
	math::vertex<> x = sinh_data[i];
	x = p.W_ica * x + p.b_ica;

	for(unsigned int ii=0;ii<x.size();ii++)
	  // x[ii] = asinhf(x[ii].c[0]);
	  
	  if(x[ii].c[0] >= 0.9999f)
	    x[ii] = atanhf(0.9999f);
	  else if(x[ii].c[0] <= -0.9999f)
	    x[ii] = atanhf(-0.9999f);
	  else
	    x[ii] = atanhf(x[ii].c[0]);
	  

	// overwrites the data vector with preprocessed data
	data[i] = x; 
      }
    }

    
    // this layer processing is done so we move to the next layer
    {
      parameters.push_back(p);

      return deep_nonlin_ica(data, parameters, deepness - 1);
    }
  }
  

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
			   nnetwork<>& nnet)
  {
    if(parameters.size() <= 0)
      return false;
    
    nnet.randomize();
    
    
    for(unsigned int i=0;i<(2*parameters.size() - 1);i+=2){
      const deep_ica_parameters& p = parameters[i/2];;

      if(nnet.setWeights(p.W_pca, i) == false) return false;
      if(nnet.setBias(p.b_pca, i) == false) return false;

      if(nnet.setWeights(p.W_ica, i+1) == false) return false;
      if(nnet.setBias(p.b_ica, i+1) == false) return false;
    }

    return true;
  }
  
  
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
   *
   * NOTE2: normalization of the last layer does not make
   *        much sense so we don't do it as a default.
   *        (We need to get results that are close to correct ones)
   */
  bool normalize_weights_to_unity(nnetwork<>& nnet,
				  bool normalizeLastLayer)
  {
    std::vector<unsigned int> arch;

    nnet.getArchitecture(arch);

    unsigned int N = arch.size()-1;

    if(normalizeLastLayer == false)
      N = arch.size()-2;

    for(unsigned int i=0;i<N;i++){
      math::matrix<> W;
      math::vertex<> b;
     
      if(!nnet.getWeights(W, i)) return false;

      for(unsigned int j=0;j<W.ysize();j++){
	W.rowcopyto(b, j);
	b.normalize();
	W.rowcopyfrom(b, j);
      }
      
      nnet.setWeights(W,i);

      if(!nnet.getBias(b, i)) return false;
      b.normalize();
      nnet.setBias(b, i);
    }

    return true;
  }

  
};
