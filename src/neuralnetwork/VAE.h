/*
 * Variational Autoencoder code
 * Tomas Ukkonen <tomas.ukkonen@iki.fi> 2020
 * 
 * Implementation research paper:
 * "Auto-Encoding Variational Bayes. Diedrik Kingma and Max Welling."
 * Implementation extra math notes in hyperplane2 repository (non-free) 
 * [variational-bayes-encoder.tm]
 */

#ifndef __nnetwork_VAE_h
#define __nnetwork_VAE_h

#include "nnetwork.h"
#include "dinrhiw_blas.h"
#include "vertex.h"
#include "conffile.h"
#include "compressable.h"
#include "dataset.h"
#include "MemoryCompressor.h"
#include "RNG.h"

#include <vector>

namespace whiteice
{
  template <typename T = math::blas_real<float> >
    class VAE
    {
    public:

    VAE(const VAE<T>& vae);
    VAE(const nnetwork<T>& encoder, // x -> (z_mean, z_var)
	const nnetwork<T>& decoder) // z -> (x_mean)
      throw(std::invalid_argument);
    
    void getModel(nnetwork<T>& encoder,
		  nnetwork<T>& decoder);
    
    bool setModel(const nnetwork<T>& encoder,
		  const nnetwork<T>& decoder);


    // x -> z (hidden)
    bool encode(const math::vertex<T>& x,
		math::vertex<T>& zmean,
		math::vertex<T>& zstdev) const;
    
    // z (hidden) -> x
    bool decode(const math::vertex<T>& z,
		math::vertex<T>& xmean) const;

    unsigned int getDataDimension() const;
    unsigned int getEncodedDimension() const;
    
    void getParameters(math::vertex<T>& p) const;
    bool setParameters(const math::vertex<T>& p);
    
    void initializeParameters();

    T getError(const std::vector< math::vertex<T> >& xsamples) const;

    // x = samples, learn x = decode(encode(x))
    // optimizes using gradient descent,
    // divides data to teaching and test
    // datasets (50%/50%)
    //
    // model_stdev_error(last N iters)/model_mean_error(last N iters) < convergence_ratio 
    // 
    bool learnParameters(const std::vector< math::vertex<T> >& xsamples,
			 T convergence_ratio = T(0.01f));

    // calculates gradient of parameter p using all samples
    bool calculateGradient(const std::vector< math::vertex<T> >& xsamples,
			   math::vertex<T>& pgradient);

    private:
    
    nnetwork<T> encoder; // x -> (z mean ,z stdev)
    nnetwork<T> decoder; // z -> (x mean)
    
    };

  
  extern template class VAE< float >;
  extern template class VAE< double >;  
  extern template class VAE< math::blas_real<float> >;
  extern template class VAE< math::blas_real<double> >;
}

#endif
