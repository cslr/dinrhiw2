/*
 * Calculate expectation maximization clustering
 * 
 * Starts with K-means (chosen randomly initially) and
 * iteratively assigns data to cluster which mean is
 * closest to the data.
 */

#ifndef __whiteice_EnsembleMeans_h
#define __whiteice_EnsembleMeans_h

#include "vertex.h"
#include "RNG.h"
#include <vector>


namespace whiteice
{
  template<typename T> 
    class EnsembleMeans
    {
    public:
      EnsembleMeans();
      ~EnsembleMeans();

      bool learn(const unsigned int K,
		 const std::vector< math::vertex<T> >& data);
      
      bool getClustering(std::vector< math::vertex<T> >& kmeans,
			 std::vector< T >& percent) const;

      // gets cluster that has most datapoints (for denoising gradient)
      bool getMajorityCluster(math::vertex<T>& mean) const;
      

    private:
      whiteice::RNG<T> rng;

      std::vector< math::vertex<T> > kmeans;
      std::vector< T > percent;
    
  };

  
  extern template class EnsembleMeans< float >;
  extern template class EnsembleMeans< double >;
  extern template class EnsembleMeans< whiteice::math::blas_real<float> >;
  extern template class EnsembleMeans< whiteice::math::blas_real<double> >;
};



#endif
