
#include "EnsembleMeans.h"
#include <string.h>

namespace whiteice
{

  template <typename T>
  EnsembleMeans<T>::EnsembleMeans()
  {
    
  }

  template <typename T>
  EnsembleMeans<T>::~EnsembleMeans()
  {
    
  }
  
  template <typename T>
  bool EnsembleMeans<T>::learn(const unsigned int K,
			       const std::vector< math::vertex<T> >& data)
  {
    if(K == 0) return false;

    kmeans.resize(K);

    for(unsigned int k=0;k<K;k++){
      kmeans[k] = data[rng.rand() % data.size()];
    }

    std::vector<unsigned int> assignments(data.size());

    memset(assignments.data(), 0, sizeof(unsigned int)*data.size());
    
    percent.resize(K);

    unsigned int changes = 0;

    do{
      changes = 0;
      // 1. assigns data points to nearest cluster center
      {
	for(unsigned int i=0;i<data.size();i++){
	  unsigned int index = 0;
	  auto delta = (data[i] - kmeans[index]);
	  auto minError = (delta*delta)[0];

	  for(unsigned int k=1;k<K;k++){
	    auto delta = (data[i] - kmeans[k]);
	    auto error = (delta*delta)[0];

	    if(error < minError){
	      minError = error;
	      index = k;
	    }
	  }
	  
	  if(assignments[i] != index)
	    changes++;

	  assignments[i] = index;
	}
	
      }

      // 2. calculates new means from assignments
      {
	for(unsigned int k=0;k<K;k++){
	  kmeans[k].zero();
	  percent[k] = T(0.0);
	}
	
	for(unsigned int i=0;i<data.size();i++){
	  kmeans[assignments[i]] += data[i];
	  percent[assignments[i]]++;
	}

	T sum = T(0.0);

	for(unsigned int k=0;k<K;k++){
	  if(percent[k] > T(0.0))
	    kmeans[k] /= percent[k];

	  sum += percent[k];
	}

	if(sum > T(0.0))
	  for(unsigned int k=0;k<K;k++){
	    percent[k] /= sum;
	  }
      }
      
    }
    while(changes > 0);
    
    return true;
  }

  
  template <typename T>
  bool EnsembleMeans<T>::getClustering(std::vector< math::vertex<T> >& kmeans,
				       std::vector< T >& percent) const
  {
    if(this->kmeans.size() <= 0) return false;
    
    kmeans = this->kmeans;
    percent = this->percent;

    return true;
  }

  template <typename T>
  bool EnsembleMeans<T>::clusterize(const std::vector< math::vertex<T> >& data,
				    std::vector<unsigned int>& cluster)
  {
    if(data.size() <= 0) return true;
    if(kmeans.size() <= 0) return false;

    cluster.resize(data.size());

    for(unsigned int n=0;n<data.size();n++){
      unsigned int index = 0;
      auto delta = data[n] - kmeans[index];
      auto minError = (delta*delta)[0];
      
      for(unsigned int k=1;k<kmeans.size();k++){
	auto delta = data[n] - kmeans[k];
	auto error = (delta*delta)[0];

	if(error < minError){
	  index = k;
	  minError = error;
	}
      }
      
      cluster[n] = index;
    }

    return true;
  }
  

  template <typename T>
  int EnsembleMeans<T>::getMajorityCluster(math::vertex<T>& mean,
					   T& p) const
  {
    if(kmeans.size() <= 0) return -1;

    mean = kmeans[0];
    p = percent[0];
    unsigned int index = 0;

    for(unsigned int k=1;k<kmeans.size();k++){
      if(percent[k] > p){
	p = percent[k];
	mean = kmeans[k];
	index = k;
      }      
    }

    return index;
  }

  
  // gets cluster with p% probability (% of datapoints) (for denoising gradient)
  template <typename T>
  int EnsembleMeans<T>::getProbabilisticCluster(math::vertex<T>& mean,
						T& p) const
  {
    if(kmeans.size() <= 0) return -1;

    mean = kmeans[0];
    p = percent[0];
    
    T r = rng.uniform();

    if(r <= p) return 0;
    
    for(unsigned int k=1;k<kmeans.size();k++){
      p += percent[k];

      if(r <= p){
	p = percent[k];
	mean = kmeans[k];
	return k;
      }
    }
    
    mean = kmeans[kmeans.size()-1];
    p = percent[kmeans.size()-1];

    return true;

  }
  

  template class EnsembleMeans< float >;
  template class EnsembleMeans< double >;
  template class EnsembleMeans< whiteice::math::blas_real<float> >;
  template class EnsembleMeans< whiteice::math::blas_real<double> >;
};
