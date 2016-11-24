
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
    if(K == 0 || data.size() == 0) return false;

    kmeans.resize(K);
    percent.resize(K);
    
    if(K == 1){
      kmeans[0].resize(data[0].size());
      kmeans[0].zero();
      
      /*
      for(unsigned int i=0;i<data.size();i++)
	kmeans[0] += data[i]/T(data.size());
      */

      percent[0] = 1.0;
      return true;
    }

    
    for(unsigned int k=0;k<K;k++){
      kmeans[k] = data[rng.rand() % data.size()];
    }

    std::vector<unsigned int> assignments(data.size());

    memset(assignments.data(), 0, sizeof(unsigned int)*data.size());
    
    unsigned int changes = 0;
    unsigned int loop = 0;

    do{
      changes = 0;

      // 1. assigns data points to nearest cluster center
      {
	for(unsigned int i=0;i<data.size();i++){
	  unsigned int index = 0;
	  auto minError = distance(data[i], kmeans[index]);

	  for(unsigned int k=1;k<K;k++){
	    auto error = distance(data[i], kmeans[k]);
	    
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

      // printf("changes: %d\n", changes);
      loop++;
    }
    while(changes > 0 && loop < 10);
    
    return true;
  }

  
  template <typename T>
  T EnsembleMeans<T>::distance(const math::vertex<T>& a, const math::vertex<T>& b) const
  {
    T angle = (a*b)[0]/(a.norm()*b.norm());
    // T dist  = math::abs(a.norm() - b.norm());
    // T dist2 = (a - b).norm();

    return angle; // angle between vectors
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

    if(K == 1){ // only one cluster
      for(unsigned int n=0;n<data.size();n++)
	cluster[n] = 0;
      return true;
    }

    for(unsigned int n=0;n<data.size();n++){
      unsigned int index = 0;
      auto minError = distance(data[n], kmeans[index]);
      
      for(unsigned int k=1;k<kmeans.size();k++){
	auto error = distance(data[n], kmeans[k]);

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
  unsigned int EnsembleMeans<T>::getCluster(const math::vertex<T>& d) const
  {
    if(kmeans.size() <= 0) return 0; // error condition!! (silent failure)
    if(kmeans.size() == 1) return 0; // there is only a single cluster (0th cluster)

    unsigned int index = 0;
    auto minError = distance(d, kmeans[index]);
      
    for(unsigned int k=1;k<kmeans.size();k++){
      auto error = distance(d, kmeans[k]);

      if(error < minError){
	index = k;
	minError = error;
      }
    }
      
    return index;
  }
  

  template <typename T>
  int EnsembleMeans<T>::getMajorityCluster(math::vertex<T>& mean, T& p) const
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
