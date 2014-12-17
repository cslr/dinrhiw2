
#include "fastpca.h"
#include "correlation.h"


namespace whiteice
{
  namespace math
  {

    /*
     * Extracts first "dimensions" PCA vectors from data
     * PCA = X^t when Cxx = E{(x-m)(x-m)^t} = X*L*X^t
     */
    template <typename T>
    bool fastpca(const std::vector< vertex<T> >& data, 
		 const unsigned int dimensions,
		 math::matrix<T>& PCA)
    {
      if(data.size() == 0) return false;
      if(data[0].size() < dimensions) return false;
      if(dimensions == 0) return false;
      
      math::vertex<T> m;
      math::matrix<T> Cxx;
      
      if(mean_covariance_estimate(m, Cxx, data) == false)
	return false;
      
      std::vector< math::vertex<T> > pca; // pca vectors
      
      while(pca.size() < dimensions){
	math::vertex<T> gprev;
	math::vertex<T> g;
	g.resize(m.size());
	gprev.resize(m.size());
	
	for(unsigned int i=0;i<g.size();i++){
	  gprev[i] = T(2.0f*(float)rand()/RAND_MAX - 1.0f); // [-1,1]
	  g[i] = T(2.0f*(float)rand()/RAND_MAX - 1.0f); // [-1,1]
	}
	
	T convergence = T(1.0);
	T epsilon = T(10e-6);
	
	do{
	  g = Cxx*g;
	  
	  // orthonormalizes g
	  {
	    auto t = g;
	    
	    for(unsigned int i=0;i<pca.size();i++)
	      g -= (t*pca[i])*pca[i];
	    
	    g.normalize();
	  }
	  
	  convergence = whiteice::math::abs(T(1.0f) - (g*gprev)[0]);
	  
	  gprev = g;
	}
	while(convergence > epsilon);
	
	pca.push_back(g);
      }
      
      PCA.resize(pca.size(), data[0].size());
      
      for(unsigned int j=0;j<pca.size();j++)
	PCA.rowcopyfrom(pca[j], j);
      
      return true;
    }
    

    template bool fastpca<float>(const std::vector< vertex<float> >& data, 
				 const unsigned int dimensions,
				 math::matrix<float>& PCA);
    template bool fastpca<double>(const std::vector< vertex<double> >& data, 
				  const unsigned int dimensions,
				  math::matrix<double>& PCA);
    template bool fastpca< blas_real<float> >(const std::vector< vertex< blas_real<float> > >& data, 
					      const unsigned int dimensions,
					      math::matrix< blas_real<float> >& PCA);
    template bool fastpca< blas_real<double> >(const std::vector< vertex< blas_real<double> > >& data, 
					       const unsigned int dimensions,
					       math::matrix< blas_real<double> >& PCA);
    
    
  };
  
};
