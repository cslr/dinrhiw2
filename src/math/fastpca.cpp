
#include "fastpca.h"

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
      
      // IMPLEMENT ME
      
      assert(0);
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
