/*
 * toy: learns data using "ultra-deep" neural networks
 *
 * iteratively adds extra layers/transformations 
 * that improve results from the previous layer
 * 
 * AFAIK (remember) this aims to find non-linear 
 * ICA like solutions from data but is "nonsense" / quick test to try some crazy ideas..
 */

#ifndef __ultradeep_neuralnetwork__
#define __ultradeep_neuralnetwork__

#include "vertex.h"
#include "matrix.h"

#include <vector>
#include <map>

namespace whiteice
{


  class UltraDeep
  {
  public:
    
    UltraDeep();
    
    // tries to find better solutions
    float calculate(const std::vector< math::vertex<> >& input,
		    const std::vector< math::vertex<> >& output);
    

    bool calculate_linear_fit(const std::vector< math::vertex<> > input,
			      const std::vector< math::vertex<> > output,
			      math::matrix<>& A,
			      math::vertex<>& b);
    
    // selects next try using genetic algorithm
    std::multimap< math::blas_real<float>, math::vertex<> >::iterator 
      geneticAlgorithmSelect(math::vertex<>& b, math::vertex<>& d);
    
    bool calculatePCA(const std::vector< math::vertex<> >& data,
		      std::vector< math::vertex<> >& pca_data,
		      const unsigned int DIMENSIONS);
    
    
    void processNNStep(std::vector< math::vertex<> >& data,
		       const math::vertex<>& b, const math::vertex<>& d);
    
    math::blas_real<float> calculateModelError(const std::vector< math::vertex<> >& data,
					       const std::vector< math::vertex<> >& output);
    
    math::vertex<> calculateKurtosis(const std::vector< math::vertex<> >& data);
      
      
  private:
    
    struct ultradeep_parameters
    {
      math::vertex<> d;
      math::vertex<> b;
    };
    
    std::vector<ultradeep_parameters> params;
    std::multimap< math::blas_real<float>, math::vertex<> > goodness;
    
  };
  
  
};


#endif
