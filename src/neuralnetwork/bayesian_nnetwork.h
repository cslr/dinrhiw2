/*
 * bayesian neural network (using samples
 *
 * supports use of samples of weights p(w) which
 * wll be used to store and load network state and 
 * calculate responses.
 */

#ifndef bayesian_nnetwork_h
#define bayesian_nnetwork_h

#include "nnetwork.h"

namespace whiteice
{
  template < typename T = math::blas_real<float> >
    class bayesian_nnetwork
    {
    public:

    bayesian_nnetwork();
    virtual ~bayesian_nnetwork();

    /*
     * imports and exports samples of p(w) to and from nnetwork
     */
    bool importSamples(const std::vector<unsigned int>& arch,
		       const std::vector< math::vertex<T> >& weights);
    bool importNetwork(const nnetwork<T>& net);

    bool exportSamples(std::vector<unsigned int>& arch,
		       std::vector< math::vertex<T> >& weights);
    

    // calculates E[f(input,w)] = E[y|x] and Var[f(x,w)] = Var[y|x] for given input
    bool calculate(const math::vertex<T>& input,
		   math::vertex<T>& mean,
		   math::matrix<T>& covariance);

    unsigned int outputSize() const throw();
    unsigned int inputSize() const throw();

    // stores and loads bayesian nnetwork to a text file
    // (saves all samples into files)
    bool load(const std::string& filename) throw();
    bool save(const std::string& filename) const throw();

    private:

    std::vector< nnetwork<T>* > nnets;
      
      
    };

  extern template class bayesian_nnetwork< float >;
  extern template class bayesian_nnetwork< double >;  
  extern template class bayesian_nnetwork< math::blas_real<float> >;
  extern template class bayesian_nnetwork< math::blas_real<double> >;
  
};



#endif


