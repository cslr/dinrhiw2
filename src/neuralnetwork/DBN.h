/*
 * Constructs deep belief network using stacked binary RBMs.
 * GBRBM x BBRBM x BBRBM x BBRBM x BBRBM x ..
 * 
 */

#ifndef DBN_h
#define DBN_h

#include <vector>
#include "GBRBM.h"
#include "BBRBM.h"
#include "vertex.h"
#include "dataset.h"
#include "lreg_nnetwork.h"

namespace whiteice
{
  template <typename T = math::blas_real<float> >
    class DBN
    {
    public:
    
    DBN();
    
    // constructs stacked RBM network with the given architecture
    DBN(std::vector<unsigned int>& arch);
    
    DBN(const DBN<T>& dbn);
    
    bool resize(std::vector<unsigned int>& arch);

    unsigned int getNumberOfLayers() const;

    // returns supervised neural network with extra output layer
    // bool convertToNNetwork(whiteice::nnetwork<T>& net,
    //                        const whiteice::dataset<T>& data);
    
    ////////////////////////////////////////////////////////////
    
    // visible neurons/layer of the first RBM
    math::vertex<T> getVisible() const;
    bool setVisible(const math::vertex<T>& v);
    
    // hidden neurons/layer of the last RBM
    math::vertex<T> getHidden() const;
    bool setHidden(const math::vertex<T>& h);
    
    // number of iterations to simulate the system 
    bool reconstructData(unsigned int iters = 2);

    bool reconstructData(std::vector< math::vertex<T> >& samples);
    
    bool initializeWeights(); // initialize weights to small values
    
    // learns stacked RBM layer by layer, each RBM is trained one by one
    // until deltaW < dW or convergence and then algorithm moves to the next layer
    bool learnWeights(const std::vector< math::vertex<T> >& samples,
		      const T& dW, bool verbose);

    // converts DBN to supervised nnetwork by using training samples
    // (cluster 0 = input) and (cluster 1 = output) and
    // by adding linear outputlayer which is optimized locally using linear optimization
    // returned nnetwork contains layer by layer optimized values which
    // can be further optimized across all layers using nnetwork optimizers
    //
    // net - allocates new nnetwork and overwrites pointer to it as a return value
    // 
    bool convertToNNetwork(const whiteice::dataset<T>& data, whiteice::lreg_nnetwork<T>*& net);
    
    private:
    
    // stacked RBMs from the first to the last one

    whiteice::GBRBM<T> input; // input layer
    
    std::vector< whiteice::BBRBM<T> > layers; // hidden layers
    
    };
  

  extern template class DBN< float >;
  extern template class DBN< double >;  
  extern template class DBN< math::blas_real<float> >;
  extern template class DBN< math::blas_real<double> >;

};


#endif

