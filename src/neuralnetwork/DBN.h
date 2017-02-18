/*
 * Constructs deep belief network using stacked RBMs.
 * GBRBM x BBRBM x BBRBM x BBRBM x BBRBM x ..
 * 
 * Optimizes layers greedly (layer per layer basis)
 */

#ifndef DBN_h
#define DBN_h

#include <vector>
#include "GBRBM.h"
#include "BBRBM.h"
#include "vertex.h"
#include "dataset.h"
#include "nnetwork.h"

namespace whiteice
{
  template <typename T = math::blas_real<float> >
    class DBN
    {
    public:
    
    DBN(bool binary = false);
    
    // constructs stacked RBM network with the given architecture
    DBN(std::vector<unsigned int>& arch, bool binary = false);
    
    DBN(const DBN<T>& dbn);
    
    bool resize(std::vector<unsigned int>& arch);

    unsigned int getInputDimension() const;
    unsigned int getHiddenDimension() const;

    whiteice::GBRBM<T>& getInputGBRBM() throw(std::invalid_argument);
    whiteice::BBRBM<T>& getInputBBRBM() throw(std::invalid_argument);
    const whiteice::GBRBM<T>& getInputGBRBM() const throw(std::invalid_argument);
    const whiteice::BBRBM<T>& getInputBBRBM() const throw(std::invalid_argument);

    // true if GBRBM<T> is used as input otherwise BBRBM<T>
    bool getGaussianInput() const; 

    // layer is [0..getNumberOfLayers-2]
    whiteice::BBRBM<T>& getHiddenLayer(unsigned int layer) throw(std::invalid_argument);
    const whiteice::BBRBM<T>& getHiddenLayer(unsigned int layer) const throw(std::invalid_argument);

    unsigned int getNumberOfLayers() const;

    ////////////////////////////////////////////////////////////
    
    // visible neurons/layer of the first RBM
    math::vertex<T> getVisible() const;
    bool setVisible(const math::vertex<T>& v);
    
    // hidden neurons/layer of the last RBM
    math::vertex<T> getHidden() const;
    bool setHidden(const math::vertex<T>& h);
    
    // number of iterations to simulate the system (v->h, h->v)
    bool reconstructData(unsigned int iters = 2);

    // v->h, h->v
    bool reconstructData(std::vector< math::vertex<T> >& samples);

    // calculates hidden responses (v->h)
    bool calculateHidden(std::vector< math::vertex<T> >& samples);
    
    bool initializeWeights(); // initialize weights to small values
    
    // learns stacked RBM layer by layer, each RBM is trained one by one
    // until deltaW < dW or convergence and then algorithm moves to the next layer
    bool learnWeights(const std::vector< math::vertex<T> >& samples,
		      const T& dW, bool verbose);

    // converts DBN to supervised nnetwork by using training samples
    // (cluster 0 = input) and (cluster 1 = output) and
    // by adding linear outputlayer which is optimized locally
    // using linear optimization
    // returned nnetwork contains layer by layer optimized values which
    // can be further optimized across all layers using nnetwork optimizers
    // 
    // returns supervised mean-field (non-stochastic)
    // neural network with extra output layer
    // (sigmoid non-linearity except the last layer which is pureLinear)
    //
    // net - allocates new nnetwork and overwrites pointer to it as a return value
    //
    bool convertToNNetwork(const whiteice::dataset<T>& data,
			   whiteice::nnetwork<T>*& net);

    // converts DBN to supervised (mean-field) nnetwork
    // allocated using new (caller must delete)
    // returns sigmoid mean-field network
    bool convertToNNetwork(whiteice::nnetwork<T>*& net);

    // converts inverse (from hidden to visible) DBN to nnetwork
    // net is allocated using new (caller must delete)
    // returns sigmoid mean-field network
    bool convertInverseToNNetwork(whiteice::nnetwork<T>*& net);

    // converts trained DBN to autoencoder which can be trained using LBFGS etc
    // returns stochastic neural network
    bool convertToAutoEncoder(whiteice::nnetwork<T>*& net) const;

    bool save(const std::string& basefilename) const;
    bool load(const std::string& basefilename);
    
    private:
    
    // stacked RBMs from the first to the last one

    whiteice::GBRBM<T> gb_input; // input layer
    whiteice::BBRBM<T> bb_input;
    bool binaryInput;
    
    std::vector< whiteice::BBRBM<T> > layers; // hidden layers
    
    };
  

  extern template class DBN< float >;
  extern template class DBN< double >;  
  extern template class DBN< math::blas_real<float> >;
  extern template class DBN< math::blas_real<double> >;

};


#endif

