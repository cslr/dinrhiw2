/*
 * neural network implementation (V2)
 * 
 * Use rectifier non-linearity in all other layers except output
 * layer which should be linear.
 * 
 * UPDATE:
 * dropout probability is now 10% instead of 20% this should improve performance
 *
 * Optimize code to use cblas
 * 
 */

#ifndef nnetwork_h
#define nnetwork_h


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
  
  template < typename T = math::blas_real<float> >
  class nnetwork
  {
  public:
    
    enum nonLinearity
      {
       sigmoid = 0, // uses sigmoid non-linearity as the default (0) [output: [0,+1] input (-inf,inf)]
       stochasticSigmoid = 1, // clipped to 0/1 values.. (1)
       halfLinear = 2, // for deep networks (2) [f(x)=tanh(x) + 0.5x)
       pureLinear = 3, // for last-layer and comparing nnetworks (linear f(x)=x) (3)
       tanh = 4, // tanh non-linearity (output: [-1,+1] (input: [-1,1])
       rectifier = 5, // leaky ReLU f(x) = max(0.1x,x) - deep networks [biologically motivated]
       softmax = 6 // for complex neural networks: this derivates ok and is similar to ReLU rectifier
      };
    
    
    
    // creates useless 1x1 network. 
    // Use to load some useful network
    nnetwork(); 
    nnetwork(const nnetwork<T>& nn);
    nnetwork(const std::vector<unsigned int>& nnarch,
	     const nonLinearity nl = rectifier) ;
    
    
    virtual ~nnetwork();
    
    nnetwork<T>& operator=(const nnetwork<T>& nn);
    
    // prints nnetwork information (mostly for debugging purposes)
    void printInfo() const; 

    // logs (whiteice::logging Log.h)
    // nnetwork parameters statistics per layer
    // (used to notice the largest element in nnetwork)
    void diagnosticsInfo() const;

    ////////////////////////////////////////////////////////////
    
    math::vertex<T>& input() { return inputValues; }
    math::vertex<T>& output() { return outputValues; }
    const math::vertex<T>& input() const { return inputValues; }
    const math::vertex<T>& output() const { return outputValues; }
    
    // returns input and output dimensions of neural network
    unsigned int input_size() const ;
    unsigned int output_size() const ;
    unsigned int gradient_size() const ;

    void getArchitecture(std::vector<unsigned int>& arch) const;

    // invalidates all data and essentially creates a new network over previous one
    bool setArchitecture(const std::vector<unsigned int>& arch,
			 const nonLinearity nl = rectifier); // all layers except the last one (linear) has this non-linearity
    
    
    bool calculate(bool gradInfo = false, bool collectSamples = false);
    bool operator()(bool gradInfo = false, bool collectSamples = false){
      return calculate(gradInfo, collectSamples);
    }
    
    // simple thread-safe version [parallelizable version of calculate: don't calculate gradient nor collect samples]
    bool calculate(const math::vertex<T>& input, math::vertex<T>& output) const;

    // thread safe calculate call which also stores backpropagation data
    // bpdata can be used calculate mse_gradient() with backpropagation
    // in a const nnetwork<> class so that same nnetwork<> object can
    // be used with multiple threads. If dropout vector has data also
    // does dropout heuristics. This allows same nnetwork<> object to be
    // used in thread safe manner.
    bool calculate(const math::vertex<T>& input, math::vertex<T>& output,
		   const std::vector< std::vector<bool> >& dropout,
		   std::vector< math::vertex<T> >& bpdata) const;

    unsigned int length() const; // number of layers

    // set nnetworks parameters to random values
    // type = 0: random [-1,+1] values
    // type 1 = smart initialization (uniform distribution)
    // type 2 more stable initialization (normal distribution)
    bool randomize(const unsigned int type = 2,
		   const bool smallvalues = false);

      // set parameters to fit the data from dataset (we set weights to match data values) [experimental code]
    bool presetWeightsFromData(const whiteice::dataset<T>& ds);
    
    // set weights randomly using smart heuristic except last layer which is linearly optimized to fit the data
    bool presetWeightsFromDataRandom(const whiteice::dataset<T>& ds);

    // calculates gradient of parameter weights w f(v|w) when using squared error: 
    // grad(0,5*error^2) = grad(output - right) = nn(x) - y
    // used backpropagation data stored within nnetwork<> by non const calculate() call.
    bool mse_gradient(const math::vertex<T>& error, math::vertex<T>& grad) const;
    
    // calculates gradient of parameter weights w f(v|w) when using squared error: 
    // grad(0,5*error^2) = grad(output - right) = nn(x) - y
    // used backpropagation bpdata provided by caller (use calculate() with bpdata) and 
    // dropout heuristic if dropout vector is non empty object.
    bool mse_gradient(const math::vertex<T>& error,
		      const std::vector< math::vertex<T> >& bpdata,
		      const std::vector< std::vector<bool> >& dropout,
		      math::vertex<T>& grad) const;

    // calculates jacobian/gradient of parameter weights w f(v|w)
    bool jacobian(const math::vertex<T>& input, math::matrix<T>& grad) const;

    // calculates gradient of input v, grad f(v) while keeping weights w constant
    bool gradient_value(const math::vertex<T>& input, math::matrix<T>& grad) const;

     ////////////////////////////////////////////////////////////
    
    // load & saves neuralnetwork data from file
    bool load(const std::string& filename) ;
    bool save(const std::string& filename) const ;

    ////////////////////////////////////////////////////////////
    
    // exports and imports neural network parameters to/from vertex
    bool exportdata(math::vertex<T>& v) const ;
    bool importdata(const math::vertex<T>& v) ;
    
    // number of dimensions used by import/export
    unsigned int exportdatasize() const ;

    ////////////////////////////////////////////////////////////
    
    unsigned int getLayers() const ;
    unsigned int getInputs(unsigned int l) const ; // number of inputs per neuron for this layer
    unsigned int getNeurons(unsigned int l) const ; // number of neurons (outputs) per layer

    // gets and sets network parameters: Weight matrixes and biases
    bool getBias(math::vertex<T>& b, unsigned int layer) const ;
    bool setBias(const math::vertex<T>& b, unsigned int layer) ;
    bool getWeights(math::matrix<T>& w, unsigned int layer) const ;
    bool setWeights(const math::matrix<T>& w, unsigned int layer) ;

    // whole network settings (except that the last layer is set to linear)
    bool setNonlinearity(nonLinearity nl);
    
    nonLinearity getNonlinearity(unsigned int layer) const ; 
    bool setNonlinearity(unsigned int layer, nonLinearity nl);

    void getNonlinearity(std::vector<nonLinearity>& nls) const ;
    bool setNonlinearity(const std::vector<nonLinearity>& nls) ;
    
    bool setFrozen(unsigned int layer, bool frozen = true); // nnetwork layers can be set to be "frozen"
    bool setFrozen(const std::vector<bool>& frozen);        // so that gradient for those parameters is 
    bool getFrozen(unsigned int layer) const;               // always zero and optimization is restricted
    void getFrozen(std::vector<bool>& frozen) const;        // to other parts of the network

    // creates subnet starting from fromLayer:th layer to the output
    nnetwork<T>* createSubnet(const unsigned int fromLayer);

    // creates subnet starting from fromLayer to toLayer;
    nnetwork<T>* createSubnet(const unsigned int fromLayer,
			      const unsigned int toLayer);

    // injects (if possible) subnet into net starting from fromLayer:th layer
    bool injectSubnet(const unsigned int fromLayer, nnetwork<T>* nn); 
    
    unsigned int getSamplesCollected() const ;

    /* gets MAXSAMPLES samples or getSamplesCollected() number of samples whichever
     * if smaller. if MAXSAMPLES = 0 gets getSamplesCollected() number of samples */
    bool getSamples(std::vector< math::vertex<T> >& samples,
		    unsigned int layer, const unsigned int MAXSAMPLES=0) const ;
    void clearSamples() ;

    // drop out support:

    // set neurons to be non-dropout neurons with probability p [1-p are dropout neurons]
    bool setDropOut(const T retain_p = T(0.90)) ;

    // set dropout tables neurons to be non-dropout neurons
    // with probability p [1-p are dropout neurons]
    bool setDropOut(std::vector< std::vector<bool> >& dropout,
		    const T retain_p = T(0.90)) const;

    // clears drop out but scales weights according to retain_probability
    bool removeDropOut(T retain_p = T(0.90)) ;
    
    void clearDropOut() ; // remove all drop-out without changing weights
    
    ////////////////////////////////////////////////////////////
  public:

    // with dropout heuristic
    inline T nonlin(const T& input, unsigned int layer, unsigned int neuron) const ; // non-linearity used in neural network
    inline T Dnonlin(const T& input, unsigned int layer, unsigned int neuron) const ; // derivate of non-linearity used in neural network

    // without dropout heuristic
    inline T nonlin(const T& input, unsigned int layer) const;
    inline T Dnonlin(const T& input, unsigned int layer) const;

    
    T inv_nonlin(const T& input, unsigned int layer, unsigned int neuron) const ; // inverse of non-linearity used [not really used]
    
  private:

      
    inline void gemv_gvadd(unsigned int yd, unsigned int xd, 
			   const T* W, T* x, T* y,
			   unsigned int dim, T* s, const T* b,
			   T* temp) const;
    
    
    // data structures which are part of interface [avoid using these!]
    math::vertex<T> inputValues;
    math::vertex<T> outputValues;
    
    
    bool hasValidBPData;
    
    std::vector<nonLinearity> nonlinearity; // which non-linearity to use in each layer
    
    // frozen layers (that are not optimized or set to some values otherwise)
    std::vector<bool> frozen;  
    
    // stochastic retain probability during activation [feedforward]
    T retain_probability;
    
    // drop-out configuration for each layer
    // [if neuron is dropout neuron its non-linearity is zero]
    std::vector< std::vector<bool> > dropout;  // used by gradient calculation (backward step)
    
    whiteice::RNG<T> rng;
    
    // architecture (eg. 3-2-6) info
    std::vector<unsigned int> arch;
    unsigned int maxwidth;    
    unsigned int size;
    
    // USE vertex<T> instead of vector<T> because vertex is allocated
    // by cuBLAS if needed

    // parameters of the network [no support for convolutional layers yet]
    // vectorized form of parameters is (gradient format, row major matrixes):
    // vec(nnetwork) = [vec(W1) vec(b1) .. vec(Wi) vec(bi) .. vec(Wn) vec(bn)]
    std::vector< math::matrix<T> > W;
    std::vector< math::vertex<T> > b;

    // backpropagation data
    std::vector< math::vertex<T> > bpdata; // array of input and local field values
    
    // used to collect samples about data passing through the network,
    // this will then be used later to do unsupervised regularization of training data
    std::vector< std::vector< math::vertex<T> > > samples;
    
    // bool compressed;
    // MemoryCompressor* compressor;
  };
  
  
  
  //extern template class nnetwork< float >;
  //extern template class nnetwork< double >;
  
  extern template class nnetwork< math::blas_real<float> >;
  extern template class nnetwork< math::blas_real<double> >;

  extern template class nnetwork< math::blas_complex<float> >;
  extern template class nnetwork< math::blas_complex<double> >;
  
};


#endif

