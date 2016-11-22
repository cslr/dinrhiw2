/*
 * neural network implementation (V2)
 * work arounds some bugs + has more efficient implementation
 * 
 * The neural network uses:
 * - tanh(x) non-linearity y=tanh(Ax+b), 
 *   expect at the output layer where direct 
 *   linear transformation is used (y=Ax+b).
 * - uses direct memory accesses and stores parameters
 *   as a single big vector
 * - could benefit optimization from BLAS routines
 * 
 */

#ifndef nnetwork_h
#define nnetwork_h


#include "dinrhiw_blas.h"
#include "vertex.h"
#include "conffile.h"
#include "compressable.h"
#include "MemoryCompressor.h"
#include <vector>


namespace whiteice
{
  
  template < typename T = math::blas_real<float> >
    class nnetwork
    {
    public:

    enum nonLinearity {
      sigmoidNonLinearity, // uses sigmoid non-linearity as the default
      halfLinear, // for deep networks
      pureLinear  // for testing and comparing nnetworks (pure linear)
    };


    
    // creates useless 1x1 network. 
    // Use to load some useful network
    nnetwork(); 
    nnetwork(const nnetwork& nn);
    nnetwork(const std::vector<unsigned int>& nnarch,
	     const nonLinearity nl = sigmoidNonLinearity) throw(std::invalid_argument);
    
    
    virtual ~nnetwork();

    nnetwork<T>& operator=(const nnetwork<T>& nn);

    ////////////////////////////////////////////////////////////
    
    math::vertex<T>& input() throw(){ return inputValues; }
    math::vertex<T>& output() throw(){ return outputValues; }
    const math::vertex<T>& input() const throw(){ return inputValues; }
    const math::vertex<T>& output() const throw(){ return outputValues; }
    
    // returns input and output dimensions of neural network
    unsigned int input_size() const throw();
    unsigned int output_size() const throw();

    void getArchitecture(std::vector<unsigned int>& arch) const;
    
    bool calculate(bool gradInfo = false, bool collectSamples = false);
    bool operator()(bool gradInfo = false, bool collectSamples = false){
      return calculate(gradInfo, collectSamples);
    }
    
    // simple thread-safe version [parallelizable version of calculate: don't calculate gradient nor collect samples]
    bool calculate(const math::vertex<T>& input, math::vertex<T>& output) const;

    unsigned int length() const; // number of layers
    
    bool randomize();
    
    // calculates gradient grad(0,5*error^2) = grad(right - output)
    bool gradient(const math::vertex<T>& error, math::vertex<T>& grad) const;
    
    bool gradient_value(const math::vertex<T>& input, math::matrix<T>& grad) const;

    bool stochastic(); // return true if nnetwork has stochastic sigmoid activations (clipped to 0 or 1)
    void setStochastic(bool stochastic); // sets stochastic sigmoid activations
    
    ////////////////////////////////////////////////////////////
    
    // load & saves neuralnetwork data from file
    bool load(const std::string& filename) throw();
    bool save(const std::string& filename) const throw();

    ////////////////////////////////////////////////////////////
    
    // exports and imports neural network parameters to/from vertex
    bool exportdata(math::vertex<T>& v) const throw();
    bool importdata(const math::vertex<T>& v) throw();
    
    // number of dimensions used by import/export
    unsigned int exportdatasize() const throw();

    ////////////////////////////////////////////////////////////
    
    unsigned int getLayers() const throw();
    unsigned int getNeurons(unsigned int l) const throw(); // number of neurons per layer

    // gets and sets network parameters: Weight matrixes and biases
    bool getBias(math::vertex<T>& b, unsigned int layer) const throw();
    bool setBias(const math::vertex<T>& b, unsigned int layer) throw();
    bool getWeights(math::matrix<T>& w, unsigned int layer) const throw();
    bool setWeights(const math::matrix<T>& w, unsigned int layer) throw();

    nonLinearity getNonlinearity() const throw();
    bool setNonlinearity(nonLinearity nl);
    
    unsigned int getSamplesCollected() const throw();
    bool getSamples(std::vector< math::vertex<T> >& samples, unsigned int layer) const throw();
    void clearSamples() throw();
    
    ////////////////////////////////////////////////////////////
    public:
    
    T nonlin(const T& input, unsigned int layer, unsigned int neuron) const throw(); // non-linearity used in neural network
    T Dnonlin(const T& input, unsigned int layer, unsigned int neuron) const throw(); // derivate of non-linearity used in neural network
    T inv_nonlin(const T& input, unsigned int layer, unsigned int neuron) const throw(); // inverse of non-linearity used [not really used]
    
    private:
    
    inline void gemv(unsigned int yd, unsigned int xd, T* W, T* x, T* y);
    inline void gvadd(unsigned int dim, T* s, T* b);

    
    inline void gemv_gvadd(unsigned int yd, unsigned int xd, 
			   const T* W, T* x, T* y,
			   unsigned int dim, T* s, const T* b) const;
    
    
    // data structures which are part of
    // interface
    mutable math::vertex<T> inputValues;
    mutable math::vertex<T> outputValues;
    

    bool hasValidBPData;
    bool stochasticActivation;
    nonLinearity nonlinearity; // which non-linearity to use (default: sigmoid)
    
    // architecture (eg. 3-2-6) info
    std::vector<unsigned int> arch;
    unsigned int maxwidth;    
    unsigned int size;

    std::vector<T> data;
    std::vector<T> bpdata;
    
    std::vector<T> state;
    std::vector<T> temp;
    std::vector<T> lgrad;
    
    // used to collect samples about data passing through the network,
    // this will then be used later to do unsupervised regularization of training data
    std::vector< std::vector< math::vertex<T> > > samples;

    // bool compressed;
    // MemoryCompressor* compressor;
  };
  
  
  
  extern template class nnetwork< float >;
  extern template class nnetwork< double >;  
  extern template class nnetwork< math::blas_real<float> >;
  extern template class nnetwork< math::blas_real<double> >;
  
};


#endif

