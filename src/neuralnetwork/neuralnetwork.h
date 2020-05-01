/*
 * simplistic/restricted implementation of
 * feedforward neural network
 *
 * NOTE: user must also implement 'bool whiteice::convert(float&, const T&) '
 * conversion functions from T -> to float in order to make save() and load() to work.
 * (see SRC/math/blas_primitives.h for example / used by blas_read<float> )
 *
 * also T(float) should be able to convert floating point number to type T number.
 * (read save() and load() for details)
 */

#ifndef neuralnetwork_h
#define neuralnetwork_h

#include <vector>
#include <stdexcept>
#include <exception>

#include "compressable.h"
#include "dinrhiw_blas.h"


namespace whiteice
{
  template <typename T> class neuronlayer;
  template <typename T> class backpropagation;
  
  namespace math{ template <typename T> class vertex; }
  

  
  template <typename T=math::blas_real<float> >
    class neuralnetwork : public compressable
  {
    public:
    
    neuralnetwork();
    neuralnetwork(const neuralnetwork<T>& nn);
    neuralnetwork(unsigned int layers, unsigned int width); // L*W rectangle network
    
    // (W1, W2, ... Wn, width) network
    neuralnetwork(const std::vector<unsigned int>& nn_structure,
		  bool compressed_network = false) ;
    
    virtual ~neuralnetwork();
    
    neuralnetwork<T>& operator=(const neuralnetwork<T>& nn);
  
    // other stuff to create more complex nn structures
    /* accessing data / configuration */
    
    math::vertex<T>& input() ;
    math::vertex<T>& output() ;
    
    /* calculates output for input */
    bool calculate();
    bool operator()();
    
    neuronlayer<T>& operator[](unsigned int index) ;
    unsigned int length() const; // number of layers
    
    bool randomize();
    
    // sets learning rate globally
    bool setlearningrate(float rate);
    
    // load & saves neuralnetwork data from file
    // 
    // loading and saving isn't possible when network is compressed
    // (todo: add support for compressed networks)
    // 
    bool load(const std::string& filename) ;
    bool save(const std::string& filename) const ;
    
    // exports and imports neural network parameters to/from vertex
    // (doesn't change architecture of neural network)
    // export() and import() are mainly used by PSO and GA optimizers
    bool exportdata(math::vertex<T>& v) const ;
    bool importdata(const math::vertex<T>& v) ;
    
    // number of dimensions used by import/export
    unsigned int exportdatasize() const ; 
    
    
    // changes NN to compressed form of operation or
    // back to normal non-compressed form
    
    bool compress() ;
    bool decompress() ;
    bool iscompressed() const ;
    
    // returns compression ratio: compressed/orig
    float ratio() const ;
    
    friend class backpropagation<T>;
    
    private:
    
    /* final inputs and outputs */
    math::vertex<T> input_values;
    math::vertex<T> output_values;
    
    // scaling for input and output layers
  
    std::vector< neuronlayer<T>* > layers;
    
    // state is number of values which
    // is both input and output of each neuronlayer
    math::vertex<T> state;
    
    
    bool compressed;
  };
};


#include "backpropagation.h"
#include "neuronlayer.h"
#include "vertex.h"


namespace whiteice
{
  extern template class neuralnetwork< float >;
  extern template class neuralnetwork< double >;  
  extern template class neuralnetwork< math::blas_real<float> >;
  extern template class neuralnetwork< math::blas_real<double> >;
  
};


#endif

