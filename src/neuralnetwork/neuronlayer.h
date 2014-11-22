/*
 * 1-dimensional
 * neuronlayer with isize inputs and nsize neurons so the
 * output is nsize vector
 */

#ifndef neuronlayer_h
#define neuronlayer_h

#include <vector>
#include <stdexcept>
#include <exception>
#include "matrix.h"
#include "dinrhiw_blas.h"
#include "activation_function.h"
#include "compressable.h"



namespace whiteice
{
  
  template <typename T> class backpropagation;
  
  namespace math { template <typename T> class vertex; };
  
  
  
  template <typename T=math::blas_real<float> >
    class neuronlayer : public compressable
  {
    public:
    
    neuronlayer(const unsigned int isize,
		const unsigned int nsize=1);
    neuronlayer(math::vertex<T> *input, math::vertex<T> *output,
		const activation_function<T>& F);
    neuronlayer(const neuronlayer<T>& nl);
    
    virtual ~neuronlayer();
    
    unsigned int input_size();
    unsigned int size() const throw(); // (output size)
    
    math::vertex<T>*& input() throw();
    math::vertex<T>*& output() throw();
    
    bool calculate() throw();
    bool operator()() throw(); // one-step NN activation
    
    bool randomize();
    
    T& moment() throw();
    const T& moment() const throw();    
    T& learning_rate() throw();
    const T& learning_rate() const throw();
    

    math::vertex<T>& bias() throw();
    const math::vertex<T>& bias() const throw();
    
    math::matrix<T>& weights() throw();
    const math::matrix<T>& weights() const throw();
    
    // clones activation function
    bool set_activation(const activation_function<T>& F);
    
    // returns pointer to its activation function
    // caller must not free it
    // (todo: dynamic access protection to ctor/dtor)
    activation_function<T>* get_activation();  
    const activation_function<T>* get_activation() const;
    
    
    bool compress() throw();
    bool decompress() throw();
    bool iscompressed() const throw();
    float ratio() const throw(); // compression ratio
    
    
    private:
    
    friend class backpropagation<T>;
    
    
    math::vertex<T>* input_data;
    
    math::vertex<T> saved_input; // copies input to here if save_input == true.
    
    math::matrix<T> W;
    math::vertex<T> b;
    math::vertex<T> state; // local neuron field values
    
    activation_function<T>* F;
    
    math::vertex<T>* output_data;     	// number of outputs = number of neurons
    
    T moment_factor, learning_factor;  	// backpropagation parameters 
                                        // - TODO: make error correction
    
    bool save_input;
  };
  
  
  
  extern template class neuronlayer<float>;
  extern template class neuronlayer<double>;
  extern template class neuronlayer< math::blas_real<float> >;
  extern template class neuronlayer< math::blas_real<double> >;
    
}

#endif


#include "vertex.h"
#include "backpropagation.h"

