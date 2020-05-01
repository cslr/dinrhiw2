
#ifndef neuronlayer_cpp
#define neuronlayer_cpp

#include <iostream>
#include <iterator>
#include <typeinfo>
#include <stdexcept>
#include "neuronlayer.h"
#include "odd_sigmoid.h"
#include "global.h"



namespace whiteice
{
  
  /*
   * creates neuron layer with isize inputs and osize outputs
   * with nsize neurons. input and output data are set to be null.
   */
  template <typename T>
  neuronlayer<T>::neuronlayer(const unsigned int isize,
			      const unsigned int nsize)
  {
    this->F = new odd_sigmoid<T>;
    
    input_data = 0;
    output_data = 0;
    
    learning_factor = 0.01;
    moment_factor = 0.5;
    
    W.resize(nsize, isize);
    b.resize(nsize);
    state.resize(nsize);
    
    W.zero();
    b.zero();
    state.zero();
    
    save_input = true;
  }
  
  
  
  /*
   * creates neuron layer from input and output data
   * pointers can't be 0
   */
  template <typename T> // ADD PROPER EXCEPTION THROWING FOR BAD DATA!!
  neuronlayer<T>::neuronlayer(math::vertex<T> *input, math::vertex<T> *output,
			      const activation_function<T>& F)
  {
    if(input == 0 || output == 0)
      throw std::invalid_argument("neuronlayer ctor - null pointers");
    
    input_data = input;
    output_data = output;
    
    this->F = dynamic_cast<activation_function<T>*>( F.clone() );
    
    learning_factor = 0.01;
    moment_factor = 0.5;
    
    b.resize(output_data->size());
    W.resize(output_data->size(), input_data->size());
    state.resize(output_data->size());
    W.zero();
    b.zero();
    state.zero();
    
    
    save_input = true;
  }
  
  
  
  template <typename T>
  neuronlayer<T>::neuronlayer(const neuronlayer<T>& nl)
  {
    input_data = nl.input_data;
    output_data = nl.output_data;
    
    this->learning_factor = nl.learning_factor;
    this->moment_factor = nl.moment_factor;
    
    this->F = dynamic_cast<activation_function<T>*>( nl.F->clone() );
        
    b = nl.b;
    W = nl.W;    
    state = nl.state;
    
    save_input = nl.save_input;
  }
  
  
  
  /*
   * destroyes neurons, input and output are not free'ed
   */
  template <typename T>
  neuronlayer<T>::~neuronlayer(){ delete F; }
  
  
  template <typename T>
  bool neuronlayer<T>::randomize()
  {
    /*
     * target variance is 1/input_size_value (so field var is 1).
     * 
     * equally distributed variable with mean = 0, [-a,a]
     * variance:
     * Int(-a,a; x^2 * 1/2a) = 1/2a* Inted(-a,a; 1/3 x^3)
     * = 1/2a * 1/3 * 2*a^3 = a^2 * 1/3
     * => 3*var = a^2 <= a = sqrt(3*var) | var = 1.0/input_size
     *
     */
    
    double a = whiteice::math::sqrt(3.0/((double)W.xsize()));
    
    for(unsigned int j=0;j<W.ysize();j++)
      for(unsigned int i=0;i<W.xsize();i++)
	W(j,i) = T((((float)rand())/((float)RAND_MAX))*2.0f*a - a); // [-a, +a]
    
    
    for(unsigned int i=0;i<b.size();i++)
      b[i] = T((((float)rand())/((float)RAND_MAX)) - 0.5f); // bias: [-0.1, +0.1]
    
    return true;
  }
  
  
  template <typename T>
  T& neuronlayer<T>::moment() 
  {
    return moment_factor;
  }
  
  template <typename T>
  const T& neuronlayer<T>::moment() const 
  {
    return moment_factor;
  }

  template <typename T>
  T& neuronlayer<T>::learning_rate() 
  {
    return learning_factor;
  }
  
  template <typename T>
  const T& neuronlayer<T>::learning_rate() const 
  {
    return learning_factor;
  }
  
  
  template <typename T>
  math::vertex<T>& neuronlayer<T>::bias() { return b; }
  
  
  template <typename T>
  const math::vertex<T>& neuronlayer<T>::bias() const { return b; }
  
  
  template <typename T>
  math::matrix<T>& neuronlayer<T>::weights() { return W; }
  
  
  template <typename T>
  const math::matrix<T>& neuronlayer<T>::weights() const { return W; }
  
  
  template <typename T>
  unsigned int neuronlayer<T>::input_size()
  {
    return W.xsize();
  }
  
  
  template <typename T>
  unsigned int neuronlayer<T>::size() const 
  {
    return state.size();
  }
  
  
  template <typename T>
  math::vertex<T>*& neuronlayer<T>::input() 
  {
    return input_data;
  }
  

  template <typename T>
  math::vertex<T>*& neuronlayer<T>::output() 
  {
    return output_data;
  }

  
  
  template <typename T>
  bool neuronlayer<T>::set_activation(const activation_function<T>& F)
  {
    try{
      this->F = dynamic_cast<activation_function<T>*>(F.clone());
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  activation_function<T>* neuronlayer<T>::get_activation()
  {
    return F;
  }
  
  
  template <typename T>
  const activation_function<T>* neuronlayer<T>::get_activation() const
  {
    return F;
  }
  
  
  
  /*
   * calculates ouput value of input
   */
  template <typename T>
  bool neuronlayer<T>::calculate() 
  {
    // (todo: when code has stabilized move these checks to assert() and
    //  compile usually with NDEBUG defined)
    if(!input_data || !output_data) return false;
    if(W.xsize() != input_data->size()) return false;
    if(W.ysize() != output_data->size()) return false;
    
    // calculates neuronlayer linear part
    // (local field)
    
    // std::cout << "IN: " << *input_data << std::endl;
    
    if(save_input)
      saved_input = (*input_data);
    
    // (optimize with cblas_Xgemv, input_data may have
    //  more dimensions (not in use) than W have columns
    //  but rest is ignored)
    
    state = b;
    
    if(typeid(T) == typeid(math::blas_real<float>)){
      
      cblas_sgemv(CblasRowMajor, CblasNoTrans,
  		  W.ysize(), W.xsize(),             
  		  1.0f, (float*)(W.data), W.xsize(),
		  (float*)(input_data->data), 1,
  		  1.0f, (float*)state.data, 1);
    }
    else if(typeid(T) == typeid(math::blas_complex<float>)){
      math::blas_complex<float> a; a = 1.0f;
      
      
      cblas_cgemv(CblasRowMajor, CblasNoTrans,
  		  W.ysize(), W.xsize(),
  		  (float*)(&a), (float*)(W.data), W.xsize(), 
		  (float*)(input_data->data), 1,
  		  (float*)(&a), (float*)state.data, 1);
    }
    else if(typeid(T) == typeid(math::blas_real<double>)){
      
      cblas_dgemv(CblasRowMajor, CblasNoTrans,
  		  W.ysize(), W.xsize(),
  		  1.0, (double*)(W.data), W.size(), 
		  (double*)(input_data->data), 1,
  		  1.0, (double*)state.data, 1);
    }
    else if(typeid(T) == typeid(math::blas_complex<double>)){
      math::blas_complex<double> a; a = 1.0;
      
      cblas_zgemv(CblasRowMajor, CblasNoTrans,
  		  W.ysize(), W.xsize(),
  		  (double*)(&a), (double*)(W.data), W.xsize(), 
		  (double*)(input_data->data), 1,
  		  (double*)(&a), (double*)state.data, 1);
    }
    else{ // generic matrix * vertex code
      
      unsigned int k = 0;
      for(unsigned int j=0;j<state.size();j++){
  	for(unsigned int i=0;i<input_data->size();i++,k++)
  	  state[j] += W.data[k]*(input_data->data[i]);
      }
      
    }
    
    // calculates response of each output
    
    for(unsigned int i=0;i<state.size();i++)
      (*output_data)[i] = F->calculate(state[i]);
    
    return true;
  }
  
  
  /*
   * calculates() outputs from inputs
   */
  template <typename T>
  bool neuronlayer<T>::operator()() 
  {
    return calculate();
  }
  
  
  ////////////////////////////////////////////////////////////
  // compression / decompression
  // only compresses weight matrix because
  // (biases are small and won't compress very well .. I think)
  
  template <typename T>
  bool neuronlayer<T>::compress() 
  {
    return W.compress();
  }
  
  
  template <typename T>
  bool neuronlayer<T>::decompress() 
  {
    return W.decompress();
  }
  
  
  template <typename T>
  bool neuronlayer<T>::iscompressed() const 
  {
    return W.iscompressed();
  }
  
  
  template <typename T>  
  float neuronlayer<T>::ratio() const 
  {
    return W.ratio();
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class neuronlayer<float>;
  template class neuronlayer<double>;
  template class neuronlayer< math::blas_real<float> >;
  template class neuronlayer< math::blas_real<double> >;

}
  
#endif







