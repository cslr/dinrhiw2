

#ifndef neuron_cpp
#define neuron_cpp

#include <new>
#include <iterator>
#include <vector>
#include <list>
#include <cstdlib>
#include <assert.h>


#include "activation_function.h"
#include "odd_sigmoid.h"
#include "neuron.h"
#include "global.h"
// #include "dlib.h"


using namespace std;


namespace whiteice
{
  
  template <typename T>
  neuron<T>::neuron(unsigned int input_size_value)
  {
    odd_sigmoid<T>* AF = new odd_sigmoid<T>;
    local_field_value = 0;
    this->F = AF;
    
    this->input_size_value = input_size_value;
    weights = new T[input_size_value];
    delta_weights = new T[input_size_value];
    
    bias_value = 0;
    
    // must use random initialization in order to get
    // results (if everything is 0, gradient = 0..
    for(unsigned int i=0;i<input_size_value;i++){
      weights[i] = 0;
      delta_weights[i] = 0;
    }
    
  }
  
  
  template <typename T>
  neuron<T>::neuron(const activation_function<T>& F, unsigned int input_size_value)
  {
    local_field_value = 0;
    this->F = dynamic_cast<activation_function<T>*>(F.clone());
    this->input_size_value = input_size_value;
    
    weights = new T[input_size_value];
    delta_weights = new T[input_size_value];
    bias_value = 0;
    
    for(unsigned int i=0;i<input_size_value;i++){
      weights[i] = 0;
      delta_weights[i] = 0;
    }
  }

  
  template <typename T>
  neuron<T>::neuron(const neuron<T>& n)
  {
    this->local_field_value = n.local_field_value;
    this->F = dynamic_cast<activation_function<T>*>(n.F->clone());
    this->input_size_value = n.input_size_value;
    
    this->weights = new T[input_size_value];
    this->delta_weights = new T[input_size_value];
    this->bias_value = n.bias_value;
    
    for(unsigned int i=0;i<input_size_value;i++){
      this->weights[i] = n.weights[i];
      this->delta_weights[i] = n.delta_weights[i];
    }
  }
  
  
  template <typename T>
  neuron<T>::~neuron()
  {
    delete F;
    delete[] weights;
    delete[] delta_weights;
  }
  
  
  template <typename T>
  bool neuron<T>::set_activation(const activation_function<T>& F)
  {
    try{ this->F = dynamic_cast<activation_function<T>*>(F.clone()); return true; }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  activation_function<T>* neuron<T>::get_actication()
  {
    return F;
  }
  

  /* returns neuron output */
  template <typename T>
  T neuron<T>::activation_field() const
  {
  return (*F)(local_field_value);
  }
  
  
  /* calculates activation_function.derivate(local_field_value) */
  template <typename T>
  T neuron<T>::activation_derivate_field() const
  {
    return F->derivate(local_field_value);
  }
  
  
  template <typename T>
  T neuron<T>::operator() (const T& field)
  {
    local_field_value = field;
    return (*F)(local_field_value);
  }
  
  
  template <typename T>
  T neuron<T>::operator() (const std::list<T>& input) const
  {
    assert(input.size() >= input_size_value);
    
    typename list<T>::const_iterator i = input.begin();
    unsigned int k = 0;
    
    T field = bias_value;
    
    while(k < input_size_value){
      field += weights[k]*(*i);
      k++;
      i++;
    }
    
    local_field_value = field;
    
    return (*F)(local_field_value);
  }
  
  
  template <typename T>
  T neuron<T>::operator() (const std::vector<T>& input) const
  {
    assert(input.size() >= input_size_value);
    
    typename vector<T>::const_iterator i = input.begin();
    unsigned int k = 0;
    
    T field = bias_value;
    
    while(k < input_size_value){
      field += weights[k]*(*i);
      k++;
      i++;
    }
    
    local_field_value = field;
    
    return (*F)(local_field_value);
  }
  
  
  template <typename T>
  bool neuron<T>::randomize()
  {
    local_field_value = 0;
    bias_value = 0;
    
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
    
    double a = sqrt(3.0/((double)input_size_value));
    
    for(unsigned int i=0;i<input_size_value;i++)
      weights[i] = (((float)rand())/((float)RAND_MAX))*2.0f*a - a; // [-a,+a]
    
    return true;
  }
  
  
  template <typename T>
  T& neuron<T>::local_field()
  {
    return local_field_value;
  }
  
  
  template <typename T>
  T& neuron<T>::bias()
  {
  return bias_value;
  }
  
  template <typename T>
  T& neuron<T>::delta(unsigned int index)
  {
    return delta_weights[index];
  }
  
  
  template <typename T>
  T& neuron<T>::operator[](unsigned int index)
  {
    assert(index < input_size_value);
    return weights[index];
  }
  
  
  template <typename T>
  unsigned int neuron<T>::input_size()
  {
    return input_size_value;
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class neuron< float >;
  template class neuron< double >;  
  template class neuron< math::atlas_real<float> >;
  template class neuron< math::atlas_real<double> >;
  
}

#endif


