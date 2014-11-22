/*
 * simple single neuron with many inputs and weights
 */

#include <list>
#include <vector>

#include "activation_function.h"
#include "odd_sigmoid.h"


#ifndef neuron_h
#define neuron_h

namespace whiteice
{

  template <typename T>
    class neuron
    {
    public:
      
      neuron(unsigned int input_size = 1);
      neuron(const activation_function<T>& F, unsigned int input_size = 1);
      neuron(const neuron<T>& n);
      ~neuron();
      
      bool set_activation(const activation_function<T>& F);
      activation_function<T>* get_actication(); // returns pointer (not copy) to neuron's activation function
      
      T operator() (const T& field);
      
      T operator() (const std::list<T>& input) const; // ADD EXCEPTIONS FOR WRONG SIZE INPUTS!
      T operator() (const std::vector<T>& input) const; // ADD EXCEPTIONS FOR WRONG SIZE INPUTS!
      
      /* returns output of neuron */
      T activation_field() const;
      
      /* calculates activation_function.derivate(local_field_value) */
      T activation_derivate_field() const;
      
      bool randomize();
      
      
      T& local_field();
      T& bias();
      
      // returns delta of weights (used by backpropagation)
      T& delta(unsigned int index);
      
      /* returns nth weight */
      T& operator[](unsigned int index);
      
      unsigned int input_size();
      
    private:
      
      unsigned int input_size_value;
      T bias_value;
      T* weights;
      T* delta_weights;
      
      T moment;
      
      activation_function<T>* F;
      
      mutable T local_field_value;
      
    };
  
  
  extern template class neuron< float >;
  extern template class neuron< double >;
  extern template class neuron< math::blas_real<float> >;
  extern template class neuron< math::blas_real<double> >;
  
}
  

#endif
