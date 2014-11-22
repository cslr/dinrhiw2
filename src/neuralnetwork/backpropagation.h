/*
 * implements the basic, now quite outdated
 * optimization method for neural networks
 *
 */
#ifndef backpropagation_h
#define backpropagation_h


#include "dinrhiw_blas.h"
#include "nn_iterative_correction.h"


namespace whiteice
{
  template <typename T> class neuralnetwork;  
  template <typename T> class dataset;
  namespace math{ template <typename T> class vertex; };  
  
  template <typename T>
    class backpropagation : public nn_iterative_correction<T>
    {
    public:
      backpropagation();
      backpropagation(neuralnetwork<T>* nn,
		      const dataset<T>* input,
		      const dataset<T>* output);
      backpropagation(neuralnetwork<T>* nn,
		      const dataset<T>* data);
      
      virtual ~backpropagation();
      
      virtual bool operator()(neuralnetwork<T>& nn,
			      const math::vertex<T>& correct_output) const;
      
      virtual bool calculate (neuralnetwork<T>& nn,
			      const math::vertex<T>& correct_output) const;
      
      // creates copy of object
      virtual nn_iterative_correction<T>* clone() const;
      
      
      bool forced_improve();
      bool improve(unsigned int niters = 50);
      
      // sets neural network for improve() and
      // getCurrentError() and getError() calls
      // also resets error values
      void setData(neuralnetwork<T>* nn,
		   const dataset<T>* input,
		   const dataset<T>* output);
      
      // resets with a single dataset containing
      // both input and output data
      void setData(neuralnetwork<T>* nn,
		   const dataset<T>* data);
      
      T getError();
      T getCurrentError();
      
    private:
      neuralnetwork<T>* nnetwork;
      const dataset<T>* input;
      const dataset<T>* output;
      T latestError;
      
      /* moments not used (for now) */
      T moment;
    };
};


#include "vertex.h"
#include "neuralnetwork.h"


namespace whiteice
{
  extern template class backpropagation< float >;
  extern template class backpropagation< double >;
  extern template class backpropagation< math::blas_real<float> >;
  extern template class backpropagation< math::blas_real<double> >;
};



#endif

