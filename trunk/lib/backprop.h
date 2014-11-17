/*
 * backpropagation based gradient descent
 * for nnetwork (V2) neural network implementation
 *
 */

#ifndef backprop_h
#define backprop_h

#include "atlas.h"
#include "nnetwork.h"
#include "dataset.h"


namespace whiteice
{
  
  template <typename T = math::atlas_real<float> >
    class backprop
    {
      public:
      
      backprop();
      backprop(nnetwork<T>* nn, const dataset<T>* data);
      backprop(nnetwork<T>* nn, 
	       const dataset<T>* in, 
	       const dataset<T>* out);
      
      virtual ~backprop();
      
      bool improve(unsigned int niters = 50);
      
      void setData(nnetwork<T>* nn,
		   const dataset<T>* data);
      
      void setData(nnetwork<T>* nn,
		   const dataset<T>* in,
		   const dataset<T>* out);
      
      T getError();
      T getCurrentError();
      
      private:
      
      nnetwork<T>* nn;
      const dataset<T>* input;
      const dataset<T>* output;
      
      T latestError;
    };
  
  
  extern template class backprop< float >;
  extern template class backprop< double >;
  extern template class backprop< math::atlas_real<float> >;
  extern template class backprop< math::atlas_real<double> >;
};


#endif


