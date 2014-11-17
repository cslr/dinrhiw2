/*
 * pure virtual / abstract class
 */

#ifndef nn_iterative_correction_h
#define nn_iterative_correction_h

namespace whiteice
{
  template <typename T> class neuralnetwork;
  namespace math{ template <typename T> class vertex; };
  
  template <typename T>
    class nn_iterative_correction
    {
    public:
      
      // nn_iterative_correction(){ }
      virtual ~nn_iterative_correction(){ }
      
      virtual bool operator()(neuralnetwork<T>& nn,
			      const math::vertex<T>& correct_output) const = 0;
      
      virtual bool calculate (neuralnetwork<T>& nn,
			      const math::vertex<T>& correct_output) const = 0;
      
      // creates copy of object
      virtual nn_iterative_correction<T>* clone() const = 0;
      
    };
};


#include "vertex.h"
#include "neuralnetwork.h"


#endif
  
