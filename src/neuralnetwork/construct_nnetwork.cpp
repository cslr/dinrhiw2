/*
 * implements function that generates pre-trained nnetwork from DBN and adds output layer to it
 *
 */

#include "construct_nnetwork.h"

namespace whiteice
{
  /*
   * function to construct deep nnetwork from binary DBN generated using stacked binary RBMs
   * 
   * dbn             - trained deep belief network with data
   * nnetwork        - deep neural network to be generated
   * outputDimension - output dimension of data (the final layer's weights are set 
   *                                             to small random values)
   *
   * the generated nnetwork has binary (0/1) inputs and the output can be anything 
   * that one wants to train to
   * 
   */
  template <typename T>
  bool construct_nnetwork(const DBN<T>& dbn, 
			  nnetwork<T>* lreg_nnetwork, 
			  const unsigned int outputDimension)
  {
    // not implemented yet
    
    assert(0); // IMPLEMENT ME!!!
    
    return false;
  }
  
};



