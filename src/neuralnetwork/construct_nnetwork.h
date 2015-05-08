/*
 * implements function that generates pre-trained nnetwork from DBN and adds output layer to it
 *
 */

#ifndef construct_nnetwork_h
#define constrcut_nnetwork_h

#include "DBN.h"
#include "nnetwork.h"

namespace whiteice
{
  
  /*
   * function to construct deep nnetwork from binary DBN generated using stacked binary RBMs
   * 
   * dbn             - trained deep belief network with data
   * nnetwork        - deep neural network to be generated
   * outputDimension - output dimension of data (the final layer's weights are set to small random values)
   *
   * the generated nnetwork has binary (0/1) inputs and the output can be anything that one wants to train to
   * 
   */
  template <typename T>
    bool construct_nnetwork(const DBN<T>& dbn, nnetwork<T>& nnetwork, const unsigned int outputDimension);
};
			    

#endif


