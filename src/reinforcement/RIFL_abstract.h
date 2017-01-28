/*
 * Reinforcement learning for 
 * discrete actions and continuous states
 * uses neural network (nnetwork) to learn 
 * utility function which is used to select 
 * the next action.
 * 
 * Implementation is based on the following paper:
 * 
 * Self-Improving Reactive Agents Based On 
 * Reinforcement Learning, Planning and Teaching
 * LONG-JI, LIN
 * Machine Learning, 8, 293-321 (1992)
 *
 */

#ifndef whiteice_RIFL_abstract_h
#define whiteice_RIFL_abstract_h

#include "dinrhiw_blas.h"

namespace whiteice
{

  template <typename T = math::blas_real<float> >
    class RIFL_abstract
    {
    public:
    
    RIFL_abstract();
    ~RIFL_abstract() throw();

    
    };

  
};

#endif
