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

#include <string>
#include <mutex>
#include <thread>
#include <vector>

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "bayesian_nnetwork.h"
#include "RNG.h"


namespace whiteice
{

  template <typename T = math::blas_real<float> >
    class RIFL_abstract
    {
    public:
    
    RIFL_abstract(unsigned int numActions, unsigned int numStates);
    ~RIFL_abstract() throw();

    // starts Reinforcement Learning thread
    bool start();

    // stops Reinforcement Learning thread
    bool stop();
    
    bool isRunning() const;

    // saves learnt Reinforcement Learning Model to file
    bool save(const std::string& filename) const;
    
    // loads learnt Reinforcement Learning Model from file
    bool load(const std::string& filename);

    protected:

    unsigned int numActions, numStates;

    virtual bool getState(whiteice::math::vertex<T>& state) = 0;
    
    virtual bool performAction(const unsigned int action,
			       whiteice::math::vertex<T>& newstate,
			       T& r) = 0;

    std::vector< whiteice::bayesian_nnetwork<T> > models;
    
    T temperature;
    T gamma;
    
    T lrate; // learning rate

    RNG<T> rng;

    volatile int thread_is_running;
    std::thread* rifl_thread;
    std::mutex thread_mutex;
    
    void loop();
    
    
    };


  extern template class RIFL_abstract< float >;
  extern template class RIFL_abstract< double >;
  extern template class RIFL_abstract< math::blas_real<float> >;
  extern template class RIFL_abstract< math::blas_real<double> >;
};

#endif
