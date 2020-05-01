/*
 * Reinforcement learning for 
 * continuous actions and continuous states
 * uses neural network (nnetwork) to learn 
 * utility (Q) and policy functions
 * 
 * Implementation is mostly based on the following paper:
 * 
 * Continuous Control With Deep Reinforcement Learning.
 * Timothy P. Lillicrap*, Jonathan J. Hunt*, 
 * Alexander Pritzel, Nicolas Heess, Tom Erez, 
 * Yuval Tassa, David Silver & Daan Wierstra
 * Google DeepMind, London, UK.
 * Conference paper at ICLR 2016
 *
 */

#ifndef whiteice_RIFL_abstract2_h
#define whiteice_RIFL_abstract2_h

#include <string>
#include <mutex>
#include <thread>
#include <vector>

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "bayesian_nnetwork.h"
#include "RNG.h"
#include "dataset.h"


namespace whiteice
{
  template <typename T>
    class CreateRIFL2dataset;

  template <typename T>
    class CreatePolicyDataset;

  template <typename T = math::blas_real<float> >
    class RIFL_abstract2
    {
    public:

    // parameters are dimensions of vectors dimActions and dimStates: R^d
    RIFL_abstract2(unsigned int numActions, unsigned int numStates);
    ~RIFL_abstract2() ;

    // starts Reinforcement Learning thread
    bool start();

    // stops Reinforcement Learning thread
    bool stop();
    
    bool isRunning() const;

    // epsilon E [0,1] percentage of actions are chosen according to model
    //                 1-e percentage of actions are random (exploration)
    bool setEpsilon(T epsilon) ;

    T getEpsilon() const ;

    /*
     * sets/gets learning mode 
     * (do we do just control or also try to learn from data)
     */
    void setLearningMode(bool learn) ;
    bool getLearningMode() const ;

    /*
     * hasModel flag means we have a proper model
     * (from optimization or from load)
     *
     * as long as we don't have a proper model
     * we make random actions (initially) 
     */
    void setHasModel(unsigned int hasModel) ;
    unsigned int getHasModel() ;

    // saves learnt Reinforcement Learning Model to file
    bool save(const std::string& filename) const;
    
    // loads learnt Reinforcement Learning Model from file
    bool load(const std::string& filename);

    protected:

    unsigned int numActions, numStates; // dimensions of R^d vectors

    virtual bool getState(whiteice::math::vertex<T>& state) = 0;

    // action vector is [0,1]^d (output of sigmoid non-linearity)
    virtual bool performAction(const whiteice::math::vertex<T>& action,
			       whiteice::math::vertex<T>& newstate,
			       T& reinforcement) = 0;

    // reinforcement Q model: Q(state, action) ~ discounted future cost
    whiteice::bayesian_nnetwork<T> Q;
    whiteice::dataset<T> Q_preprocess;
    mutable std::mutex Q_mutex;

    // f(state) = action
    whiteice::bayesian_nnetwork<T> policy;
    whiteice::dataset<T> policy_preprocess;
    mutable std::mutex policy_mutex;

    std::vector<unsigned int> hasModel;
    bool learningMode;
    
    T epsilon;
    T gamma;
    
    RNG<T> rng;

    volatile int thread_is_running;
    std::thread* rifl_thread;
    std::mutex thread_mutex;
    
    void loop();
    
    // friend thread class to do heavy computations in background
    // out of main loop 
    friend class CreateRIFL2dataset<T>;

    friend class CreatePolicyDataset<T>;
    
    };

  template <typename T>
    struct rifl2_datapoint
    {
      whiteice::math::vertex<T> state, newstate;
      whiteice::math::vertex<T> action;
      T reinforcement;
    };


  extern template class RIFL_abstract2< math::blas_real<float> >;
  extern template class RIFL_abstract2< math::blas_real<double> >;
};

#include "CreateRIFL2dataset.h"
#include "CreatePolicyDataset.h"

#endif
