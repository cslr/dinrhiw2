/*
 * Reinforcement learning for 
 * discrete actions and continuous states
 * uses neural network (nnetwork) to learn 
 * utility function which is used to select 
 * the next action.
 * 
 * Implementation is mostly based on the following paper:
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
#include "dataset.h"
#include "RNG.h"


namespace whiteice
{

  template <typename T = math::blas_real<float> >
    class RIFL_abstract
    {
    public:

    /*
     * numActions        - the number of discrete different actions
     * numStates         - the number of dimensions in state vectors
     * dimActionFeatures - the number of dimensoins in action vectors
     *                     (numActions different feature vectors)
     */
    RIFL_abstract(const unsigned int numActions,
		  const unsigned int numStates,
		  const unsigned int dimActionFeatures);
    ~RIFL_abstract() throw();

    // starts Reinforcement Learning thread
    bool start();

    // stops Reinforcement Learning thread
    bool stop();
    
    bool isRunning() const;

    // epsilon E [0,1] percentage of actions are chosen according to model
    //                 1-e percentage of actions are random (exploration)
    bool setEpsilon(T epsilon) throw();

    T getEpsilon() const throw();

    /*
     * sets/gets learning mode 
     * (do we do just control or also try to learn from data)
     */
    void setLearningMode(bool learn) throw();
    bool getLearningMode() const throw();

    /*
     * hasModel flag means we have a proper model
     * (from optimization or from load)
     *
     * as long as we don't have a proper model
     * we make random actions (initially) 
     */
    void setHasModel(bool hasModel) throw();
    bool getHasModel() throw();

    // saves learnt Reinforcement Learning Model to file
    bool save(const std::string& filename) const;
    
    // loads learnt Reinforcement Learning Model from file
    bool load(const std::string& filename);

    protected:

    unsigned int numActions, numStates, dimActionFeatures;

    virtual bool getState(whiteice::math::vertex<T>& state) = 0;
    
    virtual bool performAction(const unsigned int action,
			       whiteice::math::vertex<T>& newstate,
			       T& reinforcement) = 0;

    virtual bool getActionFeature(const unsigned int action,
				  whiteice::math::vertex<T>& feature) = 0;

    // helper function, returns minimum value in vec
    unsigned int min(const std::vector<unsigned int>& vec) const throw();

    // separate network for each action
    whiteice::bayesian_nnetwork<T> model;
    whiteice::dataset<T> preprocess;
    mutable std::mutex model_mutex;

    bool hasModel;
    bool learningMode;
    
    T epsilon;
    T gamma;
    
    whiteice::RNG<T> rng;

    volatile int thread_is_running;
    std::thread* rifl_thread;
    std::mutex thread_mutex;
    
    void loop();
    
    
    };

  template <typename T>
    struct rifl_datapoint
    {
      whiteice::math::vertex<T> state, newstate;
      T reinforcement;
    };


  extern template class RIFL_abstract< math::blas_real<float> >;
  extern template class RIFL_abstract< math::blas_real<double> >;
};

#endif
