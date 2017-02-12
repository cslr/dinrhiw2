// Reinforcement learning using continuous state and continuous actions


#include "RIFL_abstract2.h"
#include "NNGradDescent.h"

#include <assert.h>
#include <list>

namespace whiteice
{

  template <typename T>
  RIFL_abstract2<T>::RIFL_abstract2(unsigned int numActions,
				    unsigned int numStates)
  {
    // initializes parameters
    {
      gamma = T(0.8);
      epsilon = T(0.66);

      learningMode = true;
      hasModel = false;

      this->numActions = numActions;
      this->numStates  = numStates;
    }

    
    // initializes neural network architecture and weights randomly
    {
      std::lock_guard<std::mutex> lock(model_mutex);

      std::vector<unsigned int> arch;
      arch.push_back(numStates + numActions);
      arch.push_back(numStates*20);
      arch.push_back(numStates*20);
      //arch.push_back(numStates*100);
      arch.push_back(1);

      {
	whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
	nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	nn.randomize();
	
	Q.importNetwork(nn);
      }

      arch.clear();
      arch.push_back(numStates);
      arch.push_back(numStates*20);
      arch.push_back(numStates*20);
      arch.push_back(numActions);

      {
	whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
	nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	nn.randomize();
	
	policy.importNetwork(nn);
      }
      
    }
    
    
    thread_is_running = 0;
    rifl_thread = nullptr;
  }

  template <typename T>
  RIFL_abstract2<T>::~RIFL_abstract2() throw()
  {
    // stops executing thread
    {
      if(thread_is_running <= 0) return;

      std::lock_guard<std::mutex> lock(thread_mutex);

      if(thread_is_running <= 0) return;

      thread_is_running--;

      if(rifl_thread){
	rifl_thread->join();
	delete rifl_thread;
      }

      rifl_thread = nullptr;
    }
  }

  
  // starts Reinforcement Learning thread
  template <typename T>
  bool RIFL_abstract2<T>::start()
  {
    if(thread_is_running != 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_is_running != 0) return false;

    try{
      thread_is_running++;
      rifl_thread = new thread(std::bind(&RIFL_abstract2<T>::loop, this));
    }
    catch(std::exception& e){
      thread_is_running--;
      rifl_thread = nullptr;

      return false;
    }

    return true;
  }

  
  // stops Reinforcement Learning thread
  template <typename T>
  bool RIFL_abstract2<T>::stop()
  {
    if(thread_is_running <= 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_is_running <= 0) return false;

    thread_is_running--;

    if(rifl_thread){
      rifl_thread->join();
      delete rifl_thread;
    }

    rifl_thread = nullptr;
    return true;
  }

  template <typename T>
  bool RIFL_abstract2<T>::isRunning() const
  {
    return (thread_is_running > 0);
  }


  // epsilon E [0,1] percentage of actions are chosen according to model
  //                 1-e percentage of actions are random (exploration)
  template <typename T>
  bool RIFL_abstract2<T>::setEpsilon(T epsilon) throw()
  {
    if(epsilon < T(0.0) || epsilon > T(1.0)) return false;
    this->epsilon = epsilon;
    return true;
  }
  

  template <typename T>
  T RIFL_abstract2<T>::getEpsilon() const throw()
  {
    return epsilon;
  }


  template <typename T>
  void RIFL_abstract2<T>::setLearningMode(bool learn) throw()
  {
    learningMode = learn;
  }

  template <typename T>
  bool RIFL_abstract2<T>::getLearningMode() const throw()
  {
    return learningMode;
  }


  template <typename T>
  void RIFL_abstract2<T>::setHasModel(bool hasModel) throw()
  {
    this->hasModel = hasModel;
  }

  template <typename T>
  bool RIFL_abstract2<T>::getHasModel() throw()
  {
    return hasModel;
  }

  
  // saves learnt Reinforcement Learning Model to file
  template <typename T>
  bool RIFL_abstract2<T>::save(const std::string& filename) const
  {
    std::lock_guard<std::mutex> lock(model_mutex);

    return model.save(filename);
  }
  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract2<T>::load(const std::string& filename)
  {
    std::lock_guard<std::mutex> lock(model_mutex);
    
    return model.load(filename);
  }
  

  template <typename T>
  void RIFL_abstract2<T>::loop()
  {
    std::vector< rifl_datapoint<T> > database;
    whiteice::dataset<T> data;
    whiteice::math::NNGradDescent<T> grad;
    unsigned int epoch = 0;

    const unsigned int DATASIZE = 50000;
    const unsigned int SAMPLESIZE = 1000;
    T temperature = T(0.010);

    

    bool firstTime = true;
    whiteice::math::vertex<T> state;

    while(thread_is_running > 0){

      // 1. gets current state
      {
	auto oldstate = state;
      
	if(getState(state) == false){
	  state = oldstate;
	  if(firstTime) continue;
	}

	firstTime = false;
      }

      // 2. selects action using policy (+ add random noise ~Normal for exploration)
      whiteice::math::vertex<T> action(numActions);
      
      {
	std::lock_guard<std::mutex> lock(policy_mutex);

	whiteice::math::vertex<T> u;
	whiteice::math::matrix<T> e;

	if(policy.calculate(state, u, e, 1, 0) == true){
	  if(u.size() != numActions){
	    u.resize(numActions);
	    for(unsigned int i=0;i<numActions;i++){
	      u[i] = T(0.0);
	    }
	  }
	}
	else{
	  u.resize(numActions);
	  for(unsigned int i=0;i<numActions;i++){
	    u[i] = T(0.0);
	  }
	}

	// TODO add random normally distributed noise (exploration)
	if(learningMode){
	  
	}

	action = u;
      }

      whiteice::math::vertex<T> newstate;
      T reinforcement = T(0.0);

      // 3. perform action and get newstate and reinforcement
      {
	if(performAction(action, newstate, reinforcement) == false){
	  continue;
	}
      }

      
      if(learningMode == false){
	continue; // we do not do learning
      }
      

      // 4. updates database (of actions and responses)
      {
	struct rifl2_datapoint<T> data;

	data.state = state;
	data.action = action;
	data.newstate = newstate;
	data.reinforcement = reinforcement;	

	if(database.size() >= DATASIZE){
	  const unsigned int index = rng.rand() % database.size();
	  database[index] = data;
	}
	else{
	  database.push_back(data);
	}

	printf("DATABASE SIZE: %d\n", (int)database.size());
	fflush(stdout);
      }

      
      // 5. update/optimize Q(state, action) network
      // activates batch learning if it is not running
      if(database.size() >= SAMPLESIZE)
      {
	if(grad.isRunning() == false){
	  whiteice::nnetwork<T> nn;
	  T error;
	  unsigned int iters;

	  if(grad.getSolution(nn, error, iters) == false){
	    std::vector< math::vertex<T> > weights;

	    std::lock_guard<std::mutex> lock(Q_mutex);
	    
	    if(Q.exportSamples(nn, weights, 1) == false){
	      assert(0);
	    }
	    
	    assert(weights.size() > 0);
	    
	    if(nn.importdata(weights[0]) == false){
	      assert(0);
	    }
	  }
	  else{
	    std::lock_guard<std::mutex> lock(Q_mutex);
	    Q.importNetwork(nn);
	    hasModel = true;
	  }

	  epoch++;
	    
	  const unsigned int BATCHSIZE = database.size()/2;
	  
	  data.clear();
	  data.createCluster("input-state", numStates + numActions);
	  data.createCluster("output-action", 1);
	  
	  for(unsigned int i=0;i<BATCHSIZE;){
	    const unsigned int index = rng.rand() % database.size();

	    whiteice::math::vertex<T> in = database[index].state;
	    
	    whiteice::math::vertex<T> in(numStates + numActions);
	    in.zero();
	    in.write_subvertex(database[index].state, 0);
	    in.write_subvertex(database[index].action, numStates);
	    
	    whiteice::math::vertex<T> out(1);
	    out.zero();

	    out[0] = database[index].reinforcement;
	    
	    // calculates updated utility value
	    whiteice::math::vertex<T> y(1);
	    
	    {
	      whiteice::math::vertex<T> tmp(numStates + numActions);
	      tmp.write_subvertex(database[index].newstate, 0);
	      
	      {
		whiteice::math::vertex<T> u; // new action..
		whiteice::math::matrix<T> e;

		policy.calculate(database[index].newstate, u, e, 1, 0);

		// add exporation noise?

		tmp.write_subvertex(u, numStates); // writes policy's action
	      }
	      
	      nn.calculate(tmp, y);

	      out[0] += gamma*y[0];
	    }

	    data.add(0, in);
	    data.add(1, out);

	    i++;
	  }

	  grad.startOptimize(data, nn, 2, 150);
	}
	else{
	  whiteice::nnetwork<T> nn;
	  T error = T(0.0);
	  unsigned int iters = 0;

	  if(grad.getSolution(nn, error, iters)){
	    printf("EPOCH %d OPTIMIZER %d ITERS: ERROR %f\n", epoch, iters, error.c[0]);
	  }
	}
      }


      // 6. update/optimize policy(state) network
      // activates batch learning if it is not running
      if(database.size() >= SAMPLESIZE)
      {
	if(grad2.isRunning())
	
      }
      
    }

    grad.stopComputation();
  }

  template class RIFL_abstract2< math::blas_real<float> >;
  template class RIFL_abstract2< math::blas_real<double> >;
  
};
