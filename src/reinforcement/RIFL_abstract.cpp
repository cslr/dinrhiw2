
#include "RIFL_abstract.h"
#include "NNGradDescent.h"

#include <assert.h>
#include <list>

namespace whiteice
{

  template <typename T>
  RIFL_abstract<T>::RIFL_abstract(unsigned int numActions,
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
      arch.push_back(numStates);
      // arch.push_back(numStates*100);
      arch.push_back(numStates*20);
      arch.push_back(numStates*20);
      arch.push_back(numActions);

      whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);
      nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
      nn.randomize();

      model.importNetwork(nn);

      // creates empty preprocessing
      preprocess.createCluster("input-state", numStates);
      preprocess.createCluster("output-action", numActions);
    }
    
    
    thread_is_running = 0;
    rifl_thread = nullptr;
  }

  template <typename T>
  RIFL_abstract<T>::~RIFL_abstract() throw()
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
  bool RIFL_abstract<T>::start()
  {
    if(thread_is_running != 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_is_running != 0) return false;

    try{
      thread_is_running++;
      rifl_thread = new thread(std::bind(&RIFL_abstract<T>::loop, this));
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
  bool RIFL_abstract<T>::stop()
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
  bool RIFL_abstract<T>::isRunning() const
  {
    return (thread_is_running > 0);
  }


  // epsilon E [0,1] percentage of actions are chosen according to model
  //                 1-e percentage of actions are random (exploration)
  template <typename T>
  bool RIFL_abstract<T>::setEpsilon(T epsilon) throw()
  {
    if(epsilon < T(0.0) || epsilon > T(1.0)) return false;
    this->epsilon = epsilon;
    return true;
  }
  

  template <typename T>
  T RIFL_abstract<T>::getEpsilon() const throw()
  {
    return epsilon;
  }


  template <typename T>
  void RIFL_abstract<T>::setLearningMode(bool learn) throw()
  {
    learningMode = learn;
  }

  template <typename T>
  bool RIFL_abstract<T>::getLearningMode() const throw()
  {
    return learningMode;
  }


  template <typename T>
  void RIFL_abstract<T>::setHasModel(bool hasModel) throw()
  {
    this->hasModel = hasModel;
  }

  template <typename T>
  bool RIFL_abstract<T>::getHasModel() throw()
  {
    return hasModel;
  }

  
  // saves learnt Reinforcement Learning Model to file
  template <typename T>
  bool RIFL_abstract<T>::save(const std::string& filename) const
  {
    std::lock_guard<std::mutex> lock(model_mutex);

    std::string pfilename = filename + ".preprocess";

    return model.save(filename) && preprocess.save(pfilename);
  }
  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract<T>::load(const std::string& filename)
  {
    std::lock_guard<std::mutex> lock(model_mutex);

    std::string pfilename = filename + ".preprocess";
    
    return model.load(filename) && preprocess.load(pfilename);
  }
  

  template <typename T>
  void RIFL_abstract<T>::loop()
  {
    std::vector< rifl_datapoint<T> > database;
    
    whiteice::dataset<T> data;
    whiteice::math::NNGradDescent<T> grad(false, true);
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

      // 2. activates neural networks to get utility values for each command
      std::vector<T> U;
      
      {
	std::lock_guard<std::mutex> lock(model_mutex);
	
	whiteice::math::vertex<T> u;
	whiteice::math::matrix<T> e;

	whiteice::math::vertex<T> input = state;

	preprocess.preprocess(0, input);

	if(model.calculate(input, u, e, 1, 0) == true){
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

	for(unsigned int i=0;i<u.size();i++){
	  U.push_back(u[i]);
	}

#if 0
	for(auto& ui : U){
	  printf("%f ", ui.c[0]);
	}
	printf("\n");
#endif
      }

      // 3. selects action according to probabilities
      unsigned int action = 0;

      {
#if 0
	T psum = T(0.0);

	std::vector<T> p;

	for(unsigned int i=0;i<U.size();i++){
	  psum += exp(U[i]/temperature);
	  p.push_back(psum);
	}

	for(unsigned int i=0;i<U.size();i++){
	  p[i] /= psum;
	}

	T r = rng.uniform();
	
	unsigned int index = 0;

	while(p[index] < r) index++;

	action = index;
#endif	
	
#if 1
	T r = rng.uniform();

	if(learningMode == false)
	  r = T(0.0); // always selects the largest value

	if(r < epsilon){ // EPSILON% selects the largest value
	  T maxv = U[action];
	  
	  for(unsigned int i=0;i<U.size();i++){
	    if(maxv < U[i]){
	      action = i;
	      maxv = U[i];
	    }
	  }
	}
	else{ // (100 - EPSILON)% select action randomly
	  action = rng.rand() % U.size();
	}
#endif

	// if we don't have not yet optimized model, then we make random choices
	if(hasModel == false)
	  action = rng.rand() % U.size();
      }
      
      whiteice::math::vertex<T> newstate;
      T reinforcement = T(0.0);

      // 4. perform action 
      {
	if(performAction(action, newstate, reinforcement) == false){
	  continue;
	}
      }

      if(learningMode == false){
	continue; // we do not do learning
      }

      // 6. updates database
      {
	struct rifl_datapoint<T> data;

	data.state = state;
	data.newstate = newstate;
	data.reinforcement = reinforcement;
	data.action = action;

	if(database.size() >= DATASIZE){
	  const unsigned int index = rng.rand() % database.size();
	  database[index] = data;
	}
	else{
	  database.push_back(data);
	}

	// printf("DATABASE SIZE: %d\n", (int)database.size());
	// fflush(stdout);
      }


      // activates batch learning if it is not running
      if(database.size() >= SAMPLESIZE)
      {
	if(grad.isRunning() == false){
	  printf("*************************************************************\n");
	  
	  whiteice::nnetwork<T> nn;
	  T error;
	  unsigned int iters;

	  if(grad.getSolution(nn, error, iters) == false){
	    std::vector< math::vertex<T> > weights;

	    std::lock_guard<std::mutex> lock(model_mutex);
	    
	    if(model.exportSamples(nn, weights, 1) == false){
	      assert(0);
	    }
	    
	    assert(weights.size() > 0);
	    
	    if(nn.importdata(weights[0]) == false){
	      assert(0);
	    }
	  }
	  else{
	    std::lock_guard<std::mutex> lock(model_mutex);
	    model.importNetwork(nn);
	    
	    data.clearData(0);
	    data.clearData(1);

	    preprocess = data;
	    
	    hasModel = true;
	  }

	  epoch++;
	    
	  const unsigned int BATCHSIZE = database.size()/2;
	  
	  data.clear();
	  data.createCluster("input-state", numStates);
	  data.createCluster("output-action", numActions);
	  
	  for(unsigned int i=0;i<BATCHSIZE;){
	    const unsigned int index = rng.rand() % database.size();

	    whiteice::math::vertex<T> in = database[index].state;
	    whiteice::math::vertex<T> out(numActions);
	    out.zero();
	    
	    // calculates updated utility value
	    
	    whiteice::math::vertex<T> u;
#if 0
	    T u_value = T(0.0);
	    
	    {
	      nn.calculate(database[index].state, u);
	      u_value = u[database[index].action];
	    }
#endif
	    
	    
	    T unew_value = T(0.0);
	    
	    {
	      T maxvalue = T(-INFINITY);
	      
	      if(nn.calculate(database[index].newstate, u)){
		for(unsigned int i=0;i<u.size();i++){
		  if(maxvalue < u[i]){
		    maxvalue = u[i];
		  }
		}
	      }
	      
	      unew_value = database[index].reinforcement + gamma*maxvalue;
	    }
	    
	    // out[database[index].action] = unew_value - u_value;

	    for(unsigned int i=0;i<out.size();i++)
	      out[i] = T(NAN);
	    
	    out[database[index].action] = unew_value;

	    data.add(0, in);
	    data.add(1, out);

	    i++;
	  }
	  
	  // add preprocessing to dataset
	  data.preprocess(0, whiteice::dataset<T>::dnMeanVarianceNormalization);

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
      
    }

    grad.stopComputation();
  }

  template class RIFL_abstract< math::blas_real<float> >;
  template class RIFL_abstract< math::blas_real<double> >;
  
};
