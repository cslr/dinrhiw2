// Reinforcement learning using continuous state and continuous actions


#include "RIFL_abstract2.h"

#include "NNGradDescent.h"
#include "PolicyGradAscent.h"

#include "Log.h"
#include "blade_math.h"

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

      hasModel.resize(2);
      hasModel[0] = 0; // Q-network
      hasModel[1] = 0; // policy-network
      
      this->numActions = numActions;
      this->numStates  = numStates;
    }

    
    // initializes neural network architecture and weights randomly
    {
      std::vector<unsigned int> arch;
      
      {
	std::lock_guard<std::mutex> lock(Q_mutex);
	
	arch.push_back(numStates + numActions);
	arch.push_back(numStates*20);
	arch.push_back(numStates*20);
	arch.push_back(1);
	
	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	  nn.randomize();
	  
	  Q.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract2: ctor Q diagnostics");
	  Q.diagnosticsInfo();
	}
      }

      {
	std::lock_guard<std::mutex> lock(policy_mutex);
	
	arch.clear();
	arch.push_back(numStates);
	arch.push_back(numStates*20);
	arch.push_back(numStates*20);
	arch.push_back(numActions);

	// policy outputs action is (should be) [0,1]^D vector
	{
	  whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
	  nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::sigmoid);
	  // nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
	  nn.randomize();
	  
	  policy.importNetwork(nn);

	  whiteice::logging.info("RIFL_abstract2: ctor policy diagnostics");
	  policy.diagnosticsInfo();
	}
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
      whiteice::logging.info("RIFL_abstract2: starting main thread");
      
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
  void RIFL_abstract2<T>::setHasModel(unsigned int hasModel) throw()
  {
    this->hasModel[0] = hasModel;
    this->hasModel[1] = hasModel;
  }

  template <typename T>
  unsigned int RIFL_abstract2<T>::getHasModel() throw()
  {
    if(hasModel[0] < hasModel[1]) return hasModel[0];
    else return hasModel[1];
  }

  
  // saves learnt Reinforcement Learning Model to file
  template <typename T>
  bool RIFL_abstract2<T>::save(const std::string& filename) const
  {
    std::lock_guard<std::mutex> lock1(Q_mutex);
    std::lock_guard<std::mutex> lock2(policy_mutex);

    char buffer[256];
    
    snprintf(buffer, 256, "%s-q", filename.c_str());    
    if(Q.save(buffer) == false) return false;

    snprintf(buffer, 256, "%s-policy", filename.c_str());
    if(policy.save(buffer) == false) return false;

    snprintf(buffer, 256, "%s-q-preprocess", filename.c_str());    
    if(Q_preprocess.save(buffer) == false) return false;

    snprintf(buffer, 256, "%s-policy-preprocess", filename.c_str());
    if(policy_preprocess.save(buffer) == false) return false;

    return true;
  }

  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract2<T>::load(const std::string& filename)
  {
    std::lock_guard<std::mutex> lock1(Q_mutex);
    std::lock_guard<std::mutex> lock2(policy_mutex);

    char buffer[256];
        
    snprintf(buffer, 256, "%s-q", filename.c_str());    
    if(Q.load(buffer) == false) return false;

    snprintf(buffer, 256, "%s-policy", filename.c_str());
    if(policy.load(buffer) == false) return false;

    snprintf(buffer, 256, "%s-q-preprocess", filename.c_str());    
    if(Q_preprocess.load(buffer) == false) return false;

    snprintf(buffer, 256, "%s-policy-preprocess", filename.c_str());
    if(policy_preprocess.load(buffer) == false) return false;
    
    return true;
  }
  

  template <typename T>
  void RIFL_abstract2<T>::loop()
  {
    std::vector< rifl2_datapoint<T> > database;
    
    whiteice::dataset<T> data;    
    whiteice::math::NNGradDescent<T> grad; // Q(state,action) model optimizer

    whiteice::dataset<T> data2;
    whiteice::PolicyGradAscent<T> grad2;   // policy(state)=action model optimizer
    
    std::vector<unsigned int> epoch;

    epoch.resize(2);
    epoch[0] = 0;
    epoch[1] = 0;

    std::vector<bool> hasPreprocess;

    hasPreprocess.resize(2);
    hasPreprocess[0] = false;
    hasPreprocess[1] = false;

    const unsigned int DATASIZE = 10000;
    const unsigned int SAMPLESIZE = 100;

    // const T tau = T(0.30); // we keep 30% of the new networks weights (60% old)
    const T tau = T(1.00); // we keep 100% of the new networks weights (0% old)

    const bool debug = true; // debugging messages

    bool firstTime = true;
    whiteice::math::vertex<T> state;

    whiteice::logging.info("RIFL_abstract2: starting optimization loop");

    whiteice::logging.info("RIFL_abstract2: initial Q diagnostics");
    Q.diagnosticsInfo();

    while(thread_is_running > 0){

      // 1. gets current state
      {
	if(debug)
	  whiteice::logging.info("RIFL_abstract2: get current state");
	
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
	if(debug)
	  whiteice::logging.info("RIFL_abstract2: use policy to select action");
	
	std::lock_guard<std::mutex> lock(policy_mutex);

	whiteice::math::vertex<T> u;
	whiteice::math::matrix<T> e;

	auto input = state;
	policy_preprocess.preprocess(0, input);

	if(policy.calculate(input, u, e, 1, 0) == true){
	  if(u.size() != numActions){
	    u.resize(numActions);
	    for(unsigned int i=0;i<numActions;i++){
	      u[i] = T(0.0);
	    }
	  }
	  else{
	    policy_preprocess.invpreprocess(1, u);
	  }
	}
	else{
	  u.resize(numActions);
	  for(unsigned int i=0;i<numActions;i++){
	    u[i] = T(0.0);
	  }
	}

	// it is assumed that action data should have zero mean and is roughly
	// normally distributed (with StDev[n] = 1) so data is close to zero

	// FIXME add better random normally distributed noise (exploration)
	{
	  if(rng.uniform() > epsilon){ // 1-epsilon % are chosen randomly
#if 0
	    auto noise = u;
	    rng.normal(noise); // Normal E[n]=0 StDev[n]=1
	    u += noise;
#endif
	    rng.normal(u); // Normal E[n]=0 StDev[n]=1
	  }

#if 0
	  for(unsigned int i=0;i<u.size();i++){
	    if(u[i] < T(0.0)) u[i] = T(0.0); // [keep things between [0,1]
	    else if(u[i] > T(1.0)) u[i] = T(1.0);
	  }
#endif
	}

	// if there's no model then make random selection (normally distributed)
	if(hasModel[0] == 0 || hasModel[1] == 0){
	  rng.normal(u);

#if 0
	  for(unsigned int i=0;i<u.size();i++){
	    if(u[i] < T(0.0)) u[i] = T(0.0); // [keep things between [0,1]
	    else if(u[i] > T(1.0)) u[i] = T(1.0);
	  }
#endif
	}

	action = u;
      }

      whiteice::math::vertex<T> newstate;
      T reinforcement = T(0.0);

      // 3. perform action and get newstate and reinforcement
      {
	if(debug)
	  whiteice::logging.info("RIFL_abstract2: perform action");
	
	if(performAction(action, newstate, reinforcement) == false){
	  continue;
	}
      }

      
      if(learningMode == false){
	continue; // we do not do learning
      }
      

      // 4. updates database (of actions and responses)
      {
	if(debug)
	  whiteice::logging.info("RIFL_abstract2: update database");
	
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

	{
	  char buffer[128];
	  snprintf(buffer, 128, "RIFL_abstract2: database size: %d",
		   (int)database.size());
	  whiteice::logging.info(buffer);
	}
	
      }

      
      // 5. update/optimize Q(state, action) network
      // activates batch learning if it is not running
      if(database.size() >= SAMPLESIZE)
      {
	// skip if other optimization step is behind us
	if(epoch[0] > epoch[1])
	  goto q_optimization_done;

	if(debug)
	  whiteice::logging.info("RIFL_abstract2: update/optimize Q-network");
	
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
	    
	    if(epoch[0] > 0){
	      std::lock_guard<std::mutex> lock(Q_mutex);
	      
	      // we keep previous network to some degree
	      {
		whiteice::nnetwork<T> nnprev = nn;
		std::vector< whiteice::math::vertex<T> > prevweights;
		
		if(Q.exportSamples(nnprev, prevweights)){
		  whiteice::math::vertex<T> newweights;
		  
		  if(nn.exportdata(newweights)){
		    newweights = tau*newweights + (T(1.0)-tau)*prevweights[0];
		    
		    nn.importdata(newweights);
		  }
		}
		
		Q.importNetwork(nn);
		
		whiteice::logging.info("RIFL_abstract2: new Q diagnostics");
		Q.diagnosticsInfo();
		
		hasModel[0]++;
	      }
	    }
	    else{
	      std::lock_guard<std::mutex> lock(Q_mutex);
	      Q.importNetwork(nn);

	      whiteice::logging.info("RIFL_abstract2: new Q diagnostics");
	      Q.diagnosticsInfo();

	      hasModel[0]++;
	    }

	    {
	      data.clearData(0);
	      data.clearData(1);

	      std::lock_guard<std::mutex> lock(Q_mutex);
	      Q_preprocess = data;
	      hasPreprocess[0] = true;
	    }

	    epoch[0]++;
	  }

	  // const unsigned int BATCHSIZE = database.size()/2;
	  const unsigned int BATCHSIZE = 1000;

	  bool newPreprocess = false;

#if 0
	  if(data.getNumberOfClusters() != 2){
	    data.clear();
	    data.createCluster("input-state", numStates + numActions);
	    data.createCluster("output-action", 1);
	    newPreprocess = true;
	  }
	  else{
	    data.clearData(0);
	    data.clearData(1);
	    newPreprocess = false;
	  }
#else
	  {
	    data.clear();
	    data.createCluster("input-state", numStates + numActions);
	    data.createCluster("output-action", 1);
	    newPreprocess = true;
	  }
#endif
	  
	  for(unsigned int i=0;i<BATCHSIZE;){
	    const unsigned int index = rng.rand() % database.size();

	    whiteice::math::vertex<T> in(numStates + numActions);
	    in.zero();
	    in.write_subvertex(database[index].state, 0);
	    in.write_subvertex(database[index].action, numStates);
	    
	    whiteice::math::vertex<T> out(1);
	    out.zero();

	    // calculates updated utility value
	    whiteice::math::vertex<T> y(1);
	    
	    {
	      whiteice::math::vertex<T> tmp(numStates + numActions);
	      tmp.write_subvertex(database[index].newstate, 0);
	      
	      {
		whiteice::math::vertex<T> u; // new action..
		whiteice::math::matrix<T> e;

		auto input = database[index].newstate;
		
		policy_preprocess.preprocess(0, input);
		
		policy.calculate(input, u, e, 1, 0);
		
		policy_preprocess.invpreprocess(1, u);

		// add exploration noise?

		tmp.write_subvertex(u, numStates); // writes policy's action
	      }

	      Q_preprocess.preprocess(0, tmp);
	      
	      nn.calculate(tmp, y);

	      Q_preprocess.invpreprocess(1, y);
	      
	      if(epoch[0] > 0){
		out[0] = database[index].reinforcement + gamma*y[0];
	      }
	      else{ // the first iteration of reinforcement learning do not use Q
		out[0] = database[index].reinforcement;
	      }

	    }

	    data.add(0, in);
	    data.add(1, out);

	    i++;
	  }

	  if(newPreprocess){
#if 1
	    data.preprocess
	      (0, whiteice::dataset<T>::dnMeanVarianceNormalization);

	    data.preprocess
	      (1, whiteice::dataset<T>::dnMeanVarianceNormalization);
#endif
	  }

	  // DEBUG: saves dataset used for training to disk
	  // data.exportAscii("rifl-abstract2-q-dataset.txt", true, true);

	  // grad.startOptimize(data, nn, 2, 150);
	  grad.startOptimize(data, nn, 1, 150);
	}
	else{
	  whiteice::nnetwork<T> nn;
	  T error = T(0.0);
	  unsigned int iters = 0;

	  if(grad.getSolutionStatistics(error, iters)){
	    char buffer[128];
	    
	    double e;
	    whiteice::math::convert(e, error);
	    
	    snprintf(buffer, 128,
		     "RIFL_abstract2: Q-optimizer epoch %d iter %d error %f",
		     epoch[0], iters, e);
	    
	    whiteice::logging.info(buffer);
	  }
	}
      }
      
    q_optimization_done:

      
      // 6. update/optimize policy(state) network
      // activates batch learning if it is not running
      
      if(database.size() >= SAMPLESIZE)
      {
	// skip if other optimization step is behind us
	// we only start calculating policy after Q() has been optimized..
	if(epoch[1] > epoch[0] || epoch[0] == 0 || hasPreprocess[0] == false) 
	  goto policy_optimization_done;

	if(debug)
	  whiteice::logging.info("RIFL_abstract2: update/optimize policy-network");

	
	if(grad2.isRunning() == false){
	  whiteice::nnetwork<T> nn;
	  T meanq;
	  unsigned int iters;

	  if(grad2.getSolution(nn, meanq, iters) == false){
	    std::vector< math::vertex<T> > weights;

	    std::lock_guard<std::mutex> lock(policy_mutex);
	    
	    if(policy.exportSamples(nn, weights, 1) == false){
	      assert(0);
	    }
	    
	    assert(weights.size() > 0);
	    
	    if(nn.importdata(weights[0]) == false){
	      assert(0);
	    }
	  }
	  else{
	    if(epoch[1] > 0){
	      std::lock_guard<std::mutex> lock(policy_mutex);
	      
	      // we keep previous network to some degree
	      {
		whiteice::nnetwork<T> nnprev = nn;
		std::vector< whiteice::math::vertex<T> > prevweights;
		
		if(policy.exportSamples(nnprev, prevweights)){
		  whiteice::math::vertex<T> newweights;
		  
		  if(nn.exportdata(newweights)){
		    newweights = tau*newweights + (T(1.0)-tau)*prevweights[0];
		    
		    nn.importdata(newweights);
		  }
		}
		
		policy.importNetwork(nn);
		hasModel[1]++;
	      }
	    }
	    else{
	      std::lock_guard<std::mutex> lock(policy_mutex);
	      policy.importNetwork(nn);
	      hasModel[1]++;
	    }

	    {
	      data2.clearData(0);
	      data2.clearData(1);

	      policy_preprocess = data2;
	      hasPreprocess[1] = true;
	    }

	    epoch[1]++;
	  }

	  // epoch[1]++;
	    
	  // const unsigned int BATCHSIZE = database.size()/2;
	  const unsigned int BATCHSIZE = 1000;

	  bool newPreprocess = false;

#if 0
	  if(data2.getNumberOfClusters() != 1){
	    data2.clear();
	    data2.createCluster("input-state", numStates);
	    newPreprocess = true;
	  }
	  else{
	    data2.clearData(0);
	    data2.clearData(1);
	    newPreprocess = false;
	  }
#else
	  {
	    data2.clear();
	    data2.createCluster("input-state", numStates);
	    newPreprocess = true;
	  }
#endif
	  
	  for(unsigned int i=0;i<BATCHSIZE;i++){
	    const unsigned int index = rng.rand() % database.size();
	    data2.add(0, database[index].state);
	  }

	  {
	    whiteice::nnetwork<T> q_nn;	    

	    {
	      std::lock_guard<std::mutex> lock(Q_mutex);
	      std::vector< math::vertex<T> > weights;
	      
	      if(Q.exportSamples(q_nn, weights, 1) == false){
		assert(0);
	      }
	      
	      assert(weights.size() > 0);
	      
	      if(q_nn.importdata(weights[0]) == false){
		assert(0);
	      }
	    }

	    if(newPreprocess){
#if 1
	      data2.preprocess
		(0, whiteice::dataset<T>::dnMeanVarianceNormalization);
#endif
	    }
	    
	    // grad2.startOptimize(&data2, q_nn, nn, 2, 150);
	    grad2.startOptimize(&data2, q_nn, Q_preprocess, nn, 1, 150);
	  }
	}
	else{
	  whiteice::nnetwork<T> nn;
	  T value = T(0.0);
	  unsigned int iters = 0;

	  if(grad2.getSolution(nn, value, iters)){
	    char buffer[128];
	    
	    double v;
	    whiteice::math::convert(v, value);
	    
	    snprintf(buffer, 128,
		     "RIFL_abstract2: policy-optimizer epoch %d iter %d mean q-value %f",
		     epoch[1], iters, v);
	    
	    whiteice::logging.info(buffer);
	  }
	}
      }
      
    policy_optimization_done:
      
      (1 == 1); // dummy [work-around bug/feature goto requiring expression]
      
    }

    grad.stopComputation();
    grad2.stopComputation();
    
  }

  template class RIFL_abstract2< math::blas_real<float> >;
  template class RIFL_abstract2< math::blas_real<double> >;
  
};
