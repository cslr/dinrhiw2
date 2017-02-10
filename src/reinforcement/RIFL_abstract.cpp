
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
      gamma = T(0.80);
      epsilon = T(0.66);

      learningMode = true;
      hasModel = false;
      
      this->numActions = numActions;
      this->numStates  = numStates;

      model_mutex.resize(numActions);
      
      for(auto& m : model_mutex)
	m = new std::mutex;
    }

    
    // initializes neural network architecture and weights randomly
    {
      model.resize(numActions);
      preprocess.resize(numActions);

      updatedModel.resize(numActions);
      updatedPreprocess.resize(numActions);
      
      std::vector<unsigned int> arch;
      arch.push_back(numStates);
      // arch.push_back(numStates*100);
      // arch.push_back(numStates*100);
      arch.push_back(20);
      arch.push_back(20);
      arch.push_back(1);

      whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);
      nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);

      for(unsigned int i=0;i<numActions;i++){
	std::lock_guard<std::mutex> lock(*model_mutex[i]);
	
	nn.randomize();
	model[i].importNetwork(nn);
	
	// creates empty preprocessing
	preprocess[i].createCluster("input-state", numStates);
	preprocess[i].createCluster("output-action", 1);

	updatedModel[i] = model[i];
	updatedPreprocess[i] = preprocess[i];
      }
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

    for(auto& m : model_mutex)
      delete m;
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

    for(unsigned int i=0;i<model.size();i++){
      std::lock_guard<std::mutex> lock(*model_mutex[i]);
      
      char buffer[256];

      snprintf(buffer, 256, "%d-%s", i, filename.c_str());

      if(model[i].save(buffer) == false) return false;

      snprintf(buffer, 256, "%d-preprocess-%s", i, filename.c_str());

      if(preprocess[i].save(buffer) == false) return false;
    }

    return true;
  }
  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract<T>::load(const std::string& filename)
  {
    for(unsigned int i=0;i<model.size();i++){
      std::lock_guard<std::mutex> lock(*model_mutex[i]);
      
      char buffer[256];
	  
      snprintf(buffer, 256, "%d-%s", i, filename.c_str());

      if(model[i].load(buffer) == false) return false;

      snprintf(buffer, 256, "%d-preprocess-%s", i, filename.c_str());

      if(preprocess[i].load(buffer) == false) return false;
    }

    return true;
  }


  // helper function, returns minimum value in v
  template <typename T>
  unsigned int RIFL_abstract<T>::min(const std::vector<unsigned int>& vec) const throw()
  {
    if(vec.size() <= 0) return 0;
    unsigned int min = vec[0];
    for(const auto& v : vec)
      if(v < min) min = v;

    return min;
  }
  

  template <typename T>
  void RIFL_abstract<T>::loop()
  {
    std::vector< std::vector< rifl_datapoint<T> > > database;
    std::vector< whiteice::dataset<T> > data;
    std::vector< whiteice::math::NNGradDescent<T> > grad;

    std::vector<unsigned int> epoch;
    unsigned int updates = 0;
    

    database.resize(model.size());
    data.resize(model.size());
    grad.resize(model.size());
    epoch.resize(model.size());

    for(unsigned int i=0;i<epoch.size();i++)
      epoch[i] = 0;
    
    const unsigned int DATASIZE = 50000;
    const unsigned int SAMPLESIZE = 100;
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
	U.resize(model.size());

	for(unsigned int i=0;i<model.size();i++){
	  std::lock_guard<std::mutex> lock(*model_mutex[i]);
	  
	  whiteice::math::vertex<T> u;
	  whiteice::math::matrix<T> e;
	  
	  whiteice::math::vertex<T> input = state;

	  preprocess[i].preprocess(0, input);

	  if(model[i].calculate(input, u, e, 1, 0) == true){
	    if(u.size() != 1){
	      u.resize(1);
	      u[0] = T(0.0);
	    }
	    else
	      preprocess[i].invpreprocess(1, u);
	  }
	  else{
	    u.resize(1);
	    u[0] = T(0.0);
	  }

	  U[i] = u[0];
	}

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
	

	{ // selects the largest value
	  T maxv = U[action];
	  
	  for(unsigned int i=0;i<U.size();i++){
	    if(maxv < U[i]){
	      action = i;
	      maxv = U[i];
	    }
	  }
	}

	{
	  printf("U = ");
	  for(unsigned int i=0;i<U.size();i++){
	    if(action == i) printf("%f* ", U[i].c[0]);
	    else printf("%f  ", U[i].c[0]);
	  }
	  printf("\n");
	}
	
	// random selection with (1-epsilon) probability
	T r = rng.uniform();
	
	if(learningMode == false)
	  r = T(0.0); // always selects the largest value

	if(r > epsilon){
	  action = rng.rand() % U.size();
	}

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

	if(database[action].size() >= DATASIZE){
	  const unsigned int index = rng.rand() % database[action].size();
	  database[action][index] = data;
	}
	else{
	  database[action].push_back(data);
	}

	// printf("DATABASE SIZE: %d\n", (int)database.size());
	// fflush(stdout);
      }
      

      // activates batch learning if it is not running
      for(unsigned int action=0;action<numActions;action++){
	if(database[action].size() >= SAMPLESIZE)
	{
	  // printf("[%d/%d] EPOCH %d ABOUT TO START ACTION OPTIMIZER *********\n",
	  // action, (int)model.size(), epoch[action]);
	  
	  // only trains single action at the same time
	  {
	    bool oneIsRunning = false;
	    unsigned int index = 0;
	    
	    for(unsigned int i=0;i<grad.size();i++){
	      if(grad[i].isRunning()){
		index = i;
		oneIsRunning = true;
	      }
	    }

	    if(oneIsRunning){
	      if(action == 0){
		// get temporary solutions
		whiteice::nnetwork<T> nn;
		T error = T(0.0);
		unsigned int iters = 0;
		
		if(grad[index].getSolution(nn, error, iters)){
		  printf("[%d/%d] EPOCH %d OPTIMIZER %d ITERS: ERROR %f HASMODEL: %d\n",
			 index, (int)model.size(), 
			 epoch[index], iters, error.c[0], (int)hasModel);
		}
		else{
		  printf("[%d/%d] EPOCH %d GETSOLUTION() FAILED\n",
			 index, (int)model.size(), 
			 epoch[index]);
		}
		
	      }
	      
	      continue;
	    }
	  }
	  
	  if(grad[action].isRunning() == false){
	    
	    if(epoch[action] > min(epoch))
	      continue; // do not start execution until we have minimum epoch

	    if(epoch[action] > 0 && hasModel == false)
	      continue; // do not go the next epochs until we have a model..
	  
	    whiteice::nnetwork<T> nn;
	    T error;
	    unsigned int iters;
	    
	    if(grad[action].getSolution(nn, error, iters) == false){
	      std::vector< math::vertex<T> > weights;

	      std::lock_guard<std::mutex> lock(*model_mutex[action]);
	      
	      if(model[action].exportSamples(nn, weights, 1) == false){
		assert(0);
	      }
	      
	      assert(weights.size() > 0);
	      
	      if(nn.importdata(weights[0]) == false){
		assert(0);
	      }
	    }
	    else{
	      std::lock_guard<std::mutex> lock(*model_mutex[action]);

	      // we keep previous network to some degree
	      // (interpolation between networks)
	      if(hasModel){
		T tau = T(0.3);
		{
		  whiteice::nnetwork<T> nnprev = nn;
		  std::vector< whiteice::math::vertex<T> > prevweights;
		  
		  if(model[action].exportSamples(nnprev, prevweights)){
		    whiteice::math::vertex<T> newweights;
		    
		    if(nn.exportdata(newweights)){
		      newweights = tau*newweights + (T(1.0)-tau)*prevweights[0];
		      
		      nn.importdata(newweights);
		    }
		  }
		  
		  
		  updatedModel[action].importNetwork(nn);
		}
	      }
	      else{
		updatedModel[action].importNetwork(nn);
	      }

	      
	      {
		data[action].clearData(0);
		data[action].clearData(1);
		
		updatedPreprocess[action] = data[action];
	      }

	      epoch[action]++;

	      updates++; // action has updated model [for this epoch]
	    }


	    // when all actions have updated model update the data structures
	    if(updates == updatedModel.size()){
	      printf("*** MODEL UPDATE ***\n");
	      
	      for(unsigned int i=0;i<updatedModel.size();i++){
		std::lock_guard<std::mutex> lock(*model_mutex[i]);
		model[i] = updatedModel[i];
		preprocess[i] = updatedPreprocess[i];
	      }

	      // all actions has processed at least one epoch
	      if(min(epoch) >= 1) 
		hasModel = true;

	      updates = 0;
	    }
	    

	    const unsigned int BATCHSIZE = database[action].size()/2;

	    bool newPreprocess = false;

	    if(data[action].getNumberOfClusters() != 2){
	      data[action].clear();
	      data[action].createCluster("input-state", numStates);
	      data[action].createCluster("output-action", 1);
	      newPreprocess = true;
	    }
	    else{
	      data[action].clearData(0);
	      data[action].clearData(1);
	      newPreprocess = false;
	    }
	    
	    
	    for(unsigned int i=0;i<BATCHSIZE;){
	      const unsigned int index = rng.rand() % database[action].size();
	      
	      whiteice::math::vertex<T> in = database[action][index].state;
	      whiteice::math::vertex<T> out(1);
	      out.zero();
	      
	      // calculates updated utility value
	      
	      whiteice::math::vertex<T> u;
	      
	      T unew_value = T(0.0);

	      {
		T maxvalue = T(-INFINITY);

		for(unsigned int j=0;j<model.size();j++){
		  std::lock_guard<std::mutex> lock(*model_mutex[j]);
		  
		  whiteice::math::vertex<T> u;
		  whiteice::math::matrix<T> e;
		  
		  whiteice::math::vertex<T> input = newstate;

		  preprocess[j].preprocess(0, input);

		  if(model[j].calculate(input, u, e, 1, 0) == true){
		    if(u.size() != 1){
		      u.resize(1);
		      u[0] = T(0.0);
		    }
		    else
		      preprocess[j].invpreprocess(1, u);
		  }
		  else{
		    u.resize(1);
		    u[0] = T(0.0);
		  }
		  
		  if(maxvalue < u[0])
		    maxvalue = u[0];
		}

		if(hasModel == true){
		  unew_value =
		    database[action][index].reinforcement + gamma*maxvalue;
		}
		else{ // first iteration uses raw reinforcement values
		  unew_value =
		    database[action][index].reinforcement;
		}
	      }
	      
	      out[0] = unew_value;
	      
	      data[action].add(0, in);
	      data[action].add(1, out);
	      
	      i++;
	    }
	    
	    // add preprocessing to dataset (only at epoch 0)
	    if(newPreprocess){
	      data[action].preprocess
		(0, whiteice::dataset<T>::dnMeanVarianceNormalization);

#if 0
	      data[action].preprocess
		(1, whiteice::dataset<T>::dnMeanVarianceNormalization);
#endif
	    }

	    const bool dropout = false;
	    
	    if(grad[action].startOptimize(data[action], nn, 2, 250,
					  dropout) == false){
	      printf("[%d/%d] STARTING GRAD OPTIMIZATION FAILED\n",
		     action, (int)model.size());
	    }
	    else{
	      printf("[%d/%d] STARTED GRAD OPTIMIZER EPOCH %d\n",
		     action, (int)model.size(), epoch[action]);
	    }
	    
	  }
	  else{
	    
	  }
	}
      }

      
      
    }

    for(unsigned int i=0;i<grad.size();i++)
      grad[i].stopComputation();
  }

  template class RIFL_abstract< math::blas_real<float> >;
  template class RIFL_abstract< math::blas_real<double> >;
  
};
