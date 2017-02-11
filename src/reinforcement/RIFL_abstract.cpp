
#include "RIFL_abstract.h"
#include "NNGradDescent.h"

#include <assert.h>
#include <list>

namespace whiteice
{

  template <typename T>
  RIFL_abstract<T>::RIFL_abstract(const unsigned int numActions,
				  const unsigned int numStates,
				  const unsigned int dimActionFeatures)
  {
    // initializes parameters
    {
      gamma = T(0.80);
      epsilon = T(0.66);

      learningMode = true;
      hasModel = false;
      
      this->numActions        = numActions;
      this->numStates         = numStates;
      this->dimActionFeatures = dimActionFeatures;
    }

    
    // initializes neural network architecture and weights randomly
    {
      std::vector<unsigned int> arch;
      arch.push_back(numStates + dimActionFeatures);
      // arch.push_back(numStates*100);
      // arch.push_back(numStates*100);

      unsigned int L1 = (numStates + dimActionFeatures)/2;
      if(L1 < 20) L1 = 20;
      
      arch.push_back(L1);

      unsigned int L2 = sqrt(numStates + dimActionFeatures);
      if(L2 < 20) L2 = 20;
      
      arch.push_back(L2);
      arch.push_back(1);

      whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);
      nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);

      {
	std::lock_guard<std::mutex> lock(model_mutex);
	
	nn.randomize();
	model.importNetwork(nn);
	
	// creates empty preprocessing
	preprocess.createCluster("input-state", numStates + dimActionFeatures);
	preprocess.createCluster("output-action", 1);
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

    {
      std::lock_guard<std::mutex> lock(model_mutex);
      
      char buffer[256];

      snprintf(buffer, 256, "model-%s", filename.c_str());

      if(model.save(buffer) == false) return false;

      snprintf(buffer, 256, "preprocess-%s", filename.c_str());

      if(preprocess.save(buffer) == false) return false;
    }

    return true;
  }
  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract<T>::load(const std::string& filename)
  {
    {
      std::lock_guard<std::mutex> lock(model_mutex);
      
      char buffer[256];
	  
      snprintf(buffer, 256, "model-%s", filename.c_str());

      if(model.load(buffer) == false) return false;

      snprintf(buffer, 256, "preprocess-%s", filename.c_str());

      if(preprocess.load(buffer) == false) return false;
    }

    return true;
  }


  // helper function, returns minimum value in v
  template <typename T>
  unsigned int RIFL_abstract<T>::min(const std::vector<unsigned int>& vec)
    const throw()
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
    whiteice::dataset<T> data;
    whiteice::math::NNGradDescent<T> grad;

    unsigned int epoch = 0;
    

    const unsigned int DATASIZE = 50000;
    const unsigned int SAMPLESIZE = 100;
    T temperature = T(0.010);

    database.resize(numActions);

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
	U.resize(numActions);

	for(unsigned int i=0;i<numActions;i++){
	  std::lock_guard<std::mutex> lock(model_mutex);
	  
	  whiteice::math::vertex<T> u;
	  whiteice::math::matrix<T> e;
	  
	  whiteice::math::vertex<T> input;
	  whiteice::math::vertex<T> feature(dimActionFeatures);
	  
	  feature.zero();
	  getActionFeature(i, feature);

	  input.resize(numStates + dimActionFeatures);
	  input.zero();
	  input.write_subvertex(state, 0);
	  input.write_subvertex(feature, numStates);

	  preprocess.preprocess(0, input);
	  
	  if(model.calculate(input, u, e, 1, 0) == true){
	    if(u.size() != 1){
	      u.resize(1);
	      u[0] = T(0.0);
	    }
	    else
	      preprocess.invpreprocess(1, u);
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
	  action = rng.rand() % (numActions);
	}

	// if we don't have not yet optimized model, then we make random choices
	if(hasModel == false)
	  action = rng.rand() % (numActions);
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

	unsigned int min_database_size = database[action].size();
	unsigned int total_database_size = 0;

	for(unsigned int i=0;i<database.size();i++){
	  if(min_database_size > database[i].size())
	    min_database_size = (unsigned int)database[i].size();
	  total_database_size += database[i].size();
	}
	
	printf("%d SAMPLES TOTAL. MIN ACTION DATABASE SIZE: %d\n",
	       total_database_size, min_database_size);
      }
      

      // activates batch learning if it is not running
      {
	unsigned int samples = 0;
	bool allHasSamples = true;

	for(unsigned int i=0;i<database.size();i++){
	  samples += database[i].size();
	  if(database[i].size() == 0){
	    allHasSamples = false;
	  }
	}

	// debugging..
	printf("%d SAMPLES. ALL HAS SAMPLES %d\n", samples, (int)allHasSamples);

	
	if(samples >= SAMPLESIZE && allHasSamples)
	{
	  // printf("[%d/%d] EPOCH %d ABOUT TO START ACTION OPTIMIZER *********\n",
	  // action, (int)model.size(), epoch[action]);
	  
	  whiteice::nnetwork<T> nn;
	  T error;
	  unsigned int iters;
	  
	  
	  if(grad.isRunning() == false){
	    
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

	      // we keep previous network to some degree
	      // (interpolation between networks)
	      if(epoch > 0){
		T tau = T(0.3);
		{
		  whiteice::nnetwork<T> nnprev = nn;
		  std::vector< whiteice::math::vertex<T> > prevweights;
		  
		  if(model.exportSamples(nnprev, prevweights)){
		    whiteice::math::vertex<T> newweights;
		    
		    if(nn.exportdata(newweights)){
		      newweights = tau*newweights + (T(1.0)-tau)*prevweights[0];
		      
		      nn.importdata(newweights);
		    }
		  }
		  
		  model.importNetwork(nn);
		}
	      }
	      else{
		model.importNetwork(nn);
	      }

	      
	      {
		data.clearData(0);
		data.clearData(1);
		
		preprocess = data;
	      }

	      epoch++;
	      hasModel = true;
	    }


	    // uses half the samples in database
	    unsigned int BATCHSIZE = 0;
	    
	    {
	      for(unsigned int i=0;i<database.size();i++){
		BATCHSIZE += database[i].size();
	      }

	      BATCHSIZE /= 2;
	    }

	    bool newPreprocess = false;

	    if(data.getNumberOfClusters() != 2){
	      data.clear();
	      data.createCluster("input-state", numStates + dimActionFeatures);
	      data.createCluster("output-action", 1);
	      newPreprocess = true;
	    }
	    else{
	      data.clearData(0);
	      data.clearData(1);
	      newPreprocess = false;
	    }
	    

#pragma omp parallel for schedule(dynamic)
	    for(unsigned int i=0;i<BATCHSIZE;i++){
	      const unsigned int action = rng.rand() % numActions;
	      const unsigned int index = rng.rand() % database[action].size();
	      
	      whiteice::math::vertex<T> in;
	      whiteice::math::vertex<T> feature;

	      getActionFeature(action, feature);
	      
	      in.resize(numStates + dimActionFeatures);
	      in.zero();
	      in.write_subvertex(database[action][index].state, 0);
	      in.write_subvertex(feature, numStates);
	      
	      whiteice::math::vertex<T> out(1);
	      out.zero();
	      
	      // calculates updated utility value
	      
	      whiteice::math::vertex<T> u;
	      
	      T unew_value = T(0.0);

	      {
		T maxvalue = T(-INFINITY);

		for(unsigned int j=0;j<numActions;j++){
		  std::lock_guard<std::mutex> lock(model_mutex);
		  
		  whiteice::math::vertex<T> u;
		  whiteice::math::matrix<T> e;
		  
		  whiteice::math::vertex<T> input(numStates + dimActionFeatures);
		  whiteice::math::vertex<T> f;
		  
		  input.zero();
		  input.write_subvertex(database[action][index].newstate, 0);

		  getActionFeature(j, f);
		  input.write_subvertex(f, numStates);

		  preprocess.preprocess(0, input);

		  if(model.calculate(input, u, e, 1, 0) == true){
		    if(u.size() != 1){
		      u.resize(1);
		      u[0] = T(0.0);
		    }
		    else
		      preprocess.invpreprocess(1, u);
		  }
		  else{
		    u.resize(1);
		    u[0] = T(0.0);
		  }
		  
		  if(maxvalue < u[0])
		    maxvalue = u[0];
		}

		if(epoch > 0){
		  unew_value =
		    database[action][index].reinforcement + gamma*maxvalue;
		}
		else{ // first iteration always uses raw reinforcement values
		  unew_value =
		    database[action][index].reinforcement;
		}
	      }
	      
	      out[0] = unew_value;

#pragma omp critical
	      {
		data.add(0, in);
		data.add(1, out);
	      }
	      
	    }
	    
	    // add preprocessing to dataset (only at epoch 0)
	    if(newPreprocess){
	      data.preprocess
		(0, whiteice::dataset<T>::dnMeanVarianceNormalization);

#if 0
	      data.preprocess
		(1, whiteice::dataset<T>::dnMeanVarianceNormalization);
#endif
	    }

	    const bool dropout = false;
	    
	    if(grad.startOptimize(data, nn, 2, 250, dropout) == false){
				  
	      printf("STARTING GRAD OPTIMIZATION FAILED\n");
	    }
	    else{
	      printf("STARTED GRAD OPTIMIZER EPOCH %d\n", epoch);
	    }
	    
	  }
	  else{

	    if(grad.getSolution(nn, error, iters)){
	      printf("EPOCH %d OPTIMIZER %d ITERS: ERROR %f HASMODEL: %d\n",
		     epoch, iters, error.c[0], (int)hasModel);
	    }
	    else{
	      printf("EPOCH %d GETSOLUTION() FAILED\n",
		     epoch);
	    }
	    
	    
	  }
	}
      }

      
      
    }


    grad.stopComputation();
    
  }

  template class RIFL_abstract< math::blas_real<float> >;
  template class RIFL_abstract< math::blas_real<double> >;
  
};
