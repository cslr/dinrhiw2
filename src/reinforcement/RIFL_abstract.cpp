
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
      hasModel = 0;
      
      this->numActions        = numActions;
      this->numStates         = numStates;
      this->dimActionFeatures = dimActionFeatures;
    }

    
    // initializes neural network architecture and weights randomly
    {
      // wide neural network..
      std::vector<unsigned int> arch;
      arch.push_back(numStates + dimActionFeatures);
      arch.push_back((numStates + dimActionFeatures)*20);
      arch.push_back((numStates + dimActionFeatures)*20);
      arch.push_back(1);

      whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::tanh);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
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
  void RIFL_abstract<T>::setHasModel(unsigned int hasModel) throw()
  {
    this->hasModel = hasModel;
  }

  template <typename T>
  unsigned int RIFL_abstract<T>::getHasModel() throw()
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

      snprintf(buffer, 256, "%s-model", filename.c_str());

      if(model.save(buffer) == false) return false;

      snprintf(buffer, 256, "%s-preprocess", filename.c_str());

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
	  
      snprintf(buffer, 256, "%s-model", filename.c_str());

      if(model.load(buffer) == false) return false;

      snprintf(buffer, 256, "%s-preprocess", filename.c_str());

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
    std::mutex database_mutex;
    
    whiteice::dataset<T> data;

    // FIXME/DEBUG disable??? (heuristics keep weights at unity..)
    // whiteice::math::NNGradDescent<T> grad(true);
    whiteice::math::NNGradDescent<T> grad(false);

    // used to calculate dataset in background for NNGradDescent..
    whiteice::CreateRIFLdataset<T>* dataset_thread = nullptr;

    unsigned int epoch = 0;
    

    const unsigned int DATASIZE = 10000;
    const unsigned int SAMPLESIZE = 100;
    T temperature = T(0.010);

    // keep 100% of the new network weights (was 30%)
    const T tau = T(1.0); // T tau = T(0.3);

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

#if 0
	{
	  printf("U = ");
	  for(unsigned int i=0;i<U.size();i++){
	    if(action == i) printf("%f* ", U[i].c[0]);
	    else printf("%f  ", U[i].c[0]);
	  }
	  printf("\n");
	}
#endif
	
	// random selection with (1-epsilon) probability
	// show model pich with epsilon probability
	T r = rng.uniform();
	
	if(learningMode == false)
	  r = T(0.0); // always selects the largest value

	if(r > epsilon){
	  action = rng.rand() % (numActions);
	}

	// if we don't have not yet optimized model, then we make random choices
	if(hasModel == 0)
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
	struct rifl_datapoint<T> datum;

	datum.state = state;
	datum.newstate = newstate;
	datum.reinforcement = reinforcement;

	// for synchronizing access to database datastructure
	// (also used by CreateRIFLdataset class/thread)
	std::lock_guard<std::mutex> lock(database_mutex);
	
	if(database[action].size() >= DATASIZE){
	  const unsigned int index = rng.rand() % database[action].size();
	  database[action][index] = datum;
	}
	else{
	  database[action].push_back(datum);
	}
	
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

	char buffer[80];
	snprintf(buffer, 80, "RIFL_abstract: %d samples. all has samples %d",
		 samples, (int)allHasSamples);
	whiteice::logging.info(buffer);

	
	if(samples >= SAMPLESIZE && allHasSamples)
	{
	  whiteice::nnetwork<T> nn;
	  T error;
	  unsigned int iters;

	  
	  // if gradient thread have stopped or is not yet running..
	  if(grad.isRunning() == false){
	    
	    
	    if(grad.getSolutionStatistics(error, iters) == false){
#if 0
	      // gradient is not yet running.. preparing nn for launch..
	      // (no preprocess)
	      std::vector< math::vertex<T> > weights;

	      std::lock_guard<std::mutex> lock(model_mutex);
	      
	      if(model.exportSamples(nn, weights, 1) == false){
		assert(0);
	      }
	      
	      assert(weights.size() > 0);
	      
	      if(nn.importdata(weights[0]) == false){
		assert(0);
	      }
#endif	      
	    }
	    else{
	      // gradient have stopped running

	      if(dataset_thread == nullptr){
		// we do not have proper dataset/model yet so we fetch params
		grad.getSolution(nn);
		
	      
		char buffer[128];
		double tmp = 0.0;
		whiteice::math::convert(tmp, error);
		snprintf(buffer, 128,
			 "RIFL_abstract: new optimized Q-model (%f error, %d iters, epoch %d)",
			 tmp, iters, epoch);
		whiteice::logging.info(buffer);

		std::lock_guard<std::mutex> lock(model_mutex);

		// we keep previous network to some degree
		// (interpolation between networks)
		if(epoch > 0){
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

		  nn.diagnosticsInfo();

		  data.clearData(0);
		  data.clearData(1);

		  preprocess = data;

		  whiteice::logging.info("RIFL_abstract: new model imported");
		}
		else{
		  model.importNetwork(nn);
		  
		  nn.diagnosticsInfo();
		  
		  data.clearData(0);
		  data.clearData(1);
		  
		  preprocess = data;

		  whiteice::logging.info("RIFL_abstract: new model imported");
		}
	      
		epoch++;
		hasModel++;
	      }
	      
	    }


	    if(dataset_thread == nullptr){
	      data.clear();
	      data.createCluster("input-state", numStates + dimActionFeatures);
	      data.createCluster("output-action", 1);
	      
	      dataset_thread = new CreateRIFLdataset<T>(*this,
							database,
							database_mutex,
							epoch,
							data);
	      dataset_thread->start(samples);

	      whiteice::logging.info("RIFL_abstract: new dataset_thread started");
	      
	      continue;
	      
	    }
	    else{
	      if(dataset_thread->isCompleted() != true)
		continue; // we havent computed proper dataset yet..
	    }

	    whiteice::logging.info("RIFL_abstract: new dataset_thread finished");
	    dataset_thread->stop();

	    // fetch NN parameters from model
	    {
	      // gradient is not yet running.. preparing nn for launch..
	      // (no preprocess)
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
	    
	    const bool dropout = false;
	    
	    // if(grad.startOptimize(data, nn, 2, 250, dropout) == false){
	    if(grad.startOptimize(data, nn, 1, 250, dropout) == false){
	      whiteice::logging.error("RIFL_abstract: starting grad optimizer FAILED");
	      assert(0);
	    }
	    else
	      whiteice::logging.info("RIFL_abstract: grad optimizer started");
	    
	    delete dataset_thread;
	    dataset_thread = nullptr;
	  }
	  else{

	    if(grad.getSolutionStatistics(error, iters)){
	      snprintf(buffer, 80,
		       "RIFL_abstract: epoch %d optimizer %d iters. error: %f hasmodel %d",
		       epoch, iters, error.c[0], hasModel);
	      
	      whiteice::logging.info(buffer);
	    }
	    else{
	      snprintf(buffer, 80,
		       "RIFL_abstract: epoch %d grad.getSolution() FAILED",
		       epoch);
	      
	      whiteice::logging.error(buffer);
	    }
	    
	    
	  }
	}
      }
      
    }

    
    grad.stopComputation();
    
    if(dataset_thread){
      dataset_thread->stop();
      delete dataset_thread;
      dataset_thread = nullptr;
    }
    
  }

  template class RIFL_abstract< math::blas_real<float> >;
  template class RIFL_abstract< math::blas_real<double> >;
  
};
