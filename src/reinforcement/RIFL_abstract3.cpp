
#include "RIFL_abstract3.h"
#include "NNGradDescent.h"

#include "SGD_recurrent_nnetwork.h"
#include "rLBFGS_recurrent_nnetwork_softmax_actions.h"

#include <assert.h>
#include <list>
#include <functional>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include <map>


#include <unistd.h>

#ifdef _GLIBCXX_DEBUG

#ifndef _WIN32

#undef __STRICT_ANSI__
#include <float.h>
#include <fenv.h>

#endif

#endif


#ifdef WINOS
#include <windows.h>
#endif



namespace whiteice
{

  template <typename T>
  RIFL_abstract3<T>::RIFL_abstract3(const unsigned int numActions,
				    const unsigned int numStates)
  {
    {
      char buffer[128];
      snprintf(buffer, 128, "RIFL_abstract3 CTOR called (%d, %d)",
	       numActions, numStates);
      whiteice::logging.info(buffer);
    }
    
    // initializes parameters
    {
      gamma = T(0.99); // was: 0.95
      epsilon = T(0.80);
      
      learningMode = true;
      hasModel = 0;
      
      this->numActions        = numActions;
      this->numStates         = numStates;
    }

    
    // initializes neural network architecture and weights randomly
    // neural network is deep 6-layer residual neural network
    {
      // wide neural network..
      std::vector<unsigned int> arch;
      arch.push_back(numStates + RECURRENT_DIMENSIONS);
      arch.push_back(50);
      arch.push_back(50);
      arch.push_back(50);
      arch.push_back(50);
      arch.push_back(50);
      arch.push_back(numActions + RECURRENT_DIMENSIONS);

      whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);
      nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh10); // was: tanh

      nn.setResidual(true);
      
      {
	std::lock_guard<std::mutex> lock(model_mutex);
	
	nn.randomize(2, T(0.50f)); // was: 0.10
	model.importNetwork(nn);

	math::vertex<T> w;
	nn.exportdata(w);
	w.zero();
	nn.importdata(w);
	lagged_Q.importNetwork(nn);
	
	// creates empty preprocessing
	preprocess.createCluster("input-state", numStates);
	preprocess.createCluster("output-action-qs", numActions);
	preprocess.createCluster("episode-ranges", 2);
      }
    }

    thread_is_running = 0;
    rifl_thread = nullptr;
    
    whiteice::logging.info("RIFL_abstract3 CTOR finished");
  }

  
  template <typename T>
  RIFL_abstract3<T>::RIFL_abstract3(const unsigned int numActions,
				    const unsigned int numStates,
				    std::vector<unsigned int> arch)
  {
    {
      char buffer[128];
      snprintf(buffer, 128, "RIFL_abstract3 CTOR called (%d, %d)",
	       numActions, numStates);
      whiteice::logging.info(buffer);
    }
    
    // initializes parameters
    {
      gamma = T(0.99); // was: 0.95
      epsilon = T(0.80);

      learningMode = true;
      hasModel = 0;
      
      this->numActions        = numActions;
      this->numStates         = numStates;

      if(arch.size() < 2)
	arch.resize(2);

      arch[0] = numStates + RECURRENT_DIMENSIONS;
      arch[arch.size()-1] = numActions + RECURRENT_DIMENSIONS;
    }

    
    // initializes neural network architecture and weights randomly
    // neural network is deep 6-layer residual neural network
    {

      whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
      // whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::sigmoid);
      nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh10); // was: tanh

      nn.setResidual(true);
      
      {
	std::lock_guard<std::mutex> lock(model_mutex);
	
	nn.randomize(2, T(0.50f)); // was: 0.10
	model.importNetwork(nn);

	math::vertex<T> w;
	nn.exportdata(w);
	w.zero();
	nn.importdata(w);
	lagged_Q.importNetwork(nn);

	// creates empty preprocessing
	preprocess.createCluster("input-state", numStates);
	preprocess.createCluster("output-action-qs", numActions);
	preprocess.createCluster("episode-ranges", 2);
      }
    }

    thread_is_running = 0;
    rifl_thread = nullptr;
    
    whiteice::logging.info("RIFL_abstract3 CTOR finished");
  }
  
  
  template <typename T>
  RIFL_abstract3<T>::~RIFL_abstract3() 
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
  bool RIFL_abstract3<T>::start()
  {
    if(thread_is_running != 0) return false;

    std::lock_guard<std::mutex> lock(thread_mutex);

    if(thread_is_running != 0) return false;

    try{
      thread_is_running++;
      rifl_thread = new std::thread(std::bind(&RIFL_abstract3<T>::loop, this));
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
  bool RIFL_abstract3<T>::stop()
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
  bool RIFL_abstract3<T>::isRunning() const
  {
    return (thread_is_running > 0);
  }


  // epsilon E [0,1] percentage of actions are chosen according to model
  //                 1-e percentage of actions are random (exploration)
  template <typename T>
  bool RIFL_abstract3<T>::setEpsilon(T epsilon) 
  {
    if(epsilon < T(0.0) || epsilon > T(1.0)) return false;
    this->epsilon = epsilon;
    return true;
  }
  

  template <typename T>
  T RIFL_abstract3<T>::getEpsilon() const 
  {
    return epsilon;
  }


  template <typename T>
  void RIFL_abstract3<T>::setLearningMode(bool learn) 
  {
    learningMode = learn;
  }

  template <typename T>
  bool RIFL_abstract3<T>::getLearningMode() const 
  {
    return learningMode;
  }


  template <typename T>
  void RIFL_abstract3<T>::setHasModel(unsigned int hasModel) 
  {
    this->hasModel = hasModel;
  }

  template <typename T>
  unsigned int RIFL_abstract3<T>::getHasModel() 
  {
    return hasModel;
  }

  
  // saves learnt Reinforcement Learning Model to file
  template <typename T>
  bool RIFL_abstract3<T>::save(const std::string& filename) const
  {    

    {
      std::lock_guard<std::mutex> lock(model_mutex);
      
      char buffer[256];

      snprintf(buffer, 256, "%s-model", filename.c_str());

      if(model.save(buffer) == false) return false;

      snprintf(buffer, 256, "%s-lagged-model", filename.c_str());

      if(lagged_Q.save(buffer) == false) return false;

      snprintf(buffer, 256, "%s-preprocess", filename.c_str());

      if(preprocess.save(buffer) == false) return false;
    }

    return true;
  }
  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract3<T>::load(const std::string& filename)
  {
    {
      std::lock_guard<std::mutex> lock(model_mutex);
      
      char buffer[256];
	  
      snprintf(buffer, 256, "%s-model", filename.c_str());

      if(model.load(buffer) == false) return false;

      snprintf(buffer, 256, "%s-lagged-model", filename.c_str());

      if(lagged_Q.load(buffer) == false) return false;

      snprintf(buffer, 256, "%s-preprocess", filename.c_str());

      if(preprocess.load(buffer) == false) return false;
    }

    return true;
  }


  // helper function, returns minimum value in v
  template <typename T>
  unsigned int RIFL_abstract3<T>::min(const std::vector<unsigned int>& vec)
    const 
  {
    if(vec.size() <= 0) return 0;
    unsigned int min = vec[0];
    for(const auto& v : vec)
      if(v < min) min = v;

    return min;
  }


  template <typename T>
  unsigned int RIFL_abstract3<T>::prob_action_select(std::vector<T> v) const
  {
    T psum = T(0.0);
    
    for(unsigned int i=0;i<v.size();i++){
      psum += v[i];
    }
    
    for(unsigned int i=0;i<v.size();i++){
      v[i] /= psum;
    }
    
    psum = T(0.0f);
    for(unsigned int i=0;i<v.size();i++){
      auto more = v[i];
      v[i] += psum;
      psum += more;
    }
    
    T r = rng.uniformf();
    
    unsigned int index = 0;
    
    while(r > v[index]){
      index++;
      if(index >= v.size()){
	index = v.size()-1;
	break;
      }
    }

    std::vector<unsigned int> same_values;
  
    for(unsigned int i=0;i<v.size();i++){
      if(v[index] == v[i]) same_values.push_back(i);
    }

    if(same_values.size() > 1){
      unsigned int r = rng.rand() % same_values.size();
      return same_values[r];
    }
    else{
      return index;
    }
  }
  

  template <typename T>
  unsigned int RIFL_abstract3<T>::prob_action_select(whiteice::math::vertex<T> v) const
  {
    T psum = T(0.0);
    
    for(unsigned int i=0;i<v.size();i++){
      psum += v[i];
    }
    
    for(unsigned int i=0;i<v.size();i++){
      v[i] /= psum;
    }
    
    psum = T(0.0f);
    for(unsigned int i=0;i<v.size();i++){
      auto more = v[i];
      v[i] += psum;
      psum += more;
    }
    
    T r = rng.uniformf();
    
    unsigned int index = 0;
    
    while(r > v[index]){
      index++;
      if(index >= v.size()){
	index = v.size()-1;
	break;
      }
    }

    
    std::vector<unsigned int> same_values;
  
    for(unsigned int i=0;i<v.size();i++){
      if(v[index] == v[i]) same_values.push_back(i);
    }

    if(same_values.size() > 1){
      unsigned int r = rng.rand() % same_values.size();
      return same_values[r];
    }
    else{
      return index;
    }
    
  }

  
  template <typename T>
  void RIFL_abstract3<T>::loop()
  {
#ifdef _GLIBCXX_DEBUG  
#ifndef _WIN32    
    {
      // enables FPU exceptions
      feenableexcept(FE_INVALID | FE_DIVBYZERO);
    }
#endif
#endif
    
    // sets optimizer thread priority to minimum background thread
    
    {
      sched_param sch_params;
      int policy = SCHED_FIFO; // SCHED_RR
      
      pthread_getschedparam(pthread_self(), &policy, &sch_params);
      
#ifdef linux
      policy = SCHED_IDLE; // in linux we can set idle priority
#endif
      sch_params.sched_priority = sched_get_priority_min(policy);
      
      if(pthread_setschedparam(pthread_self(),
			       policy, &sch_params) != 0){
      }
      
#ifdef WINOS
      SetThreadPriority(GetCurrentThread(),
			THREAD_PRIORITY_IDLE);
#endif
      
    }
    

    
    std::mutex database_mutex;

    bool endFlag = false;

    std::vector< rifl_datapoint<T> > episode, full_episode;
    std::vector< std::vector< rifl_datapoint<T> > > episodes;
    //std::multimap< T, std::vector< rifl_datapoint<T> > > episodes;
    
    FILE* episodesFile = fopen("episodes-result.txt", "w");
    
    unsigned long episodes_counter = 0;
    unsigned long full_episodes_counter = 0;

    //
    //// gradient descent learning class
    //whiteice::math::NNGradDescent<T> grad;
    //
    //grad.setOverfit(false);
    //grad.setUseMinibatch(true);
    //grad.setSGD(lrate);
    //grad.setRegularizer(T(0.001f)); // enable regularizer against large values
    //

    // whiteice::SGD_recurrent_nnetwork<T>* grad = nullptr;
    whiteice::rLBFGS_recurrent_nnetwork_softmax_actions<T>* grad = nullptr;

    const T tau = T(0.01); // lagged network update weights
    unsigned int tau_counter = 0;
    const unsigned int TAU_DELAY_BETWEEN_SYNC = 100;

    const T lrate_orig = T(1e-6); // was: 1e-4, WAS: 1e-6, was: 1e-2
    const T lrate = T(1e-6); // was: 1e-4, WAS: 1e-6, was: 1e-2


    whiteice::math::vertex<T> recurrent_data;
    recurrent_data.resize(RECURRENT_DIMENSIONS);
    recurrent_data.zero();

    // dataset for learning class
    whiteice::dataset<T> data;

    // used to calculate dataset in background for gradient descent..
    whiteice::CreateRIFL3dataset<T>* dataset_thread = nullptr;

    unsigned int epoch = 0;
    int old_grad_iterations = 0;

    const unsigned int MAX_EPISODE_LENGTH = 10; // was: 25 for episode length in learning
    const unsigned int EPISODES_BATCHSIZE = 512/MAX_EPISODE_LENGTH; // was: 500/10 = 50
    const unsigned int ITERATIONS = 1; // was 250, WAS: 1 // iterations in gradient descent
    const unsigned int MINIMUM_EPISODE_SIZE = 5000/MAX_EPISODE_LENGTH; // episodes to start learning
    const unsigned int EPISODES_MAX = 1000000/MAX_EPISODE_LENGTH; // max episodes stored (10^6 samples)

    //bool first_time = true;
    whiteice::math::vertex<T> state;
    state.resize(numStates);
    state.zero();

    
    while(thread_is_running > 0){

      // 1. gets current state
      {
	auto oldstate = state;

	if(getState(state) == false)
	  state = oldstate;
      }

      // 2. activates neural networks to get utility values for each command
      std::vector<T> U;
      
      {	
	U.resize(numActions);
	
	whiteice::math::vertex<T> u;

	{
	  whiteice::math::vertex<T> input, output;
	  
	  input.resize(numStates + RECURRENT_DIMENSIONS);
	  output.resize(numActions);
	  
	  {
	    std::lock_guard<std::mutex> lock(model_mutex);
	    
	    preprocess.preprocess(0, state);
	    
	    input.write_subvertex(state, 0);
	    input.write_subvertex(recurrent_data, state.size());
	    
	    assert(model.calculate(input, u, 1, 0) == true);
	    assert(u.size() == numActions + RECURRENT_DIMENSIONS);

	    u.subvertex(output, 0, numActions);
	    
	    model.getNetwork().softmax_output(output, 0, output.size());
	    
	    preprocess.invpreprocess(1, output);
	    
	    
	    u.subvertex(recurrent_data, numActions, RECURRENT_DIMENSIONS);
	  }

	  for(unsigned int i=0;i<numActions;i++)
	    U[i] = output[i];
	  
	}
      }
      
      // 3. selects action according to probabilities
      unsigned int action = 0;

      {
#if 1	
	action = prob_action_select(U);
#else
	

	{ // selects the largest value
	  action = 0;
	  T maxv = U[action];
	  
	  for(unsigned int i=1;i<U.size();i++){
	    if(maxv < U[i]){
	      action = i;
	      maxv = U[i];
	    }
	  }
	}
#endif

	// random selection with (1-epsilon) probability
	// show model with epsilon probability
	{
	  T r = rng.uniformf();
	  
	  if(r > epsilon){
	    action = rng.rand() % (numActions);
	  }
	}

	// if we don't have not yet optimized model, then we make random choices
	if(hasModel == 0)
	  action = rng.rand() % (numActions);

#if 0
	{
	  printf("U = ");
	  for(unsigned int i=0;i<U.size();i++){
	    if(action == i) printf("%e* ", U[i].c[0]);
	    else printf("%e  ", U[i].c[0]);
	  }
	  printf("\n");
	}
#endif

      }
      
      whiteice::math::vertex<T> newstate;
      T reinforcement = T(0.0);

      // 4. perform action 
      {
	if(performAction(action, newstate, reinforcement, endFlag) == false){
	  continue;
	}
	
	if(endFlag){
	  recurrent_data.zero(); // zeroes the memory/starting point of the model
	}

	//auto delta = newstate - state;
	//std::cout << "state ||delta||^2 = " << (delta*delta)[0] << std::endl;
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
	datum.action = action;
	datum.lastStep = endFlag;

	//episode.push_back(datum);
	full_episode.push_back(datum);

	//if(datum.lastStep || episode.size() >= MAX_EPISODE_LENGTH){
	if(datum.lastStep){

	  // only reset and calculate full episode statistics
	  // when sequence really ends
	  
	  T total_reward = T(0.0f);
	  
	  for(const auto& e : full_episode)
	    total_reward += e.reinforcement;
	  
	  total_reward /= T(full_episode.size());
	  
	  char buffer[80];
	  
	  snprintf(buffer, 80, "Episode %d avg reward: %f (%d moves) [%d models]",
		   (int)full_episodes_counter, total_reward.c[0], (int)full_episode.size(),
		   hasModel);
	  
	  whiteice::logging.info(buffer);
	  
	  
	  if(episodesFile){
	    fprintf(episodesFile, "%f\n", total_reward.c[0]);
	    fflush(episodesFile);
	  }

	  
	  for(unsigned int j=0;j<full_episode.size();j+=MAX_EPISODE_LENGTH){

	    const unsigned int END =
	      ((j+MAX_EPISODE_LENGTH) > full_episode.size()) ?
	      full_episode.size() : (j+MAX_EPISODE_LENGTH);

	    episode.clear();

	    for(unsigned int i=j;i<END;i++)
	      episode.push_back(full_episode[i]);

	    if(episodes.size() >= EPISODES_MAX){
	      const unsigned int index = rng.rand() % episodes.size();
	      episodes[index] = episode;
	    }
	    else{
	      episodes.push_back(episode);
	    }

	    episodes_counter++;

#if 0
	    std::pair< T, std::vector< rifl_datapoint<T> > > p;
	    p.first = total_reward;
	    p.second = episode;

	    {
	      std::lock_guard<std::mutex> lock(database_mutex);
	      
	      episodes.insert(p);
	      
	      while(episodes.size() > EPISODES_MAX){

		// removes the smallest reward elements (which we have trained)
		episodes.erase(episodes.begin()); 

#if 0
		// removes cases in the middle (keep low and large reward values)
		const unsigned int index = (rng.rand() % (episodes.size()/2)) + episodes.size()/4;

		auto iter = episodes.begin();
		std::advance(iter, index);
		episodes.erase(iter);

#if 0
		if(rng.rand() & 1){
		  episodes.erase(episodes.begin()); // removes the smallest reward elements
		}
		else{
		  auto iter = episodes.end();
		  iter--;
		  episodes.erase(iter); // removes the largest reward elements
		}
#endif
#endif
	      }
	      
	      episodes_counter++;
	    }
#endif
	    
	  }

	  //if(episodes.size() < EPISODES_MAX)
	  //  episodes.push_back(episode);
	  //else{
	  //  episodes[((episodes_counter)%EPISODES_MAX)] = episode;
	  //}

	  //episode.clear();	  
	  
	  episode.clear();
	  full_episode.clear();
	  full_episodes_counter++;
	}
	
      }

      
      // activates batch learning if it is not running
      {
	
	
	if(episodes.size() > MINIMUM_EPISODE_SIZE)
	{
	  whiteice::nnetwork<T> nn;
	  T error;
	  unsigned int iters = 0;

	  
	  if(dataset_thread != nullptr){
	    if(dataset_thread->isCompleted() != true || dataset_thread->isRunning()){
	      continue; // keep running only dataset_thread
	    }
	    else{
	      // dataset_thread computation completed, delete data_thread and start gradient descent
	      dataset_thread->stop();
	      delete dataset_thread;
	      dataset_thread = nullptr;
	      
	      // data.diagnostics(-1, true);

	      std::vector< math::vertex<T> > weights;

	      // fetch NN parameters from model
	      {
		// gradient is not yet running.. preparing nn for launch..
		// (no preprocess)
		
		std::lock_guard<std::mutex> lock(model_mutex);
		
		if(model.exportSamples(nn, weights, 1) == false){
		  assert(0);
		}
		
		assert(weights.size() > 0);
		
		if(nn.importdata(weights[0]) == false){
		  assert(0);
		}

		//nn.setNonlinearity(whiteice::nnetwork<T>::rectifier);
		//nn.setNonlinearity(nn.getLayers()-1,whiteice::nnetwork<T>::rectifier);
	      }
	      
	      //const bool dropout = false;
	      //const bool useInitialNN = true;

	      if(grad){ delete grad; grad = nullptr; }

	      //grad = new class whiteice::SGD_recurrent_nnetwork<T>(nn, data, false);
	      grad = new whiteice::rLBFGS_recurrent_nnetwork_softmax_actions<T>(nn, data, false);
	      
	      //grad.setOverfit(false);
	      //grad.setUseMinibatch(true);
	      //grad.setNormalizeError(false);
	      //grad.setSGD(lrate);
	      
	      if(hasModel >= WARMUP_ITERS){
		// grad.startOptimize(data, nn, 1, ITERATIONS, dropout, useInitialNN);
		
		//grad->setKeepWorse(true);
		//grad->minimize(weights[0], lrate, ITERATIONS);

		grad->setMaxIterations(ITERATIONS);
		grad->minimize(weights[0]);
	      }
	      else{
		// grad.setUseMinibatch(false);
		// grad.setSGD(T(-1.0f)); // disable stochastic gradient descent
		// grad.startOptimize(data, nn, 1, 5, dropout, useInitialNN);
		
		//grad->setKeepWorse(false);
		//grad->minimize(weights[0], lrate_orig, 20);

		grad->setMaxIterations(30);
		grad->minimize(weights[0]);

	      }
	    
	      old_grad_iterations = -1;
	      
	      continue;
	    }
	  }

	  bool grad_is_running = false;

	  if(grad) if(grad->isRunning()) grad_is_running = true;

	  // dataset_thread not running
	  ///if(grad.isRunning() == false){
	  if(grad_is_running == false){
	    if(grad){
	      // gradient descent has completed, fetch results and start dataset_thread again here
	      
	      // we do not have proper dataset/model yet so we fetch params
	      //grad.getSolution(nn);

	      std::vector< whiteice::math::vertex<T> > w;
	      // std::vector< whiteice::math::vertex<T> > lagged_w;

	      {
		std::lock_guard<std::mutex> lock(model_mutex);

		//lagged_Q.exportSamples(nn, lagged_w);
		model.exportSamples(nn, w);
	      }
	      
	      assert(w.size() >= 1);
	      grad->getSolution(w[0], error, iters);

	      // lagged_w[0] = tau*w[0] + (T(1.0f)-tau)*lagged_w[0];
	      
	      char buffer[128];
	      double tmp = 0.0;
	      whiteice::math::convert(tmp, error);
	      snprintf(buffer, 128,
		       "RIFL_abstract3: new optimizer Q-model (%f error, %d iters, epoch %d)",
		       tmp, iters, epoch);
	      whiteice::logging.info(buffer);

	      {
		std::lock_guard<std::mutex> lock(model_mutex);

		nn.importdata(w[0]);

		// MODIFIED TO KEEP LAGGED_Q IN TIGHT SYNC..
		if(tau_counter >= TAU_DELAY_BETWEEN_SYNC){
		  lagged_Q.importNetwork(nn); // update lagged_Q between every N steps
		  whiteice::logging.info("Update lagged Q network");
		  tau_counter = 0;
		}
		else tau_counter++;

		
		model.importNetwork(nn);
		
		nn.diagnosticsInfo();
		
		data.clearData(0);
		data.clearData(1);
		data.clearData(2);
		
		preprocess = data;
	      }
	      
	      whiteice::logging.info("RIFL_abstract3: new model imported");

	      delete grad;
	      grad = nullptr;
	     
	      hasModel++;
	      epoch++;
	      //first_time = false;
	    }
	    
	    // start dataset_thread
	    
	    data.clear();
	    data.createCluster("input-state", numStates);
	    data.createCluster("output-action-q", numActions);
	    data.createCluster("episode-ranges", 2);

	    dataset_thread = new CreateRIFL3dataset<T>(*this,
						       episodes,
						       database_mutex,
						       model_mutex,
						       epoch,
						       data);
	    dataset_thread->start(EPISODES_BATCHSIZE);
	    
	    whiteice::logging.info("RIFL_abstract3: new dataset_thread started");
	    
	    //first_time = false;

	    continue;
	  }
	  else{ // grad.isRunning() == true, report progress

	    if(grad == nullptr) continue;

	    if(grad->getSolutionStatistics(error, iters)){
	      if(((signed int)iters) > old_grad_iterations){
		char buffer[80];
		
		snprintf(buffer, 80,
			 "RIFL_abstract3: epoch %d optimizer %d iters. error: %f hasmodel %d",
			 epoch, iters, error.c[0], hasModel);
		
		whiteice::logging.info(buffer);
		
		old_grad_iterations = (int)iters;
	      }
	    }
	    else{
	      char buffer[80];
	      snprintf(buffer, 80,
		       "RIFL_abstract3: epoch %d grad.getSolution() FAILED",
		       epoch);
	      
	      whiteice::logging.error(buffer);
	    }

	    continue;
	  }
	  
	}
	
      }
      
    }

    if(episodesFile){
      fclose(episodesFile);
      episodesFile = NULL;
    }
    
    if(dataset_thread){
      dataset_thread->stop();
      delete dataset_thread;
      dataset_thread = nullptr;
    }

    if(grad){
      grad->stopComputation();
      delete grad;
      grad = nullptr;
    }
    
  }

  template class RIFL_abstract3< math::blas_real<float> >;
  template class RIFL_abstract3< math::blas_real<double> >;
  
};
