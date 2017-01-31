
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
      temperature = T(1.0);
      gamma = T(0.8);
      lrate = T(0.01);

      this->numActions = numActions;
      this->numStates  = numStates;
    }

    
    // initializes neural network architecture and weights randomly
    {

      std::vector<unsigned int> arch;
      arch.push_back(numStates);
      arch.push_back(numStates*100);
      arch.push_back(numStates*100);
      arch.push_back(numStates*100);
      arch.push_back(numActions);

      whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::halfLinear);
      nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::pureLinear);
      nn.randomize();

      model.importNetwork(nn);
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

  
  // saves learnt Reinforcement Learning Model to file
  template <typename T>
  bool RIFL_abstract<T>::save(const std::string& filename) const
  {
    return false;
  }
  
  // loads learnt Reinforcement Learning Model from file
  template <typename T>
  bool RIFL_abstract<T>::load(const std::string& filename)
  {
    return false;
  }
  

  template <typename T>
  void RIFL_abstract<T>::loop()
  {
    std::vector< rifl_datapoint<T> > database;
    whiteice::dataset<T> data;
    whiteice::math::NNGradDescent<T> grad;
    unsigned int epoch = 0;
    

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
	whiteice::math::vertex<T> u;
	whiteice::math::matrix<T> e;

	if(model.calculate(state, u, e, 1, 0) == true){
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
	
	for(auto& ui : U){
	  printf("%f ", ui.c[0]);
	}
	printf("\n");
      }

      // 3. selects action according to probabilities
      unsigned int action = 0;

      {
	T epsilon = rng.uniform();

	if(epsilon < T(0.33)){ // 33% selects the largest value
	  T maxv = U[action];
	  
	  for(unsigned int i=0;i<U.size();i++){
	    if(maxv < U[i]){
	      action = i;
	      maxv = U[i];
	    }
	  }
	}
	else{ // 66% select action randomly
	  action = rng.rand() % U.size();
	}
      }
      
      if(0){
	std::vector<T> p;

	T psum = T(0.0);

	for(unsigned int i=0;i<U.size();i++){
	  T expm = whiteice::math::exp(U[i]/temperature);
	  psum += expm;
	  p.push_back(psum);
	}

	for(unsigned int i=0;i<p.size();i++){
	  p[i] = p[i] / psum;
	}

	T v = rng.uniform();
	unsigned int index = 0;

	while(p[index] < v)
	  index++;

	action = index;
      }


      whiteice::math::vertex<T> newstate;
      T reinforcement = T(0.0);

      // 4. perform action 
      {
	if(performAction(action, newstate, reinforcement) == false){
	  continue;
	}
      }



      // 6. updates database
      {
	struct rifl_datapoint<T> data;

	data.state = state;
	data.newstate = newstate;
	data.reinforcement = reinforcement;
	data.action = action;

	database.push_back(data);

	printf("DATABASE SIZE: %d\n", (int)database.size());

	while(database.size() >= 10000){
	  const unsigned int index = rng.rand() % database.size();

	  database[index] = database[database.size()-1];
	  database.erase(std::prev(database.end()));
	}
      }


      // activates minibatch learning if it is not running
      if(database.size() > 1000)
      {
	if(grad.isRunning() == false){
	  printf("*************************************************************\n");
	  
	  whiteice::nnetwork<T> nn;
	  T error;
	  unsigned int iters;

	  if(grad.getSolution(nn, error, iters) == false){
	    std::vector< math::vertex<T> > weights;
	    
	    if(model.exportSamples(nn, weights, 1) == false){
	      assert(0);
	    }
	    
	    assert(weights.size() > 0);
	    
	    if(nn.importdata(weights[0]) == false){
	      assert(0);
	    }
	  }
	  else{
	    model.importNetwork(nn);
	  }

	  epoch++;
	    
	  const unsigned int BATCHSIZE = database.size()/2;

	  data.clear();
	  data.createCluster("input-state", numStates);
	  data.createCluster("output-action", numActions);
	  
	  for(unsigned int i=0;i<BATCHSIZE;i++){
	    const unsigned int index = rng.rand() % database.size();

	    whiteice::math::vertex<T> in = database[index].state;
	    whiteice::math::vertex<T> out(numActions);
	    out.zero();
	    
	    // calculates updated utility value
	    
	    whiteice::math::vertex<T> u;
	    T u_value = T(0.0);
	    
	    {
	      nn.calculate(database[index].state, u);

	      u_value = u[database[index].action];
	    }

	    
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
	    
	    out[database[index].action] = unew_value - u_value;

	    data.add(0, in);
	    data.add(1, out);
	  }

	  grad.startOptimize(data, nn, 2, 25);
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

#if 0
      if(0){
	whiteice::nnetwork<T> nn;
	std::vector< math::vertex<T> > weights;
	
	{
	  if(model.exportSamples(nn, weights, 1) == false){
	    assert(0);
	  }

	  assert(weights.size() > 0);

	  if(nn.importdata(weights[0]) == false){
	    assert(0);
	  }
	}

	nn.input() = state;
	nn.calculate(true);

	whiteice::math::vertex<T> grad;
	whiteice::math::vertex<T> error;
	error.resize(numActions);
	error.zero();
	error[action] = (unew - nn.output()[0]);

	if(nn.gradient(error, grad) == false)
	  assert(0);

	weights[0] -= lrate*grad;

	nn.importdata(weights[0]);

	if(model.importNetwork(nn) == false){
	  assert(0);
	}
      }
#endif 
      
    }
    
    
  }
    


  template class RIFL_abstract< math::blas_real<float> >;
  template class RIFL_abstract< math::blas_real<double> >;
  
};
