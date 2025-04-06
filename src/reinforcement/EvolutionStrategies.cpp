
#include "EvolutionStrategies.h" // ES
#include <functional>


namespace whiteice
{

  template <typename T>
  EvolutionStrategies<T>::EvolutionStrategies()
  {
    sigma = T(0.1);
    POPULATION_DIMENSIONS = 0;
  }

  
  template <typename T>
  EvolutionStrategies<T>::~EvolutionStrategies()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);

    try{
      if(running){
	running = false;
	if(es_thread) es_thread->join();
      }
      
      if(es_thread) delete es_thread;
    }
    catch(std::exception& e){
      
    }
  }


  template <typename T>
  bool EvolutionStrategies<T>::startOptimize(const unsigned int N)
  {
    if(N <= 0) return false;
        
    std::lock_guard<std::mutex> lock(thread_mutex);

    if(es_thread != nullptr) return false; // already running

    //std::cout << "startOptimize() started" << std::endl;
    //fflush(stdout);

    // generate population of N agents
    {
      std::lock_guard<std::mutex> lock2(population_mutex);

      POPULATION_DIMENSIONS = PARAMETER_DIMENSIONS();

      if(POPULATION_DIMENSIONS <= 0) return false;
      
      population.clear();
      rewards.clear();
      
      for(unsigned int n=0;n<N;n++){
	math::vertex<T> x;
	x.resize(POPULATION_DIMENSIONS);
	rng.normal(x);
	population.push_back(x);
      }

      //std::cout << "startOptimize() init 1" << std::endl;
      //fflush(stdout);

      for(unsigned int n=0;n<N;n++){
	T reward = T(0.0);
	if(this->estimateReward(population[n], population, reward) == false)
	  return false;

	rewards.push_back(reward);
      }

      //std::cout << "startOptimize() init 2" << std::endl;
      //fflush(stdout);
    }
    

    try{
      running = true;
      es_thread = new std::thread(std::bind(&EvolutionStrategies<T>::es_loop, this));

      return true;
    }
    catch(std::exception& e){
      running = false;
      if(es_thread) delete es_thread;
      es_thread = nullptr;
      
      return false;
    }
  }
  

  template <typename T>
  bool EvolutionStrategies<T>::stopOptimize()
  {
    std::lock_guard<std::mutex> lock(thread_mutex);
    
    try{
      if(running){
	running = false;
	if(es_thread) es_thread->join();
      }
      
      if(es_thread){
	delete es_thread;
	es_thread = nullptr;
	return true;
      }
      else{
	return false;
      }
    }
    catch(std::exception& e){
      return false;
    }
  }
  

  template <typename T>
  T EvolutionStrategies<T>::getPopulationMeanReward(unsigned int& iterations_,
						    unsigned int& best_index_,
						    T& best_reward,
						    T& mean_solution_against_reference) const
  {
    std::lock_guard<std::mutex> lock2(population_mutex);

    T sum = T(0.0f);

    if(rewards.size() > 0){
      best_reward = rewards[0];
    }
    else{
      best_reward = T(-1.0f);
      mean_solution_against_reference = T(-1.0f);
      best_index_ = 0;
      iterations_ = 0;

      return T(-1.0f);
    }
    
    unsigned int index = 0, best_index = 0;

    for(auto& r : rewards){
      sum += r;
      if(r > best_reward){
	best_reward = r;
	best_index = index;
      }

      index++;
    }

    if(rewards.size())
      sum /= T(rewards.size());

    iterations_ = this->iterations;

    if(estimateMeanRewardReference(population[best_index], mean_solution_against_reference) == false)
      mean_solution_against_reference = T(-1.0f);

    best_index_ = best_index;

    return sum;
  }

  
  template <typename T>
  void EvolutionStrategies<T>::es_loop()
  {
    const unsigned int K = 33; // was: 100 searched samples around current solution

    {
      std::lock_guard<std::mutex> lock(population_mutex);
      
      iterations = 0;
    }

    //std::cout << "es_loop() started" << std::endl;
    //fflush(stdout);
    
    while(running){
      
      population_mutex.lock();
      
      const auto pop = population;
      auto pop2 = population;
      auto rew2 = rewards;

      T r_mean = T(0.0f);
      T r_std  = T(0.0f);
      const T r_epsilon = T(1e-3);

      for(unsigned int n=0;n<rewards.size();n++){
	r_mean += rewards[n];
	r_std  += rewards[n]*rewards[n];
      }

      r_mean /= rewards.size();
      r_std  /= rewards.size();

      r_std -= r_mean*r_mean;
      r_std = whiteice::math::sqrt(whiteice::math::abs(r_std));

      population_mutex.unlock();

#pragma omp parallel for
      for(unsigned int n=0;n<pop.size();n++){
	const auto& x = pop[n];

	std::multimap<T, math::vertex<T> > reward_noise;
	
	for(unsigned int k=0;k<K;k++){	  
	  math::vertex<T> noise;
	  noise.resize(POPULATION_DIMENSIONS);
	  rng.normal(noise);

	  noise *= (sigma);

	  T reward = T(0.0f);

	  const auto y = x + noise;

	  if(estimateReward(y, pop, reward)){
	    if(reward >= T(0.0f)){
	      // reward_noise.insert(std::pair<T, math::vertex<T> >(reward, noise));
	      reward_noise.insert(std::pair<T, math::vertex<T> >((reward-r_mean)/(r_std+r_epsilon), noise));
	    }
	  }
	}

	if(reward_noise.size()){

	  math::vertex<T> grad;
	  grad.resize(POPULATION_DIMENSIONS);
	  grad.zero();

	  for(auto& p : reward_noise){
	    grad += (p.first/T((float)reward_noise.size()))*p.second/sigma;
	  }

	  const auto new_x = x + lrate*grad; // increase reward/fitness

	  T reward = T(0.0f);

	  if(estimateReward(new_x, pop, reward)){
	    if(reward >= T(0.0f)){
	      pop2[n] = new_x;
	      rew2[n] = reward;
	    }
	  }
	}
	
      }

      // keep (1-p)% of the top reward population and drop p% worst results which are replaced by top p% solutions
      if(pop2.size() >= 2 && populationEvolve){
	std::multimap<T, math::vertex<T> > evopop;

	for(unsigned int i=0;i<pop2.size();i++){
	  evopop.insert(std::pair<T, math::vertex<T> >(rew2[i], pop2[i]));
	}

	unsigned int REPLACE = (pop2.size()*evo_rate.c[0]);

	if(REPLACE <= 0) REPLACE = 1;

	auto worst_iter = evopop.begin();
	std::vector< typename std::multimap<T, math::vertex<T> >::iterator > removed;

	for(unsigned int r=0;r<REPLACE;r++){
	  removed.push_back(worst_iter);
	  worst_iter++;
	}

	for(auto& it : removed){
	  // std::cout << "remove: " << it->first << std::endl;
	  evopop.erase(it); // removes worst one and adds best solution
	}

	auto best_iter  = evopop.rbegin();
	std::vector< typename std::multimap<T, math::vertex<T> >::reverse_iterator > added;

	for(unsigned int r=0;r<REPLACE;r++){
	  added.push_back(best_iter);
	  best_iter++;
	}

	for(auto& it : added){
	  // std::cout << "add: " << it->first << std::endl;
	  evopop.insert(*it);
	}

	unsigned int counter = 0;
	
	for(auto& p : evopop){
	  pop2[counter] = p.second;
	  rew2[counter] = p.first;
	  
	  counter++;
	}
      }

      {
	std::lock_guard<std::mutex> lock(population_mutex);

	iterations++;
	population = pop2;
	rewards = rew2;
      }
      
    }
    
    
  }



  template class EvolutionStrategies< math::blas_real<float> >;
  template class EvolutionStrategies< math::blas_real<double> >;
};
