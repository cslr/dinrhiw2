
#include "EvolutionStrategies.h" // ES
#include <functional>
#include <omp.h>

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
    {
      std::lock_guard<std::mutex> lock(population_mutex);
      
      iterations = 0;
    }

    omp_set_max_active_levels(2);

    //std::cout << "es_loop() started" << std::endl;
    //fflush(stdout);

    T previous_R_mean = T(0.0f);
    
    while(running){
      
      population_mutex.lock();
      
      const auto pop = population;
      auto pop2 = population;
      auto rew2 = rewards;

      population_mutex.unlock();

      const unsigned int K = pop2[0].size(); // was: 33, 100 searched samples around current solution

      T cur_sigma = T(0.0f);

      T R_mean = T(0.0f);
      T R_std  = T(0.0f);
      const T R_epsilon = T(1e-3);
      
      for(unsigned int n=0;n<rew2.size();n++){
	R_mean += rew2[n];
	R_std  += rew2[n]*rew2[n];
      }
      
      R_mean /= rew2.size();
      R_std  /= rew2.size();
      
      R_std -= R_mean*R_mean;
      R_std = whiteice::math::sqrt(whiteice::math::abs(R_std));

      if(previous_R_mean > T(0.0f)){
	if(R_mean - previous_R_mean < T(0.0f)){
	  sigma = sigma*(T(1.0f)-c_lrate);
	}
	else{
	  sigma = sigma*(T(1.0f)+c_lrate);
	}
      }

      previous_R_mean = R_mean;

      if(sigma <= T(1e-4f))
	sigma = T(1e-4f);
      

      //std::cout << "sigma = " << sigma << std::endl;
      //fflush(stdout);


#pragma omp parallel for
      for(unsigned int n=0;n<pop.size();n++){
	
	const auto& x = pop[n];

	std::multimap<T, math::vertex<T> > reward_noise;

#pragma omp parallel for
	for(unsigned int k=0;k<(K/2);k++){
	  math::vertex<T> noise;
	  noise.resize(POPULATION_DIMENSIONS);
	  rng.normal(noise);

	  noise *= (sigma);

	  T reward = T(0.0f);

	  const auto y = x + noise;

	  if(estimateReward(y, pop, reward)){
	    if(reward >= T(0.0f)){
	      reward_noise.insert(std::pair<T, math::vertex<T> >(reward, noise));
	      // reward_noise.insert(std::pair<T, math::vertex<T> >((reward-r_mean)/(r_std+r_epsilon), noise));
	      //noise2[n] = noise;
	      //rew2[n] = reward;
	    }
	  }


	  const auto y2 = x - noise;

	  if(estimateReward(y2, pop, reward)){
	    if(reward >= T(0.0f)){
	      reward_noise.insert(std::pair<T, math::vertex<T> >(reward, -noise));
	      // reward_noise.insert(std::pair<T, math::vertex<T> >((reward-r_mean)/(r_std+r_epsilon), noise));
	      //noise2[n] = noise;
	      //rew2[n] = reward;
	    }
	  }
	}

	std::vector<T> rew3;

	for(auto& p : reward_noise){
	  rew3.push_back(p.first);
	}

	T r_mean = T(0.0f);
	T r_std  = T(0.0f);
	const T r_epsilon = T(1e-3);
	
	for(unsigned int n=0;n<rew3.size();n++){
	  r_mean += rew3[n];
	  r_std  += rew3[n]*rew3[n];
	}
	
	r_mean /= rew3.size();
	r_std  /= rew3.size();
	
	r_std -= r_mean*r_mean;
	r_std = whiteice::math::sqrt(whiteice::math::abs(r_std));
	
	
	if(reward_noise.size()){
	  math::vertex<T> grad;
	  grad.resize(POPULATION_DIMENSIONS);
	  grad.zero();
	  
	  for(auto& p : reward_noise){
	    grad += ((p.first-r_mean)/((r_std+r_epsilon)*T((float)reward_noise.size())))*p.second/sigma;
	  }
	  
	  const auto new_x = x + lrate*grad; // increase reward/fitness
	  
	  pop2[n] = new_x;
	}
      }


#pragma omp parallel for
      for(unsigned int n=0;n<pop2.size();n++){
	const auto& new_x = pop2[n];
	
	T reward = T(0.0f);
	
	if(estimateReward(new_x, pop2, reward)){
	  //if(reward >= T(0.0f) && rew3[n] < reward && (rng.uniformf() < 0.95f)){ // 95% chance of updating the parameters
	  if(reward >= T(0.0f)){
	    pop2[n] = new_x;
	    rew2[n] = reward;
	  }
	}
	
      }

      // keep (1-p)% of the top reward population and drop p% worst results which are replaced by top p% solutions
      if(pop2.size() >= 3 && populationEvolve){	
	std::multimap<T, math::vertex<T> > evopop;

	for(unsigned int i=0;i<pop2.size();i++){
	  evopop.insert(std::pair<T, math::vertex<T> >(rew2[i], pop2[i]));
	}

	unsigned int REPLACE = (pop2.size()*evo_rate.c[0]);

	if(REPLACE <= 0) REPLACE = 1;

	for(unsigned int r=0;r<REPLACE;r++){ // removes REPLACE worst ones from the solution
	  evopop.erase(evopop.begin());
	}

	
	auto best_iter  = evopop.rbegin();
	std::multimap<T, math::vertex<T> > added;

	for(unsigned int r=0;r<REPLACE;r++){
	  added.insert(std::pair<T, math::vertex<T> >(best_iter->first, best_iter->second));
	  best_iter++;
	}

	for(const auto& it : added){
	  // std::cout << "add: " << it->first << std::endl;
	  evopop.insert(std::pair<T, math::vertex<T> >(it.first, it.second));
	  // evopop.insert(*it);
	}


	unsigned int counter = 0;

	for(auto& p : evopop){
	  rew2[counter] = p.first;
	  pop2[counter] = p.second;

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
