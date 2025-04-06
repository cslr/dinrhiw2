/*
 * Simple Evolution Strategies with gaussian sampling based search for better solutions
 *
 * Tomas Ukkonen 2025
 */

#ifndef whiteice_EvolutionStrategies_h
#define whiteice_EvolutionStrategies_h

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <map>


#include "dinrhiw_blas.h"
#include "vertex.h"
#include "RNG.h"

namespace whiteice
{

  template <typename T = math::blas_real<float> >
  class EvolutionStrategies
  {
  public:
    EvolutionStrategies();
    ~EvolutionStrategies();

    bool startOptimize(const unsigned int N_POPULATION = 100);
    
    bool stopOptimize();

    // set evolution to replace N percent of the worst population with N percent of the best
    bool getEvolution() const { return populationEvolve; }
    
    void setEvolution(const bool e){ populationEvolve = e; }


    T getPopulationMeanReward(unsigned int& iterations, unsigned int& best_index, T& best_reward, T& mean_solution_against_reference) const;

  protected:

    virtual bool estimateReward(const math::vertex<T>& x,
				const std::vector< math::vertex<T> >& population,
				T& reward) const = 0; // reward must be positive number >= 0

    virtual bool estimateMeanRewardReference(const math::vertex<T>& x, // mean reward against reference good standard solution
					     T& reward) const = 0;

    virtual unsigned int PARAMETER_DIMENSIONS() const = 0; // number of parameters in the model

    RNG<T> rng;

  private:
    
    std::thread* es_thread = nullptr;
    mutable std::mutex thread_mutex;
    std::atomic<bool> running = false;
    void es_loop();

    unsigned int POPULATION_DIMENSIONS;

    mutable std::mutex population_mutex; // population and rewards mutex
    std::vector< math::vertex<T> > population;
    std::vector< T > rewards;
    unsigned int iterations = 0;

    T sigma = T(0.1); // noise search term..
    const T lrate = T(0.1);

    const T evo_rate = T(0.10); // replace worst 10% population with top 10% population
    bool populationEvolve = true;
  };


  extern template class EvolutionStrategies< math::blas_real<float> >;
  extern template class EvolutionStrategies< math::blas_real<double> >;
};

#endif

