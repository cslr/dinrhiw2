/*
 * Tests solution of iterated prisoner's dilemma optimization using neural networks and evolution.
 *
 */

#include <stdio.h>
#include <unistd.h>

#include <iostream>

#include "IPD_ES.h"
#include "dinrhiw.h"


int main(int argc, char** argv)
{
  const unsigned int TURNS = 200; // was: 200
  const unsigned int HISTORY_LENGTH = 25;
  
  unsigned int NUM_PLAYERS = 1; // 100 players

  if(argc > 1) NUM_PLAYERS = atoi(argv[1]);
  if(NUM_PLAYERS <= 0) NUM_PLAYERS = 1;

  bool pop_evolution = true;

  if(argc > 2){
    if(strcmp(argv[2], "-noevo") == 0)
      pop_evolution = false;
    else{
      printf("Bad parameters\n");
      return -1;
    }
  }

  whiteice::IPD_ES es(TURNS, HISTORY_LENGTH);

  printf("Iterated Prisoner's Dilemma stategy optimization using evolution strategies using neural networks\n");
  printf("Number of players: %d\n", NUM_PLAYERS);
  printf("Number of parameters per player: %d\n", es.PARAMETER_DIMENSIONS());

  fflush(stdout);

  
  es.setEvolution(pop_evolution);
  
  es.startOptimize(NUM_PLAYERS); // 100 players

  int old_iterations = -1;

  whiteice::linear_ETA<> eta;

  eta.start(0, 1000);
  
  while(true){
    sleep(1);
    
    unsigned int iterations = 0;
    unsigned int best_index = 0;
    auto best_reward = whiteice::math::blas_real<float>(0.0f);
    auto reference_reward = whiteice::math::blas_real<float>(0.0f);
    
    auto mean_r = es.getPopulationMeanReward(iterations, best_index, best_reward, reference_reward);

    if(mean_r < 0.0) continue;

    if(iterations < 1000)
    eta.update(iterations);

    if(((int)iterations) > old_iterations){
      std::cout << "Iterations: " << iterations << " = " << mean_r
		<< " (best: " << best_reward
		<< " index: " << best_index
		<< ", against generous tit-for-tat reference: " << reference_reward << ")"
		<< " ETA to 1000: " << eta.estimate()/60.0f << " minutes."
		<< std::endl;
      fflush(stdout);

      old_iterations = (int)iterations;
    }
  }

  es.stopOptimize();
}
