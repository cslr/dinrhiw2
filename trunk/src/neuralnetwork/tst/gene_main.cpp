

// TODO: write genetic algorithm tests
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "GeneticAlgorithm.h"
#include "GeneticAlgorithm2.h"
#include "test_function2.h" // for testing

#include "GA3.h"
#include "ga3_test_function.h"
#include "nnetwork_function.h"
#include <unistd.h>

using namespace whiteice;

template <typename T>
void show_results(const GeneticAlgorithm<T>* GA) throw();

void show_results(const class GA* GA) throw();



int main(int argc, char ** argv, char **envp)
{
  srand(time(0));
  
  {
    std::vector<unsigned int> arch;
    arch.push_back(10);
    arch.push_back(5);
    arch.push_back(1);
    nnetwork<> nn(arch);
    nn.randomize();
    
    nnetwork_function<> nf(nn);
    GA3<> ga(&nf);
    
    ga.minimize();
    
    while(1){
      whiteice::math::vertex<> s;
      math::blas_real<float> r;

      r = ga.getBestSolution(s);
      
      const unsigned int g = ga.getGenerations();

      std::cout << "Best result (" << g << " generations) : " << r
		<< " param: " << s << std::endl;
      sleep(1);
    }
  }

  {
    ga3_test_function<> gtf;
    GA3<> ga(&gtf);
    
    ga.minimize();

    while(1){
      whiteice::math::vertex<> s;
      math::blas_real<float> r;

      r = ga.getBestSolution(s);
      
      const unsigned int g = ga.getGenerations();

      std::cout << "Best result (" << g << " generations) : " << r
		<< " param: " << s << std::endl;
      sleep(1);
    }
  }
  
  if(argc < 2) return 0;
  
  {
    GeneticAlgorithm<int>* GA;
    test_function2 of;
    
    GA = new GeneticAlgorithm<int>(of);
    GA->getCrossover() = 0.80;
    GA->verbosity(true);
    
    // (a,b) = (iterations, population size)
    GA->minimize(atoi(argv[1]), 100); // something small and simple
    
    show_results(GA);   
    
    delete GA;
  }
  
  std::cout << "-----------------------------------" << std::endl;
  
  {
    class GA* GA;
    test_function2b of;
    
    GA = new class GA(1000,of);
    GA->getCrossover() = 0.80;
    GA->verbosity(true);
    
    // (a,b) = (iterations, population size)
    GA->maximize(atoi(argv[1]), 100); // something small and simple
    
    show_results(GA);   
    
    delete GA;
  }
  
  
  return 0;
}


template <typename T>
void show_results(const GeneticAlgorithm<T>* GA) throw()
{
  T input;
  
  std::cout << "best value found: " << GA->getBest(input) << std::endl;
  std::cout << "with argument: " << input << std::endl;
  std::cout << "bogomean population value: " << GA->getMean() << std::endl;  
}



void show_results(const class GA* GA) throw()
{
  dynamic_bitset input;
  
  std::cout << "best value found: " << GA->getBest(input) << std::endl;
  std::cout << "with argument: " << input  << std::endl;
  std::cout << "bogomean population value: " << GA->getMean() << std::endl;  
}














