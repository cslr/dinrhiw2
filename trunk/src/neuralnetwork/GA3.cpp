
#include "GA3.h"
#include <unistd.h>


extern "C" {
  static void* __ga3_thread_init(void* param);
};


namespace whiteice
{
  template <typename T>
  GA3<T>::GA3(function< math::vertex<T>, T>* f,
	      unsigned int dimension)
  {
    this->f = f;
    this->DIM = dimension;
    p_crossover = T(0.80);
    p_mutation  = T(1.0/sqrt((float)dimension));
  }
  

  template <typename T>
  GA3<T>::~GA3()
  {
    if(running) stop();
    running = false;
  }

  
  template <typename T>
  bool GA3<T>::minimize()
  {
    running = true;
    if(pthread_create(&optimizer_thread, 0,
		      __ga3_thread_init, (void*)this) == 0)
      return true;
    else
      return false;
  }
  

  template <typename T>
  bool GA3<T>::isRunning() const throw()
  {
    return running;
  }
  

  template <typename T>
  bool GA3<T>::stop() throw()
  {
    running = false;
    while(thread_running)
      sleep(1); // wait for optimizer thread to finish
    return true;
  }
  
  
  // returns the best solution found so far
  template <typename T>
  T GA3<T>::getBestSolution(math::vertex<T>& solution) const throw()
  {
    solution = very_best_candidate;
    return very_best_result;
  }

  
  template <typename T>
  void GA3<T>::__optimization_thread()
  {
    thread_running = true;

    // generate initial population
    const unsigned int POPSIZE = 5000;

    very_best_result = T(100000000000.0);

    for(unsigned int i=0;i<POPSIZE;i++){
      math::vertex<T> v;
      v.resize(DIM);
      for(unsigned int j=0;j<DIM;j++)
	v[j] = T(2.0f*((float)rand())/((float)RAND_MAX) - 1.0f);

      T r = f->calculate(v);

      solutions.push_back(v);
      results.push_back(r);

      // population[r] = solutions.size()-1;

      if(r < very_best_result){
	very_best_result = r;
	very_best_candidate = v;
      }
    }
    
    
    while(running){

      // 1.cross-over step
      for(unsigned int i=0;i<POPSIZE;i++){
	T r = T(rand()/((float)RAND_MAX));
	
	if(r < p_crossover){
	  unsigned int mate = rand() % POPSIZE;

	  math::vertex<T> in1, in2;
	  math::vertex<T> out1, out2;

	  in1 = solutions[i];
	  in2 = solutions[mate];

	  crossover(in1, in2, out1, out2);

	  T r1 = f->calculate(out1);
	  T r2 = f->calculate(out2);

	  if(r1 < results[i]){
	    results[i] = r1;
	    solutions[i] = out1;

	    if(r1 < very_best_result){
	      very_best_result = r1;
	      very_best_candidate = out1;
	    }
	  }

	  if(r2 < results[mate]){
	    results[mate] = r2;
	    solutions[mate] = out2;

	    if(r2 < very_best_result){
	      very_best_result = r2;
	      very_best_candidate = out2;
	    }
	  }
	}
      }

      // 2. mutation step
      for(unsigned int i=0;i<POPSIZE;i++){
	T r = T(rand()/((float)RAND_MAX));
	
	if(r < p_mutation){
	  math::vertex<T> in1;
	  math::vertex<T> out1;

	  in1 = solutions[i];

	  mutate(in1, out1);

	  T r1 = f->calculate(out1);
	  
	  if(r1 < results[i]){
	    results[i] = r1;
	    solutions[i] = out1;

	    if(r1 < very_best_result){
	      very_best_result = r1;
	      very_best_candidate = out1;
	    }
	  }

	}
      }
      
      
    }

    thread_running = false;
  }

  template <typename T>
  void GA3<T>::crossover(const math::vertex<T>& in1,
			 const math::vertex<T>& in2,
			 math::vertex<T>& out1,
			 math::vertex<T>& out2)
  {
    unsigned int crossover_point = rand() % DIM;

    out1.resize(in1.size());
    out2.resize(in2.size());

    for(unsigned int j=0;j<DIM;j++){
      if(j<crossover_point){
	out1[j] = in1[j];
	out2[j] = in2[j];
      }
      else{
	out1[j] = in2[j];
	out2[j] = in1[j];	
      }
    }
  }

  template <typename T>
  void GA3<T>::mutate(const math::vertex<T>& in1,
		      math::vertex<T>& out1)
  {
    out1.resize(DIM);

    // add random noise (NOTE: should be normally distributed)
    
    for(unsigned i=0;i<DIM;i++){
      T r = T(((float)rand())/((float)RAND_MAX) - 0.5f);
      
      out1[i] = in1[i] + r;
    }
    
  }
  
};

namespace whiteice
{
  template class GA3< float >;
  template class GA3< double >;
  template class GA3< math::atlas_real<float> >;
  template class GA3< math::atlas_real<double> >;    
};


extern "C" {
  void* __ga3_thread_init(void *ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(ptr)
      ((whiteice::GA3< whiteice::math::atlas_real<float> >*)ptr)->__optimization_thread();
    
    pthread_exit(0);

    return 0;
  }
};
