/*
 * genetic algorithm for real-valued vectors (2014)
 *
 * minimizes target function where parameter space is [-1.0,1.0]
 *
 */

#ifndef GA3_h
#define GA3_h

#include <vector>
#include "function.h"
#include "vertex.h"
#include "dinrhiw_blas.h"
#include <pthread.h>

namespace whiteice
{
  template <typename T=math::blas_real<float> >
  class GA3
  {
  public:
  
    GA3(function< math::vertex<T>, T>* f,
	unsigned int dimension);
    ~GA3();
  
    T getCrossover() const throw(){ return p_crossover; }
    T getMutation() const throw(){ return p_mutation; }

    bool minimize();

    bool isRunning() const throw();

    bool stop() throw();
  
    // returns the best solution found so far
    T getBestSolution(math::vertex<T>& solution) const throw();
  private:
    function< math::vertex<T>, T>* f;
    unsigned int DIM; // number of dimensions
    
    T p_crossover;
    T p_mutation;

    // T goodness => solution
    // std::map< T, unsigned int > population;
    std::vector< math::vertex<T> > solutions;
    std::vector< T > results;

    math::vertex<T> very_best_candidate;
    T very_best_result;

    bool running;
    bool thread_running;
    pthread_t optimizer_thread;
  
  public:
    void __optimization_thread();

  private:
    void crossover(const math::vertex<T>& in1,
		   const math::vertex<T>& in2,
		   math::vertex<T>& out1,
		   math::vertex<T>& out2);

    void mutate(const math::vertex<T>& in1,
		math::vertex<T>& out1);
    
  };
    
};


namespace whiteice
{
  extern template class GA3< float >;
  extern template class GA3< double >;
  extern template class GA3< math::blas_real<float> >;
  extern template class GA3< math::blas_real<double> >;    
};


#endif

