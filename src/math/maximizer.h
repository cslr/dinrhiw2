/*
 * maximizer code
 *
 * simple stochastic global optimizers
 * 
 * range for maximums is assumed to be 
 * [-1,1]^D .. [-2,2]^D dimensional hypercube
 * but optimized function should return
 * some (non-maximum) values with other
 * values also.
 *
 */

#ifndef whiteice__math__maximizer_h
#define whiteice__math__maximizer_h

#include "optimized_function.h"
#include "vertex.h"

#ifndef WINOS
#include <sys/times.h>
#endif

#include <map>



namespace whiteice
{
  namespace math
  {
    
    
    // stochastic optimizer
    template <typename T>
      class StochasticOptimizer
      {
      public:
	StochasticOptimizer();
	virtual ~StochasticOptimizer();
	
	T getSolution(vertex<T>& s) const;
      
	// does further optimization by using n secs of CPU time
	// returns number of iterations
	unsigned int optimize
	  (whiteice::optimized_function<T>* f,
	   float secs) ;
	
	unsigned int optimizeMore(float secs);
      
      protected:
	// current best solution
	vertex<T> solution, tr_solution;
	T solutionValue;
      
	
	const vertex<T>& getInternalSolution() const ;
	T calculate(vertex<T>& v) const ;
	
	// candidate solution
	vertex<T> candidate;
	T candValue;
      
	whiteice::optimized_function<T>* f;
	
	// number of dimensions used in search
	unsigned int dimensions;
      
      
	// inheritator overrides!
	// candidate generation:
	
	// initialization of generation
	virtual void initialize() ;
	
	// candidate generation
	virtual void generate() ;
	
	
	// list of vectors produced by
	// optimization process
	matrix<T> R, P, curR;
	vertex<T> mean, curMean;

	// mapping from reduced space dimensions
	// to PCAed full space dimensions
	std::map<unsigned int, unsigned int> toFullSpace;
	
	
	unsigned int nextFraming;
	unsigned int framingInterval;
	
	unsigned int iteration;
	
	// timing
	
	float CLOCKS_SEC;
	float startTime, endTime;
	
	float getTime() const ;
      };
    
    
    
    template <typename T>
      class IHRSearch : public StochasticOptimizer<T>
      {
      public:
	IHRSearch();
	virtual ~IHRSearch();
	
	void initialize() ;
	void generate() ;
	
      private:
	vertex<T> dir;
	T meanSQDif;
      };
    
    
    
    /*
     * gradient descent search
     */
    template <typename T>
      class GradientDescent : public StochasticOptimizer<T>
      {
      public:
	GradientDescent();
	~GradientDescent();
	
	void initialize() ;
	void generate() ;
	
      private:
	T step;
	
	vertex<T> prevcandidate;
	vertex<T> dir, goodDir;
      };
    
    
    
    /************************************************************/
    // definitions of explicit template instantations
    
    extern template class StochasticOptimizer< blas_real<float> >;
    extern template class IHRSearch< blas_real<float> >;
    extern template class GradientDescent< blas_real<float> >;
    
  };
};


#endif
