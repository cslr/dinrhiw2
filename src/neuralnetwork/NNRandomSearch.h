/*
 * Parallel neural network gradient descent optimizer
 *
 * - keeps looking for the best possible solution forever
 *   (fresh restart after convergence to local minima)
 *
 *
 */



#include "dinrhiw_blas.h"
#include "dataset.h"
#include "dinrhiw.h"

#include <thread>
#include <mutex>


#ifndef NNRandomSearch_h
#define NNRandomSearch_h


namespace whiteice
{
  namespace math
  {
    // used to store top N results during the search
    template <typename T=blas_real<float> >
      struct nn_solution
      {
	vertex<T> solution;
	T error;
      };
      
    
    template <typename T=blas_real<float> >
      class NNRandomSearch
      {
      public:
      
	NNRandomSearch();
	~NNRandomSearch();

	/*
	 * starts the optimization process using data as 
	 * the dataset as a training and testing data 
	 * (implements early stopping)
	 *
	 * Uses neural network with architecture arch.
	 *
	 * Executes NTHREADS in parallel when looking for
	 * the optimal solution.
	 */
	bool startOptimize(const whiteice::dataset<T>& data,
			   const std::vector<unsigned int>& arch,
			   unsigned int NTHREADS);


	/*
	 * returns the best NN solution found so far and
	 * its average error in testing dataset and the number
	 * of converged solutions so far.
	 */
	bool getSolution(whiteice::nnetwork<T>& nn,
			 T& error, unsigned int& Nconverged);

	/* used to pause, continue or stop the optimization process */
	bool stopComputation();

      private:
	std::vector<unsigned int> nn_arch;
	
	vertex<T> bestx;
	T best_error;
	unsigned int converged_solutions;

        whiteice::dataset<T> const* data;
	
	unsigned int NTHREADS;
	std::vector<thread*> optimizer_thread;
        std::mutex solution_lock, start_lock;
        
        volatile bool running;
        volatile int thread_is_running;

        void optimizerloop();
	
      };
        
  };
};


namespace whiteice
{
  namespace math
  {
    extern template class NNRandomSearch< float >;
    extern template class NNRandomSearch< double >;
    extern template class NNRandomSearch< blas_real<float> >;
    extern template class NNRandomSearch< blas_real<double> >;
    
    
  };
};



#endif
