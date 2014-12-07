/*
 * Parallel neural network gradient descent optimizer
 *
 * - keeps looking for the best possible solution forever
 *   (fresh restart after convergence to local minima)
 *
 *
 */


#include <pthread.h>
#include "dinrhiw_blas.h"
#include "dataset.h"
#include "dinrhiw.h"

#ifndef NNGradDescent_h
#define NNGradDescent_h


namespace whiteice
{
  namespace math
  {
    template <typename T=blas_real<float> >
      class NNGradDescent
      {
      public:
      
	NNGradDescent(bool negativefeedback = false);
	~NNGradDescent();

	/*
	 * starts the optimization process using data as 
	 * the dataset as a training and testing data 
	 * (implements early stopping)
	 *
	 * Uses neural network with architecture arch.
	 *
	 * Executes NTHREADS in parallel when looking for
	 * the optimal solution and goes max to 
	 * MAXITERS iterations when looking for gradient
	 * descent solution
	 */
	bool startOptimize(const whiteice::dataset<T>& data,
			   const std::vector<unsigned int>& arch,
			   unsigned int NTHREADS,
			   unsigned int MAXITERS = 10000);
      

	/*
	 * returns the best NN solution found so far and
	 * its average error in testing dataset and the number
	 * of converged solutions so far.
	 */
	bool getSolution(whiteice::nnetwork<T>& nn,
			 T& error, unsigned int& Nconverged);

	/* used to stop the optimization process */
	bool stopComputation();

      private:
	std::vector<unsigned int> nn_arch;
      
        bool negativefeedback;
      
        vertex<T> bestx;
        T best_error;
        unsigned int converged_solutions;

	const whiteice::dataset<T>* data;
	
	unsigned int NTHREADS;
        unsigned int MAXITERS;
	std::vector<pthread_t> optimizer_thread;
	pthread_mutex_t solution_lock, start_lock;

        volatile bool running;
        volatile int thread_is_running;

      public:
	void __optimizerloop();
	
      };
        
  };
};


namespace whiteice
{
  namespace math
  {
    // extern template class NNGradDescent< float >;
    extern template class NNGradDescent< double >;
    // extern template class NNGradDescent< blas_real<float> >;
    // extern template class NNGradDescent< blas_real<double> >;
    
    
  };
};



#endif