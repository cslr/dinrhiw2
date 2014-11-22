
#include "NNRandomSearch.h"
#include <time.h>

#ifdef WINNT
#include <windows.h>
#endif


extern "C" {
  static void* __nnrandom_optimizer_thread_init(void* param);
};


namespace whiteice
{
  namespace math
  {

    template <typename T>
    NNRandomSearch<T>::NNRandomSearch()
    {
      best_error = T(1000.0f);
      converged_solutions = 0;
      data = NULL;
      NTHREADS = 0;

      running = false;

      pthread_mutex_init(&solution_lock, 0);
      pthread_mutex_init(&start_lock, 0);
    }

    
    template <typename T>
    NNRandomSearch<T>::~NNRandomSearch()
    {
      pthread_mutex_lock( &start_lock );

      if(running)
	for(unsigned int i=0;i<optimizer_thread.size();i++){
	  pthread_cancel( optimizer_thread[i] );
	}

      running = false;

      pthread_mutex_unlock( &start_lock );

      pthread_mutex_destroy( &solution_lock );
      pthread_mutex_destroy( &start_lock );
    }
    
    /*
     * starts the optimization process using data as 
     * the dataset as a training and testing data 
     * (implements early stopping)
     *
     * Executes NTHREADS in parallel when looking for
     * the optimal solution.
     */
    template <typename T>
    bool NNRandomSearch<T>::startOptimize(const whiteice::dataset<T>& data,
					 const std::vector<unsigned int>& arch, 
					 unsigned int NTHREADS)
    {
      if(arch.size() < 2) return false;

      if(data.getNumberOfClusters() != 2) return false;

      if(data.size(0) != data.size(1)) return false;

      // need at least 50 datapoints
      if(data.size(0) <= 50) return false;

      if(data.dimension(0) != arch[0] ||
	 data.dimension(1) != arch[arch.size()-1])
	return false;
      
      pthread_mutex_lock( &start_lock );
      
      if(running == true){
	pthread_mutex_unlock( &start_lock );
	return false;
      }
      
      
      this->data = &data;
      this->NTHREADS = NTHREADS;
      this->nn_arch = arch;
      best_error = T(1000.0f);
      converged_solutions = 0;
      running = true;

      optimizer_thread.resize(NTHREADS);

      for(unsigned int i=0;i<optimizer_thread.size();i++){
	pthread_create(& (optimizer_thread[i]), 0,
		       __nnrandom_optimizer_thread_init, (void*)this);
	pthread_detach( optimizer_thread[i] );
      }
      
      
      pthread_mutex_unlock( &start_lock );

      return true;
    }
    

    
    /*
     * returns the best NN solution found so far and
     * its average error in testing dataset and the number
     * of converged solutions so far.
     */
    template <typename T>
    bool NNRandomSearch<T>::getSolution(whiteice::nnetwork<T>& nn,
				       T& error, unsigned int& Nconverged)
    {
      // checks if the neural network architecture is the correct one

      std::vector<unsigned int> a;
      nn.getArchitecture(a);

      if(a.size() != nn_arch.size()) return false;
      for(unsigned int i=0;i<a.size();i++)
	if(a[i] != nn_arch[i]) return false;
      
      
      pthread_mutex_lock( &solution_lock );

      nn.importdata(bestx);
      error = best_error;
      Nconverged = converged_solutions;

      
      pthread_mutex_unlock( &solution_lock );

      return true;
    }

    
    /* used to pause, continue or stop the optimization process */
    template <typename T>
    bool NNRandomSearch<T>::stopComputation()
    {
      pthread_mutex_lock( &solution_lock );
      pthread_mutex_lock( &start_lock );

      running = false;
      

      for(unsigned int i=0;i<optimizer_thread.size();i++){
	pthread_cancel( optimizer_thread[i] );
      }

      pthread_mutex_unlock( &start_lock );
      pthread_mutex_unlock( &solution_lock );

      return true;
    }


    template <typename T>
    void NNRandomSearch<T>::__optimizerloop()
    {
      if(data == NULL)
	return; // silent failure if there is bad data

      std::vector< nn_solution<T> > top_solutions;
      top_solutions.resize(10); // keeps 10 best results (local mode)

      // FIXME: can throw bad_alloc exception (do try{}catch();)
      nnetwork<T>* __test_nn = new nnetwork<T>(nn_arch);
      
      normalize_weights_to_unity(*__test_nn);

      for(unsigned int i=0;i<top_solutions.size();i++){
	__test_nn->exportdata(top_solutions[i].solution);
	top_solutions[i].error = T(1000000000.0f); // nearly infinite
      }

      delete __test_nn;

      
      while(running){
	// keep looking for solution forever

	// creates random neural network
	nnetwork<T> nn(nn_arch);

	normalize_weights_to_unity(nn);
	
	math::vertex<T> err, weights;
	T error = T(0.0f);
	
	// 50% chance we select a new vector within
	// the "space" created from randomly weighted N best
	// solutions found so far
	// NOTE: because of high dimensions, we basically search
	// the base around "edges" (close to current ones)
	// NOTE2: top_solutions[i] should converge to MODE of
	// the distribution
	if((rand() & 1) == 1){
	  nn.exportdata(weights);
	  weights.zero();

	  T sum = T(0.0f);

	  for(unsigned int i=0;i<top_solutions.size();i++){
	    // alpha = random([0, 1])
	    T alpha = T( ((float)rand())/((float)RAND_MAX) );
	    weights += alpha*top_solutions[i].solution;
	    sum += alpha;
	  }

	  weights *= T(1.0f)/sum;
	  nn.importdata(weights);

	  normalize_weights_to_unity(nn);
	  nn.exportdata(weights);
	}
	// otherwise we use randomly generated weights
	// [pure random search]
	
	
	{
	  
	  // 1. calculates error from the testing dataset
	  for(unsigned int i=0;i<data->size(0);i++){
	    nn.input() = data->access(0, i);
	    nn.calculate(false);
	    err = data->access(1,i) - nn.output();
	      
	    for(unsigned int i=0;i<err.size();i++)
	      error += (err[i]*err[i]) / T((float)err.size());
	  }
	    
	  error /= T((float)data->size());
	}
	
	// 2. checks if the result is better than the best one
	pthread_mutex_lock( &solution_lock );
	
	if(error < best_error){
	  // improvement (smaller error with early stopping)
	  best_error = error;
	  nn.exportdata(bestx);
	}
	
		
	converged_solutions++;
	
	pthread_mutex_unlock( &solution_lock);
	
	
	// 3. stores the best results to top_solutions[i]
	{
	  unsigned int index = top_solutions.size() + 1;
	  T best_index_error = T(0.0f);
	  
	  for(unsigned int i=0;i<top_solutions.size();i++){
	    // if there is improvement the best N solutions
	    if(error < top_solutions[i].error){
	      // if the found solution is worse than
	      // the currently found best one
	      // (we replace the worst smaller error from
	      //  the N best solutions)
	      
	      if(best_index_error < error){ 
		best_index_error = error;
		index = i;
	      }
	    }
	  }
	  
	  
	  if(index < top_solutions.size()){
	    nn.exportdata(top_solutions[index].solution);
	    top_solutions[index].error = best_index_error;
	  }
	    
	}
		
      }
      
    }


    
    //template class NNRandomSearch< float >;
    //template class NNRandomSearch< double >;
    template class NNRandomSearch< blas_real<float> >;
    //template class NNRandomSearch< blas_real<double> >;    
    
  };
};


extern "C" {
  void* __nnrandom_optimizer_thread_init(void *optimizer_ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(optimizer_ptr)
      ((whiteice::math::NNRandomSearch< whiteice::math::blas_real<float> >*)optimizer_ptr)->__optimizerloop();
    
    pthread_exit(0);

    return 0;
  }
};
