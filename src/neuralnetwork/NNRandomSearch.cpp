
#include "NNRandomSearch.h"
#include <time.h>

#ifdef WINNT
#include <windows.h>
#endif


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
      thread_is_running = 0;
      
    }

    
    template <typename T>
    NNRandomSearch<T>::~NNRandomSearch()
    {
      start_lock.lock();

      if(running){
	running = false;
	for(unsigned int i=0;i<optimizer_thread.size();i++){
	  delete optimizer_thread[i];
	}
	optimizer_thread.resize(0);
      }

      start_lock.unlock();
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

      start_lock.lock();
      
      if(running == true){
	start_lock.unlock();
	return false;
      }
      
      
      this->data = &data;
      this->NTHREADS = NTHREADS;
      this->nn_arch = arch;
      best_error = T(1000.0f);
      converged_solutions = 0;
      running = true;
      thread_is_running = 0;

      optimizer_thread.resize(NTHREADS);

      // FIXME new std::thread can throw exceptions?
      
      for(unsigned int i=0;i<optimizer_thread.size();i++){
	optimizer_thread[i] =
	  new std::thread(&NNRandomSearch<T>::optimizerloop, this);
      }
      
      start_lock.unlock();

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
      
      solution_lock.lock();
      
      nn.importdata(bestx);
      error = best_error;
      Nconverged = converged_solutions;

      solution_lock.unlock();

      return true;
    }

    
    /* used to pause, continue or stop the optimization process */
    template <typename T>
    bool NNRandomSearch<T>::stopComputation()
    {
      start_lock.lock();

      running = false;
      
      for(unsigned int i=0;i<optimizer_thread.size();i++){
	if(optimizer_thread[i])
	  optimizer_thread[i]->join();
      }
      optimizer_thread.resize(0);
      
      start_lock.unlock();

      return true;
    }


    template <typename T>
    void NNRandomSearch<T>::optimizerloop()
    {
      if(data == NULL)
	return; // silent failure if there is bad data

      // normalize_weights_to_unity(nn);
      math::vertex<T> err, weights;
      
      thread_is_running++;
      
      // keep looking for solution forever
      while(running){
	// creates random neural network
	nnetwork<T> nn(nn_arch);
	
	// calculates PCA-rization solution
	T alpha = 0.5f;
	negative_feedback_between_neurons(nn, *data, alpha, true);

	T error = T(0.0f);
	{
	  // 1. calculates error
	  for(unsigned int i=0;i<data->size(0);i++){
	    nn.input() = data->access(0, i);
	    nn.calculate(false);
	    err = data->access(1,i) - nn.output();
	      
	    for(unsigned int i=0;i<err.size();i++)
	      error += (err[i]*err[i]) / T((float)err.size());
	  }
	    
	  error /= T((float)data->size(0));
	  error *= T(0.5f);
	}
	
	// 2. checks if the result is better than the best one
	if(error < best_error){
	  solution_lock.lock();

	  // improvement (smaller error with early stopping)
	  best_error = error;
	  nn.exportdata(bestx);

	  solution_lock.unlock();
	}
	
	converged_solutions++;
		
      }
      
      thread_is_running--;
      
    }


    
    template class NNRandomSearch< float >;
    template class NNRandomSearch< double >;
    template class NNRandomSearch< blas_real<float> >;
    template class NNRandomSearch< blas_real<double> >;    
    
  };
};

