/* 
 * Mixture model. Each data point is assigned
 * to a model that gives best performance after
 * which models are optimized again.
 * Continue until convergence. This should work
 * for cases when there are multiple good solutions
 * and our gradients are mixed gradient for each
 * function which pulls into many different directions.
 * 
 * (another approach would be do to K-means clustering 
 *  of gradient values and always pick the largest cluster)
 */

#ifndef __whiteice_Mixture_h
#define __whiteice_Mixture_h

#include <vector>
#include <mutex>
#include <thread>

#include "vertex.h"
#include "dataset.h"
#include "nnetwork.h"


namespace whiteice
{
  
  template <typename T>
    class Mixture
    {
    public:
      Mixture(unsigned int N); // N = number of experts
      ~Mixture();

      //////////////////////////////////////////////////
      
      bool minimize(const whiteice::nnetwork<T>& model,
		    const whiteice::dataset<T>& data);

      bool getSolution(std::vector< math::vertex<T> >& x,
		       std::vector< T >& y,
		       unsigned int& iterations,
		       unsigned int& changes) const;

      bool stopComputation();
      
      bool solutionConverged() const;
      
      bool isRunning() const;

    protected:
      const unsigned int N; // number of mixtures

      mutable std::mutex solutionMutex;
      std::vector< math::vertex<T> > solutions;
      std::vector< T > y;
      unsigned int global_iterations;
      unsigned int latestChanges; // how many data points switched to different solution?

      mutable std::mutex threadMutex;
      std::thread* worker_thread;
      bool thread_running;
      bool converged;

      whiteice::nnetwork<T>* model;
      whiteice::dataset<T> data;
      
      void optimizer_loop();
      
    };


  extern template class Mixture< float >;
  extern template class Mixture< double >;
  extern template class Mixture< whiteice::math::blas_real<float> >;
  extern template class Mixture< whiteice::math::blas_real<double> >;
  
};


#endif
