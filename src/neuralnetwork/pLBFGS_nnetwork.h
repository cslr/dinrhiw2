/*
 * parallel L-BFGS optimizer for neural networks
 *
 */

#ifndef pLBFGS_nnetwork_h
#define pLBFGS_nnetwork_h

#include "LBFGS_nnetwork.h"
#include "LBFGS.h"
#include "nnetwork.h"
#include "vertex.h"
#include <vector>
#include <pthread.h>


namespace whiteice
{
  
  template <typename T=math::blas_real<float> >
    class pLBFGS_nnetwork
    {
    public:

    pLBFGS_nnetwork(const nnetwork<T>& net, const dataset<T>& data, bool overfit=false);
    ~pLBFGS_nnetwork();

    bool minimize(unsigned int NUMTHREADS);

    bool getSolution(math::vertex<T>& x, T& y, unsigned int& iterations) const;

    T getError(const math::vertex<T>& x) const;
    
    // continues, pauses, stops computation
    bool continueComputation();
    bool pauseComputation();
    bool stopComputation();
    
    private:

    const nnetwork<T>& net;
    const dataset<T>& data;

    volatile bool thread_running;
    std::vector< LBFGS_nnetwork<T>* > optimizers;

    bool overfit;

    math::vertex<T> global_best_x;
    T global_best_y;
    unsigned int global_iterations;

    pthread_t updater_thread;
    mutable pthread_mutex_t bfgs_lock;
    mutable pthread_mutex_t thread_lock;

    public:

    void __updater_loop();
      
    };

  

    
  // extern template class pLBFGS_nnetwork< float >;
  // extern template class pLBFGS_nnetwork< double >;
  extern template class pLBFGS_nnetwork< math::blas_real<float> >;
  // extern template class pLBFGS_nnetwork< math::blas_real<double> >;

};

#endif
