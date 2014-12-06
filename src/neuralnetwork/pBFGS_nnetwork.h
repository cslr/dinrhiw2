/*
 * parallel BFGS optimizer for neural networks
 *
 */

#ifndef pBFGS_nnetwork_h
#define pBFGS_nnetwork_h

#include "BFGS_nnetwork.h"
#include "BFGS.h"
#include "nnetwork.h"
#include "vertex.h"
#include <vector>

#include <thread>
#include <mutex>


namespace whiteice
{
  
  template <typename T=math::blas_real<float> >
    class pBFGS_nnetwork
    {
    public:

    pBFGS_nnetwork(const nnetwork<T>& net, const dataset<T>& data, bool overfit=false, bool negativefeedback=false);
    ~pBFGS_nnetwork();

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
    bool overfit;
    bool negativefeedback;
    
    math::vertex<T> global_best_x;
    T global_best_y;
    unsigned int global_iterations;

    
    std::vector< std::unique_ptr< BFGS_nnetwork<T> > > optimizers;
    volatile bool thread_running;
    volatile int thread_is_running;
    mutable std::mutex thread_is_running_mutex;
    mutable std::condition_variable thread_is_running_cond;
    
    std::thread updater_thread;

    mutable std::mutex bfgs_mutex;
    mutable std::mutex thread_mutex;

    private:

    void updater_loop();
      
    };

  

    
  extern template class pBFGS_nnetwork< float >;
  extern template class pBFGS_nnetwork< double >;
  extern template class pBFGS_nnetwork< math::blas_real<float> >;
  extern template class pBFGS_nnetwork< math::blas_real<double> >;

};

#endif
