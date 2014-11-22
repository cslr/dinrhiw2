/*
 * Hamiltonian Monte Carlo Markov Chain sampler 
 */

// TODO: adaptive step length DO NOT work very well and
//       is disabled as the default

#ifndef HMC_abstract_h
#define HMC_abstract_h

#include <vector>
#include <pthread.h>
#include <unistd.h>

#include "vertex.h"
#include "matrix.h"
#include "dinrhiw_blas.h"


namespace whiteice
{

  template <typename T = math::blas_real<float> >
  class HMC_abstract
  {
    public:
    HMC_abstract(bool adaptive=false);
    ~HMC_abstract();
    
    // probability functions for hamiltonian MC sampling of
    // P ~ exp(-U(q)) distribution
    virtual T U(const math::vertex<T>& q) const = 0;
    virtual math::vertex<T> Ugrad(const math::vertex<T>& q) = 0;

    // a starting point q for the sampler (may not be random)
    virtual void starting_position(math::vertex<T>& q) const = 0;
    
    bool startSampler();
    bool pauseSampler();
    bool continueSampler();
    bool stopSampler();
    
    unsigned int getSamples(std::vector< math::vertex<T> >& samples) const;
    unsigned int getNumberOfSamples() const;
    
    math::vertex<T> getMean() const;
    math::matrix<T> getCovariance() const;

    // calculates mean error for the latest N samples, 0 = all samples
    T getMeanError(unsigned int latestN = 0) const;

    bool getAdaptive() const throw(){ return adaptive; }

  protected:

    bool adaptive;
    
  private:
    std::vector< math::vertex<T> > samples;
    
    // used to calculate statistics when needed
    math::vertex<T> sum_mean;
    math::matrix<T> sum_covariance;
    unsigned int sum_N;

    bool running, paused;
    
    mutable pthread_t sampling_thread;
    mutable pthread_mutex_t solution_lock, start_lock;

    bool threadIsRunning;
    
  public:
    void __sampler_loop();
  };
  
  
};


namespace whiteice
{
  extern template class HMC_abstract< float >;
  extern template class HMC_abstract< double >;
  extern template class HMC_abstract< math::blas_real<float> >;
  extern template class HMC_abstract< math::blas_real<double> >;    
};


#endif
