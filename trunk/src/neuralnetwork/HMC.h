/*
 * Hamiltonian Monte Carlo Markov Chain sampler 
 * for neurnal network training data.
 */

// TODO: adaptive step lengths DO NOT work very well and
//       are disabled as the default

#ifndef HMC_h
#define HMC_h

#include <vector>
#include <pthread.h>
#include <unistd.h>

#include "vertex.h"
#include "matrix.h"
#include "dataset.h"
#include "dinrhiw_blas.h"
#include "nnetwork.h"
#include "bayesian_nnetwork.h"


namespace whiteice
{

  template <typename T = math::blas_real<float> >
  class HMC
  {
    public:
    HMC(const whiteice::nnetwork<T>& net,
	const whiteice::dataset<T>& ds, bool adaptive=false);
    ~HMC();
    
    // probability functions for hamiltonian MC sampling
    T U(const math::vertex<T>& q, bool useRegulizer = true) const;
    math::vertex<T> Ugrad(const math::vertex<T>& q);
    
    bool startSampler(unsigned int NUMTHREADS=1);
    bool pauseSampler();
    bool continueSampler();
    bool stopSampler();
    
    unsigned int getSamples(std::vector< math::vertex<T> >& samples) const;
    unsigned int getNumberOfSamples() const;

    bool getNetwork(bayesian_nnetwork<T>& bnn);
    
    math::vertex<T> getMean() const;
    // math::matrix<T> getCovariance() const; // DO NOT SCALE TO HIGH DIMENSIONS

    // calculates mean error for the latest N samples, 0 = all samples
    T getMeanError(unsigned int latestN = 0) const;

    bool getAdaptive() const throw(){ return adaptive; }
    
  private:
    whiteice::nnetwork<T> nnet;
    const whiteice::dataset<T>& data;
    
    std::vector< math::vertex<T> > samples;

    bool adaptive;
    
    // used to calculate statistics when needed
    math::vertex<T> sum_mean;
    // math::matrix<T> sum_covariance;
    unsigned int sum_N;

    volatile bool running, paused;
    
    mutable std::vector<pthread_t> sampling_thread; // threads
    mutable pthread_mutex_t solution_lock, start_lock;

    // number of threads that are running (volatile)
    volatile int threadIsRunning;
    
  public:
    void __sampler_loop();
  };
  
  
};


namespace whiteice
{
  extern template class HMC< float >;
  extern template class HMC< double >;
  extern template class HMC< math::blas_real<float> >;
  extern template class HMC< math::blas_real<double> >;    
};


#endif
