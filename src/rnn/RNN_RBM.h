/*
 * Recurrent neural network using BB-RBM (RNN-RBM)
 *
 * Implementation follows documentation in
 * docs/neural_network_gradient.pdf/tm 
 *
 * and is based on a research article:
 * 
 * "Modeling Temporal Dependencies in High-Dimensional Sequences:
 *  Application to Polyphonic Music Generation and Transcription"
 * Boulanger-Lewandowski 2012
 *
 * Implementation uses multilayer neural network 
 * instead of linear matrices used in reseach article. Furthermore,
 * recurrent variables r[n] are function visible notes v[n-1] instead 
 * of v[n]. We have r[n] = f(v[n-1], r[n-1]) instead of 
 * r[n] = f(v[n], r[n-1]) which should work better but would make the 
 * code more complex.
 */

#ifndef __whiteice__RNN_RBM_h
#define __whiteice__RNN_RBM_h

#include <vector>
#include <string>

#include <thread>
#include <mutex>
#include <condition_variable>

#include "vertex.h"
#include "nnetwork.h"
#include "BBRBM.h"

#include <stdexcept>

namespace whiteice
{

  template <typename T = whiteice::math::blas_real<float> >
    class RNN_RBM
    {
    public:
    
    RNN_RBM(unsigned int dimVisible = 1,
	    unsigned int dimHidden = 1,
	    unsigned int dimRecurrent = 1);

    RNN_RBM(const whiteice::RNN_RBM<T>& rbm);
    
    ~RNN_RBM();

    RNN_RBM<T>& operator=(const whiteice::RNN_RBM<T>& rbm) throw(std::invalid_argument);

    unsigned int getVisibleDimensions() const;
    unsigned int getHiddenDimensions() const;
    unsigned int getRecurrentDimensions() const;

    void getRNN(whiteice::nnetwork<T>& nn) const;
    void getRBM(whiteice::BBRBM<T>& model) const;


    bool startOptimize(const std::vector< std::vector< whiteice::math::vertex<T> > >& timeseries);

    bool getOptimizeError(unsigned int& iterations, T& error);

    bool isRunning(); // optimization loop is running

    bool stopOptimize();
			  
    
    // resets timeseries synthetization parameters
    void synthStart();
    
    // synthesizes next timestep by using the model
    bool synthNext(whiteice::math::vertex<T>& vnext);
    
    // synthesizes N next candidates using the probabilistic model
    bool synthNext(unsigned int N, std::vector< whiteice::math::vertex<T> >& vnext);
    
    // selects given v as the next step in time-series
    // (needed to be called before calling again synthNext())
    bool synthSetNext(whiteice::math::vertex<T>& v);

    bool save(const std::string& basefilename) const;
    bool load(const std::string& basefilename);
    
    
    protected:
    
    unsigned int dimVisible;
    unsigned int dimHidden;
    unsigned int dimRecurrent;
    
    whiteice::nnetwork<T> nn; // recurrent neural network
    whiteice::BBRBM<T> rbm;   // rbm part
    
    // synthesization variables
    std::mutex synth_mutex;
    bool synthIsInitialized;
    whiteice::math::vertex<T> vprev;
    whiteice::math::vertex<T> rprev;

    // optimization thread parameters
    bool running;
    std::mutex thread_mutex;
    std::thread* optimization_thread;
    
    unsigned int optimization_threads;
    std::mutex optimize_mutex;
    std::condition_variable optimization_threads_cond;

    std::vector< std::vector< whiteice::math::vertex<T> > > timeseries;

    mutable std::mutex model_mutex;
    T best_error;
    unsigned int iterations;
    

    T reconstructionError(whiteice::BBRBM<T>& rbm,
			  whiteice::nnetwork<T>& nn,
			  const std::vector< std::vector< whiteice::math::vertex<T> > >& timeseries) const;

    
    /* 
     * optimizes data likelihood using N-timseries,
     * which are i step long and have dimVisible elements e
     * timeseries[N][i][e]
     */
    void optimize_loop();
    
    };
  
  
  extern template class RNN_RBM< whiteice::math::blas_real<float> >;
  extern template class RNN_RBM< whiteice::math::blas_real<double> >;
  
};


#endif
