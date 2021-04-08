/*
 * HMM.h
 *
 *  Created on: 5.7.2015
 *      Author: Tomas
 */

#ifndef HMM_HMM_H_
#define HMM_HMM_H_

#include "real.h"
// #include "RNG.h"

#include <vector>
#include <string>
#include <stdexcept>
#include <exception>
#include <thread>
#include <mutex>

namespace whiteice {

  /*
   * discrete Hidden Markov Model using arbitrary precision numbers 
   * (math::realnumber)
   * for analyzing and generating neuromancer/resonanz eeg sequences
   *
   */
  class HMM {
  public:
    HMM();
    HMM(unsigned int visibleStates, unsigned int hiddenStates);
    
    HMM(const HMM& hmm);
    virtual ~HMM();

    HMM& operator=(const HMM& hmm);
    
    /**
     * Sets arbitrary precision number's
     * precision for calculations
     * TODO set good default value so finetuning is rarely needed
     */
    bool setPrecision(unsigned int precision);

    /**
     * Sets ph, A, and B to random (initial) values before optimization
     */
    void randomize();

    void randomize(std::vector< whiteice::math::realnumber >& ph,
		   std::vector< std::vector< whiteice::math::realnumber > >& A,
		   std::vector< std::vector< std::vector< whiteice::math::realnumber > > >& B) const;

    /**
     * Saves ph, A, B values to disk (arbitrary precision)
     */
    bool saveArbitrary(const std::string& filename) const;

    /**
     * Loads ph, A, B values from disk (arbitrary precision)
     */
    bool loadArbitrary(const std::string& filename);
    
    /**
     * trains HMM parameters from discrete observational states
     * (unsigned integers are state numbers) using
     * Expectation Maximization (EM) algorithm
     *
     * returns log(probability) of training data
     */
    double train(const std::vector<unsigned int>& observations,
		 const unsigned int MAXITERS = 1000,
		 const bool verbose = true);

    /**
     * Starts background thread for computation:
     *
     * trains HMM parameters from discrete observational states
     * (unsigned integers are state numbers) using
     * Expectation Maximization (EM) algorithm
     *
     */
    bool startTrain(const std::vector<unsigned int>& observations,
		 const unsigned int MAXITERS = 1000,
		 const bool verbose = true);

    // returns true if optimizer thread is running
    bool isRunning();

    /**
     * returns current log(probability) of training data
     */
    double getSolutionGoodness();

    /**
     * Stops background thread for computation.
     */
    bool stopTrain();
    
    
    /**
     * samples given length observation stream from HMM
     */
    bool sample(const unsigned int numberOfObservations,
		std::vector<unsigned int>& observations) const;
    
    /**
     * finds maximum likelihood hidden states describing observations
     * as well as possible by using viterbi algorithm:
     * max(h) p(v|h)
     *
     * returns log(probability) of the optimum hidden states
     */
    double ml_states(std::vector<unsigned int>& hidden,
		     const std::vector<unsigned int>& observations) const;
      
    /**
     * predicts the next hidden state given observation and current known state.
     * 
     * returns log(probability) of the chosen next state
     */
    double next_state(const unsigned int currentState,
		      unsigned int& nextState,
		      const unsigned int observation) const;
    
    /*
     * calculations log(probability) of observations
     */
    double logprobability(const std::vector<unsigned int>& observations) const;

    
    std::vector< whiteice::math::realnumber >& getPI(){
      std::lock_guard<std::mutex> lock(solution_mutex);
      return ph;
    }
    
    const std::vector< whiteice::math::realnumber >& getPI() const {
      std::lock_guard<std::mutex> lock(solution_mutex);
      return ph;
    }

    std::vector< std::vector< whiteice::math::realnumber > >& getA(){
      std::lock_guard<std::mutex> lock(solution_mutex);
      return A;
    }
    
    const std::vector< std::vector< whiteice::math::realnumber > >& getA() const {
      std::lock_guard<std::mutex> lock(solution_mutex);
      return A;
    }
    
    std::vector< std::vector< std::vector< whiteice::math::realnumber > > >& getB() {
      std::lock_guard<std::mutex> lock(solution_mutex);
      return B;
    }
    
    const std::vector< std::vector< std::vector< whiteice::math::realnumber > > >& getB() const {
      std::lock_guard<std::mutex> lock(solution_mutex);
      return B;
    }

    unsigned int getNumVisibleStates() const  { return numVisible; }
    unsigned int getNumHiddenStates() const  { return numHidden; }

    
    // saves and loads HMM to binary file
    bool load(const std::string& filename) ;
    bool save(const std::string& filename) const ;
    
  private:
    // normalizes parameters by ordering hidden states according to probabilities
    void normalize_parameters();
    
    void normalize_parameters(std::vector< whiteice::math::realnumber >& ph,
			      std::vector< std::vector< whiteice::math::realnumber > >& A,
			      std::vector< std::vector< std::vector< whiteice::math::realnumber > > >& B) const;

    //////////////////////////////////////////////////////////////////////////////
    
    // function which runs optimizer loop that optimizes HMM state
    void optimizer_loop();

    std::thread* optimizer_thread = nullptr;
    mutable std::mutex thread_mutex, solution_mutex;
    bool thread_running = false;
    bool solution_converged = false;

    unsigned int iterations = 0;
    unsigned int MAXITERS = 1000;
    
    std::vector<unsigned int> observations;
    bool verbose = false;

    double best_logp;
    std::vector< whiteice::math::realnumber > best_ph;
    std::vector< std::vector< whiteice::math::realnumber > > best_A;
    std::vector< std::vector< std::vector< whiteice::math::realnumber > > > best_B;

    

    //////////////////////////////////////////////////////////////////////////////
    
    // number of visible and hidden states
    unsigned int numVisible, numHidden;
    unsigned int precision; // precision of computations
    
    // parameters of HMM:
    // initial hidden state distribution of HMM
    std::vector< whiteice::math::realnumber > ph;
    
    // state transition probabilities between hidden states i=>j: A[i][j]
    std::vector< std::vector< whiteice::math::realnumber > > A;
    
    // visible state k emission/generation probabilities
    // (i,j) => visible state k [we use current and previous hidden state when generating symbol]
    std::vector< std::vector< std::vector< whiteice::math::realnumber > > > B;
    
    // whiteice::RNG<double> rng; // random number generator used by HMM for simulations

  public:

    // samples variable i according to probability p(i)
    unsigned int sample(const std::vector<math::realnumber>& p) const;

    // returns initial hidden state from which start reading observations using next_state()
    unsigned int sampleInitialHiddenState() const{
      return this->sample(this->getPI());
    }


    
  };
  
} /* namespace whiteice */

#endif /* HMM_HMM_H_ */
