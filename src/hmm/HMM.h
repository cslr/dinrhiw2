/*
 * HMM.h
 *
 *  Created on: 5.7.2015
 *      Author: Tomas
 */

#ifndef HMM_HMM_H_
#define HMM_HMM_H_

#include "real.h"
#include "RNG.h"

#include <vector>
#include <stdexcept>
#include <exception>


namespace whiteice {

  /*
   * discrete Hidden Markov Model using arbitrary precision numbers (math::realnumber)
   * for analyzing and generating neuromancer/resonanz stimulation sequences
   *
   * TODO this will be initially non-threaded version but will be
   * threaded after the code has been tested
   */
  class HMM {
  public:
    HMM(unsigned int visibleStates, unsigned int hiddenStates) throw(std::logic_error);
    virtual ~HMM();
    
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
    
    /**
     * trains HMM parameters from discrete observational states
     * (unsigned integers are state numbers) using
     * Expectation Maximization (EM) algorithm
     *
     * returns log(probability) of training data
     */
    double train(const std::vector<unsigned int>& observations) throw (std::invalid_argument);

    
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
		     const std::vector<unsigned int>& observations) const throw (std::invalid_argument);
    
    /*
     * calculations log(probability) of observations
     */
    double logprobability(std::vector<unsigned int>& observations) const
      throw (std::invalid_argument);
    
  private:
    // number of visible and hidden states
    const unsigned int numVisible, numHidden;
    unsigned int precision; // precision of computations
    
    // parameters of HMM:
    // initial hidden state distribution of HMM
    std::vector< whiteice::math::realnumber > ph;
    
    // state transition probabilities between hidden states i=>j: A[i][j]
    std::vector< std::vector< whiteice::math::realnumber > > A;
    
    // visible state k emission/generation probabilities
    // (i,j) => visible state k [we use current and previous hidden state when generating symbol]
    std::vector< std::vector< std::vector< whiteice::math::realnumber > > > B;
    
    whiteice::RNG<double> rng; // random number generator used by HMM for simulations
    
    // samples variable i according to probability p(i)
    unsigned int sample(const std::vector<math::realnumber>& p) const;
    
  };
  
} /* namespace whiteice */

#endif /* HMM_HMM_H_ */
