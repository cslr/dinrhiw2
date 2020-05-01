/*
 * very basic
 * genetic algorithm for function optimization
 *
 * TODO: optimize, make more robust, faster, improve.
 *
 * idea: if / when lots of example values of x are
 * available, turn them to bits easily (direct binary string)
 * bits(x) and then use ica/pca to gaussianice/minimize information
 * between bits. x_ = ica(bits(x)). this should remove same information between
 * bits - bits cannot 'conflict' if b1 != b2. However making correlation matrix Rx = I
 * would be probably better/easier idea.
 *
 * If bits cannot conflict then feature which bits code is either on or off.
 * and not in-between. good/bad thing?
 *
 * for example changing representation of bits *during* optimization
 * determinized by:  ica(bits(x_good_ones)). -> coding is optimized to
 * represent information for good candidates very compactly ->
 * more bits left for coding extra features for good ones to improve.
 *
 * however: this makes mutation to be more disasterious (if it's not very good - common).
 *
 * ---
 * Some soft of histogram equalization could be also be a good method for keeping
 * distribution of genes 'flat' and without peaks. -> this kind of 'zooms' in to
 * peak so that the problem space is again flat: -> well used number of variables.
 * (uses whole problem space/dimensionality well for solving the problem).
 * ---
 *
 * above:
 * kind of feedback between 'preprocessing' stage (choosing of representation)
 * and optimization stage
 * or better (view point):  integrates choose of representation and actual optimization
 * into one task, actually kind of EM method when you think it:
 *    representation is constantly optimized for optimization task (needed parameters
 *       are constantly changing: optimizes for current instance (expected value) )
 *    and optimization optimizes normally.
 * 
 */

#ifndef GeneticAlgorithm_h
#define GeneticAlgorithm_h

#include <vector>
#include <bitset>
#include "function.h"

namespace whiteice
{

  template <typename T>
    class GeneticAlgorithm
    {
    public:
      
      // function must be positive
      // f(x) > 0  with all x !
      GeneticAlgorithm(const function<T,double>& f);
      ~GeneticAlgorithm() ;

      double& getCrossover() { return p_crossover; }
      double& getMutation()  { return p_mutation; }
      
      // optimizes with given
      // population size
      bool maximize(const unsigned int numIterations,
		    const unsigned int size) ;
      bool minimize(const unsigned int numIterations,
		    const unsigned int size) ;
      
      // continues optimization
      bool continue_optimization(const unsigned int numIterations) ;
      
      // returns value of the best candidate
      // saves it to best
      double getBest(T& best) const ;
      
      // returns mean value
      double getMean() const ;
      
      bool verbosity(bool v) { verbose = v; return verbose; }
      
    private:
      
      bool create_initial_population() ;
      bool create_candidate(const std::bitset< sizeof(T)*8 >& bits,
			    T& candidate) const ;
      
      const unsigned int bits;
      function<T,double>* f;
      bool maximization_task;
      
      // probabilities
      double p_crossover;
      double p_mutation;
      
      // q[]   = two populations, current and the previous one
      // *p[0] = current population
      // *p[1] = previous population
      
      std::vector< std::bitset< sizeof(T)*8> >* p[2];
      std::vector< std::bitset< sizeof(T)*8> >  q[2];
      std::vector< double > goodness;
      
      bool verbose; // be talkative?
      
      // best candidate: (*p[1])[best_index]
      T best_candidate;
      double very_best_value;
      
      double mean_value; // current state of the learning process
    };
}
  
#include "GeneticAlgorithm.cpp"

#endif




