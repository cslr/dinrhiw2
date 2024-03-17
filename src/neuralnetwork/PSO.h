/*
 * canonical global PSO (particle swam optimizer)
 * with function stretching heuristics
 *
 * todo: local PSO (should be better)
 */


#include "vertex.h"
#include "optimized_function.h"

#ifndef PSO_h
#define PSO_h

namespace whiteice
{
	template <typename T = math::blas_real<float> >
    class PSO {
    	public:

		struct range;
		struct particle;

		// f - optimized function
		// range of input values should be within range 0..1
		// (initial values are within range 0..1)
		PSO(const optimized_function<T>& f);

		// range of input values
		PSO(const optimized_function<T>& f,
				std::vector<PSO<T>::range> r);
		~PSO() ;

		struct range{ T min; T max; };


		struct particle
		{
			math::vertex<T> value;
			math::vertex<T> velocity;
			T fitness; // current fitness

			math::vertex<T> best;
			T best_fitness;
		};


		// iterations and swarm size
		bool maximize(const unsigned int numIterations,
				const unsigned int size) ;
      
		bool minimize(const unsigned int numIterations,
				const unsigned int size) ;
      
		// with custom starting population
		bool maximize(const unsigned int numIterations,
				const std::vector< math::vertex<T> >& data) ;

		// with custom starting population
		bool minimize(const unsigned int numIterations,
				const std::vector< math::vertex<T> >& data) ;

		// continues old optimization task with
		// additional 'numIterations'
		bool improve(const unsigned int numIterations) ;

		// returns best candidate found so far
		void getCurrentBest(math::vertex<T>& best) const ;

		// returns best candidate found so far
		void getBest(math::vertex<T>& best) const ;

		// samples vector from swarm according
		// to goodness of swarm particles
		const typename PSO<T>::particle& sample();

		// swarm size
		unsigned int size() const  PURE_FUNCTION;

		bool verbosity(bool v) {
			verbose = v;
			return verbose;
		}

    	private:
		optimized_function<T>* f;
		bool maximization_task;

		bool create_initial_population(const unsigned int size);

		// creates initial particles from given data
		bool setup_population(const std::vector< math::vertex<T> >& data) ;


		// optimization with given swarm [real stuff is here]
		bool continue_optimization(const unsigned int numIterations) ;

		// helper function - calculates percentage difference
		T percentage_change(const math::vertex<T>& old_best,
				const math::vertex<T>& new_best) const;

		void clamp_values_to_range(PSO<T>::particle& p) const ;


		T c1, c2; // parameters

		std::vector<typename PSO<T>::particle> swarm;
		typename PSO<T>::particle global_best;
		typename PSO<T>::particle old_best;

		// the very best value ever seen by PSO
		// (preserved over restarts etc.)
		typename PSO<T>::particle total_best;

		std::vector<range> datarange; // data ranges for input data

		unsigned int global_iter;
		bool first_time;
		bool verbose;


		// sampling variables
		std::vector<T> cumdistrib;
		T cumdistrib_maxvalue;

    };


	extern template class PSO<float>;
	extern template class PSO<double>;
	extern template class PSO< math::blas_real<float> >;
	extern template class PSO< math::blas_real<double> >;
	
};





#endif
