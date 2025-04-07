/*
 * Iterated Prisoners Dilemma Evolution Strategies optimization of population strategies
 *
 * Tomas Ukkonen 2025
 *
 */

#ifndef whiteice_IPD_ES_h
#define whiteice_IPD_ES_h

#include "EvolutionStrategies.h"
#include "vertex.h"
#include "nnetwork.h"

namespace whiteice
{

  template <typename T = math::blas_real<float> >
  class IPD_ES : public whiteice::EvolutionStrategies<T>
  {
  public:
    
    IPD_ES(const unsigned int TURNS = 200, const unsigned int HISTORY_LENGTH=10);
    ~IPD_ES();

    
    virtual bool estimateReward(const math::vertex<T>& x,
				const std::vector< math::vertex<T> >& population,
				T& reward) const; // reward must be positive number >= 0

    virtual bool estimateMeanRewardReference(const math::vertex<T>& x, // mean reward against reference good standard solution
					     T& reward) const;

    virtual unsigned int PARAMETER_DIMENSIONS() const; // number of parameters in the model
    
  private:

    T game_ipd(const math::vertex<T>& x, const math::vertex<T>& y) const;

    T game_ipd_generous_tit_for_tat(const math::vertex<T>& x) const;

    mutable std::mutex model_mutex;
    whiteice::nnetwork<T> model; // used to calculate responses given

    const unsigned int TURNS;
    const unsigned int HISTORY_LENGTH;
    
  };
  

  extern template class IPD_ES< math::blas_real<float> >;
  extern template class IPD_ES< math::blas_real<double> >;
};


#endif



