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
 */

#ifndef __whiteice__RNN_RBM_h
#define __whiteice__RNN_RBM_h

#include <vector>

#include "vertex.h"
#include "nnetwork.h"
#include "BBRBM.h"

namespace whiteice
{

  template <typename T = whiteice::math::blas_real<float> >
    class RNN_RBM
    {
    public:
      RNN_RBM(unsigned int dimVisible,
	      unsigned int dimHidden,
	      unsigned int dimRecurrent);

      ~RNN_RBM();

      // optimizes data likelihood using N-timseries,
      // which are i step long and have dimVisible elements e
      // timeseries[N][i][e]
      bool optimize(const std::vector< std::vector< whiteice::math::vertex<T> > >& timeseries);

      // resets timeseries synthetization parameters
      void synthStart();

      // synthesizes next timestep by using the model
      bool synthNext(whiteice::math::vertex<T>& vnext);

      // synthesizes N next candidates using the probabilistic model
      bool synthNext(unsigned int N, std::vector< whiteice::math::vertex<T> >& vnext);

      // selects given v as the next step in time-series
      // (needed to be called before calling again synthNext())
      bool synthSetNext(whiteice::math::vertex<T>& v);
      

    protected:
      unsigned int dimVisible;
      unsigned int dimHidden;
      unsigned int dimRecurrent;

      whiteice::nnetwork<T> nn; // recurrent neural network
      whiteice::BBRBM<T> rbm;   // rbm part

      T reconstructionError(const std::vector< std::vector< whiteice::math::vertex<T> > >& timeseries);
      
    };
  
  
  extern template class RNN_RBM< whiteice::math::blas_real<float> >;
  extern template class RNN_RBM< whiteice::math::blas_real<double> >;
  
};


#endif
