/*
 * t-SNE dimension reduction algorithm.
 * Mathmatical implementation notes/documentation in docs/TSNE_notes.tm (texmacs)
 * Tomas Ukkonen. 2020.
 * 
 */

#ifndef TSNE_h
#define TSNE_h

#include "dinrhiw_blas.h"
#include "vertex.h"

#include <vector>

namespace whiteice
{
  
  template <typename T = math::blas_real<float> >
    class TSNE
    {
    public:

    TSNE();
    TSNE(const TSNE<T>& tsne);

    // dimension reduces samples to DIM dimensional vectors using t-SNE algorithm
    bool calculate(const std::vector< math::vertex<T> >& samples,
		   const unsigned int DIM,
		   std::vector< math::vertex<T> >& results);

    private:
    
    
    };


  extern template class TSNE< math::blas_real<float> >;
  extern template class TSNE< math::blas_real<double> >;
  
};


#endif
