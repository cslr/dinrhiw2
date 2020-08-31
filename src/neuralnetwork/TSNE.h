/*
 * t-SNE dimension reduction algorithm.
 * Based on paper 
 * "Visualization Data using t-SNE. Laurens van der Maaten and Geoffrey Hinton. 11/2008".
 *
 * Mathmatical implementation notes/documentation in docs/TSNE_notes.tm (texmacs)
 * Tomas Ukkonen. 2020.
 * 
 */

#ifndef TSNE_h
#define TSNE_h

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "RNG.h"
#include "LoggingInterface.h"
#include "VisualizationInterface.h"

#include <vector>

namespace whiteice
{
  
  template <typename T = math::blas_real<float> >
    class TSNE
    {
    public:

      TSNE(const bool absolute_value = true);
      TSNE(const TSNE<T>& tsne);
      
      // dimension reduces samples to DIM dimensional vectors using t-SNE algorithm
      bool calculate(const std::vector< math::vertex<T> >& samples,
		     const unsigned int DIM,
		     std::vector< math::vertex<T> >& results,
		     const bool verbose = false,
		     LoggingInterface* const messages = NULL,
		     VisualizationInterface* const gui = NULL);
		     
      
    private:
      
      // calculates p values for pj|i where i = index and sigma2 for index:th vector is given
      bool calculate_pvalue_given_sigma(const std::vector< math::vertex<T> >& x,
					const unsigned int index, // to x vector
					const T sigma2,
					std::vector<T>& pj) const;
      
      // calculates distribution's perplexity
      T calculate_perplexity(const std::vector<T>& pj) const;
      
      // calculate x samples probability distribution values p
      bool calculate_pvalues(const std::vector< math::vertex<T> >& x,
			     const T perplexity,
			     std::vector< std::vector<T> >& pij) const;
      
      // calculate dimension reduced y samples probability distribution values q
      bool calculate_qvalues(const std::vector< math::vertex<T> >& y,
			     std::vector< std::vector<T> >& qij,
			     T& qsum) const;
      
      // calculates KL divergence
      bool kl_divergence(const std::vector< std::vector<T> >& pij,
			 const std::vector< std::vector<T> >& qij,
			 T& klvalue) const;
      
      // calculates gradients of KL diverence
      bool kl_gradient(const std::vector< std::vector<T> >& pij,
		       const std::vector< std::vector<T> >& qij,
		       const T& qsum,
		       const std::vector< math::vertex<T> >& y,
		       std::vector< math::vertex<T> >& ygrad) const;

      
      ////////////////////////////////////////////////////////////
      // internal variables

      bool kl_absolute_value; // whether to use absolute value in KL divergence
    
    };


  extern template class TSNE< math::blas_real<float> >;
  extern template class TSNE< math::blas_real<double> >;
  
};


#endif
