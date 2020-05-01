/*
 * normal k-means clustering implementation
 *
 * TODO: try to find the "enchanced k-means clustering paper"
 * mentioned in Haykin's book which should guarantee near
 * optimal clustering (in terms of data representation errors)
 *
 */

#include <vector>
#include <string>
#include "dinrhiw_blas.h"
#include "vertex.h"


#ifndef KMeans_h
#define KMeans_h

#include <exception>
#include <stdexcept>

#include "RNG.h"

namespace whiteice
{
  
  // T must support basic + - * / operations
  template <typename T>
  class KMeans
  {
  public:
    KMeans(T learning_rate=T(0.1), // TODO: good default learning rate?
	   bool optimalmode = false);
    KMeans(const KMeans<T>& model);
    ~KMeans();

    KMeans<T>& operator=(const KMeans<T>& model);
    
    bool learn(unsigned int k,
	       std::vector< std::vector<T> >& data) ;
    
    bool learn(unsigned int k,
	       std::vector< whiteice::math::vertex<T> >& data) ;
    
    // calculates approximative error
    // (samples N=1000 samples or uses everything)
    T error(const std::vector< std::vector<T> >& data) const;    
    T error(const std::vector< whiteice::math::vertex<T> >& data) const;
    
    
    std::vector<T>& operator[](unsigned int index);
    const std::vector<T>& operator[](unsigned int index) const;

    unsigned int getClusterIndex(const whiteice::math::vertex<T>& x) const
      ;
    unsigned int getClusterIndex(const std::vector<T>& x) const
      ;
    
    // number of clusters
    unsigned int size() const ;
    
    // reads/saves k-means clustering to a file
    bool save(const std::string& filename) ;
    bool load(const std::string& filename) ;
    
    T& rate() ;
    const T& rate() const ;
    
    bool& getOptimalMode() ;
    const bool& getOptimalMode() const ;
    
    // number of times samples are picked and used
    // when running k-means algorithm
    unsigned int& numSamplingSteps() ;
    const unsigned int& numSamplingSteps() const ;
    
  private:
    T calc_distance(const std::vector<T>& u, const std::vector<T>& v) const;
    T calc_distance(const std::vector<T>& u, const whiteice::math::vertex<T>& v) const;
    
    T second_half_error(const std::vector< std::vector<T> >& data) const;
    T second_half_error(const std::vector< whiteice::math::vertex<T> >& data) const;
    
    
    bool goodmode; // keep going till results improve,
                   // uses early stopping to prevent overfitting
    unsigned int samplesetsize;
    
    std::vector<std::vector<T> > kmeans;
    T learning_rate;

    whiteice::RNG<T> rng;
  };
  
  
  extern template class KMeans< float >;  
  extern template class KMeans< double >;  
  extern template class KMeans< math::blas_real<float> >;
  extern template class KMeans< math::blas_real<double> >;
  
};



#endif
