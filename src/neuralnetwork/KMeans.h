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

#include <thread>
#include <mutex>
#include <exception>
#include <stdexcept>

#include "RNG.h"

namespace whiteice
{
  
  // T must support basic + - * / operations
  template <typename T=whiteice::math::blas_real<float> >
  class KMeans
  {
  public:
    KMeans(T learning_rate=T(0.1), // TODO: good default learning rate?
	   bool optimalmode = false);
    KMeans(const KMeans<T>& model);
    ~KMeans();

    KMeans<T>& operator=(const KMeans<T>& model);

    void randomize(std::vector<std::vector<T> >& kmeans,
		   const std::vector< whiteice::math::vertex<T> >& data) const;

    void randomize(std::vector< whiteice::math::vertex<T> >& kmeans,
		   const std::vector< whiteice::math::vertex<T> >& data) const;
    
    bool learn(const unsigned int k,
	       std::vector< std::vector<T> >& data) ;
    
    bool learn(const unsigned int k,
	       std::vector< whiteice::math::vertex<T> >& data) ;

    /**
     * starts internal thread for computing the results
     */
    bool startTrain(const unsigned int K,
		    std::vector< whiteice::math::vertex<T> >& data);

    bool startTrain(const unsigned int K,
		    std::vector< std::vector<T> >& data);

    // returns true if optimizer thread is running
    bool isRunning();

    // returns current solution error
    double getSolutionError();

    // stops training K-means clustering
    bool stopTrain();
    
    // calculates approximative error
    // (samples N=1000 samples or uses everything)
    T error(const std::vector< std::vector<T> >& data) const;    
    T error(const std::vector< whiteice::math::vertex<T> >& data) const;
    
    T error(const std::vector< whiteice::math::vertex<T> >& kmeans,
	    const std::vector< whiteice::math::vertex<T> >& data) const;
    
    // returns index:th cluster mean value
    math::vertex<T>& operator[](unsigned int index);
    const math::vertex<T>& operator[](unsigned int index) const;

    // clusterizes given data vector, returns cluster index
    unsigned int getClusterIndex(const whiteice::math::vertex<T>& x) const;
    unsigned int getClusterIndex(const std::vector<T>& x) const;
    
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
    T calc_distance(const whiteice::math::vertex<T>& u, const whiteice::math::vertex<T>& v) const;
    
    T second_half_error(const std::vector< std::vector<T> >& data) const;
    T second_half_error(const std::vector< whiteice::math::vertex<T> >& data) const;

    bool means_changed(const std::vector< math::vertex<T> >& means1,
		       const std::vector< math::vertex<T> >& means2) const;

    //////////////////////////////////////////////////////////////////////
    // Internal threading variables

    void optimizer_loop();

    std::thread* optimizer_thread = nullptr;
    mutable std::mutex thread_mutex, solution_mutex;
    bool thread_running = false, computing_stopped = true;
    bool verbose = false;

    std::vector<math::vertex<T> > best_kmeans;
    double best_error = INFINITY;

    std::vector< math::vertex<T> > data;

    //////////////////////////////////////////////////////////////////////
    
    
    bool goodmode; // keep going till results improve,
                   // uses early stopping to prevent overfitting
    unsigned int samplesetsize;
    
    std::vector< math::vertex<T> > kmeans;
    T learning_rate;

    whiteice::RNG<T> rng;
  };
  
  
  extern template class KMeans< float >;  
  extern template class KMeans< double >;  
  extern template class KMeans< math::blas_real<float> >;
  extern template class KMeans< math::blas_real<double> >;
  
};



#endif
