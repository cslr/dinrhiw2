
#include "KMeans.h"
#include "conffile.h"
#include "dinrhiw_blas.h"
#include "blade_math.h"

#include <stdio.h>

#ifndef KMeans_cpp
#define KMeans_cpp


namespace whiteice
{
  
  template <typename T>
  KMeans<T>::KMeans(T learning_rate, bool optimalmode)
  {
    kmeans.clear();
    this->learning_rate = learning_rate;
    this->goodmode = optimalmode;
  }

  template <typename T>
  KMeans<T>::KMeans(const KMeans<T>& model)
  {
    this->goodmode = model.goodmode;
    this->samplesetsize = model.samplesetsize;
    this->learning_rate = model.learning_rate;

    this->kmeans = model.kmeans;
  }
  
  template <typename T>
  KMeans<T>::~KMeans(){  }

  template <typename T>
  KMeans<T>& KMeans<T>::operator=(const KMeans<T>& model)
  {
    this->goodmode = model.goodmode;
    this->samplesetsize = model.samplesetsize;
    this->learning_rate = model.learning_rate;

    this->kmeans = model.kmeans;

    return (*this);
  }
  
  template <typename T>
  T KMeans<T>::error(const std::vector< std::vector<T> >& data) const
  {
    T e, error = T(0.0);
    
    if(data.size() <= 1000){
      for(unsigned int i=0;i<data.size();i++){
	unsigned int j = 0;
	
	e = calc_distance(kmeans[j], data[i]);
	
	for(unsigned j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[i]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(data.size());
    }
    else{
      for(unsigned int i=0;i<1000;i++){
	unsigned int index = rng.rand() % data.size();
	unsigned int j=0;
	
	e = calc_distance(kmeans[j], data[index]);
	
	for(unsigned int j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[index]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(1000.0);
    }
    
    return error;
  }
  
  
  template <typename T>
  T KMeans<T>::error(const std::vector< whiteice::math::vertex<T> >& data) const
  {
    T e, error = T(0.0);
    
    if(data.size() <= 1000){
      for(unsigned int i=0;i<data.size();i++){
	unsigned int j = 0;
	
	e = calc_distance(kmeans[j], data[i]);
	
	for(unsigned j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[i]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(data.size());
    }
    else{
      for(unsigned int i=0;i<1000;i++){
	unsigned int index = rng.rand() % data.size();
	unsigned int j=0;
	
	e = calc_distance(kmeans[j], data[index]);
	
	for(unsigned int j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[index]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(1000.0);
    }
    
    return error;
  }
  
  
  
  template <typename T>
  bool KMeans<T>::learn(unsigned int k,
			std::vector<std::vector<T> >& data) throw()
  {
    try{
      if(data.size() < 1) return false;
      kmeans.resize(k);
      
      // initialization of k-means by picking one
      // one data vector randomly
      
      for(unsigned int i=0;i<kmeans.size();i++){
	kmeans[i].resize(data[0].size());
	
	const unsigned int index = rng.rand() % data.size();
	
	for(unsigned int j=0;j<kmeans[i].size();j++){
	  kmeans[i][j] = data[index][j];
	}
      } // random initialization done

      
      unsigned int maxstep = 100*data.size();
      
      if(samplesetsize)
	maxstep = samplesetsize;
      
      
      T min_error, e;
      unsigned int winner;
      
      typename std::vector<std::vector<T> >::iterator j;
      unsigned int means_index;
      
      
      if(goodmode){
	// divides data into two equal sized sets and
	// calculates 100 iterations of kmeans
	// algorithm and then checks if results in
	// not-used set has improved (1000 sample aprox error)
	
	unsigned int learning_failures = 0;
	
	// calculates aprox error from the latter part of data
	T tr_error = second_half_error(data);
	
	while(learning_failures < 3){	  
	  
	  for(unsigned int i=0;i<100;i++){
	    // uses only the first part of data 
	    unsigned int index = rng.rand() % (data.size()/2);
	    
	    // finds closest k-mean vector
	    min_error = calc_distance(kmeans[0], data[index]);
	    winner = 0;
	    
	    j = kmeans.begin(); j++;
	    means_index = 1;
	    
	    while(j != kmeans.end()){
	      e = calc_distance(*j, data[index]);
	      
	      if(e < min_error){
		min_error = e;
		winner = means_index;
	      }
	      
	      means_index++;
	      j++;
	    }
	    
	    const unsigned int len = kmeans[winner].size();      
	    
	    // moves winner vector towards data point      
	    for(unsigned int j=0;j<len;j++){
	      kmeans[winner][j] += learning_rate *
		(data[index][j] - kmeans[winner][j]);
	    }
	  }
	  
	  T tmp = second_half_error(data);
	  if(tmp > tr_error)
	    learning_failures++;
	  else{
	    tr_error = tmp;
	    learning_failures = 0;
	  }
	  
	}
	
      }
      else{

	for(unsigned int i=0;i<maxstep;i++){
	  unsigned int index = rng.rand() % data.size();
	  
	  // finds closest k-mean vector
	  
	  min_error = calc_distance(kmeans[0], data[index]);
	  winner = 0;
	  
	  j = kmeans.begin(); j++;
	  means_index = 1;
	  
	  while(j != kmeans.end()){
	    e = calc_distance(*j, data[index]);
	    
	    if(e < min_error){
	      min_error = e;
	      winner = means_index;
	    }
	    
	    means_index++;
	    j++;
	  }
	
	  const unsigned int len = kmeans[winner].size();      
	  
	  // moves winner vector towards data point      
	  for(unsigned int j=0;j<len;j++){
	    kmeans[winner][j] += learning_rate *
	      (data[index][j] - kmeans[winner][j]);
	  }
	  
	}	
	
      }
      
      return true;
    }
    catch(std::exception& e){
      return false;
    }
  }
  
  
  
  template <typename T>
  bool KMeans<T>::learn(unsigned int k, std::vector< whiteice::math::vertex<T> >& data) throw()
  {
    try{
      if(data.size() < 1) return false;
      kmeans.resize(k);

      // initialization of k-means by picking one
      // one data vector randomly
      
      for(unsigned int i=0;i<kmeans.size();i++){
	kmeans[i].resize(data[0].size());

	const unsigned int index = rng.rand() % data.size();

	for(unsigned int j=0;j<kmeans[i].size();j++){
	  kmeans[i][j] = data[index][j];
	}
      } // random initialization done

      
      unsigned int maxstep = 100*data.size();
      
      if(samplesetsize)
	maxstep = samplesetsize;
      
      
      T min_error, e;
      unsigned int winner;
      
      typename std::vector<std::vector<T> >::iterator j;
      unsigned int means_index;
      
      
      if(goodmode){
	// divides data into two equal sized sets and
	// calculates 100 iterations of kmeans
	// algorithm and then checks if results in
	// not-used set has improved (1000 sample aprox error)
	
	unsigned int learning_failures = 0;
	T tr_error = second_half_error(data); // calculates aprox error from the latter part of data
	
	while(learning_failures < 3){	  
	  
	  for(unsigned int i=0;i<100;i++){
	    // uses the first part of data 
	    unsigned int index = rng.rand() % (data.size()/2);
	    
	    // finds closest k-mean vector
	    min_error = calc_distance(kmeans[0], data[index]);
	    winner = 0;
	    
	    j = kmeans.begin(); j++;
	    means_index = 1;
	    
	    while(j != kmeans.end()){
	      e = calc_distance(*j, data[index]);
	      
	      if(e < min_error){
		min_error = e;
		winner = means_index;
	      }
	      
	      means_index++;
	      j++;
	    }
	    
	    const unsigned int len = kmeans[winner].size();      
	    
	    // moves winner vector towards data point      
	    for(unsigned int j=0;j<len;j++){
	      kmeans[winner][j] += learning_rate *
		(data[index][j] - kmeans[winner][j]);
	    }
	  }
	  
	  T tmp = second_half_error(data);
	  if(tmp > tr_error)
	    learning_failures++;
	  else{
	    tr_error = tmp;
	    learning_failures = 0;
	  }
	  
	}
	
      }
      else{

	for(unsigned int i=0;i<maxstep;i++){
	  unsigned int index = rng.rand() % data.size();
	  
	  // finds closest k-mean vector
	  
	  min_error = calc_distance(kmeans[0], data[index]);
	  winner = 0;
	  
	  j = kmeans.begin(); j++;
	  means_index = 1;
	  
	  while(j != kmeans.end()){
	    e = calc_distance(*j, data[index]);
	    
	    if(e < min_error){
	      min_error = e;
	      winner = means_index;
	    }
	    
	    means_index++;
	    j++;
	  }
	
	  const unsigned int len = kmeans[winner].size();      
	  
	  // moves winner vector towards data point      
	  for(unsigned int j=0;j<len;j++){
	    kmeans[winner][j] += learning_rate *
	      (data[index][j] - kmeans[winner][j]);
	  }
	  
	}	
	
      }
      
      return true;
    }
    catch(std::exception& e){
      return false;
    }
  }
  
  
  template <typename T>
  unsigned int KMeans<T>::size() const throw()
  {
    return kmeans.size();
  }
  
  
  template <typename T>
  std::vector<T>& KMeans<T>::operator[](unsigned int index)
  {
    return kmeans[index];
  }
  
  
  template <typename T>
  const std::vector<T>& KMeans<T>::operator[](unsigned int index) const
  {
    return kmeans[index];
  }

  template <typename T>
  unsigned int KMeans<T>::getClusterIndex(const whiteice::math::vertex<T>& x) const
    throw(std::logic_error)
  {
    if(kmeans.size() <= 0)
      throw std::logic_error("KMeans: No clustering available");

    T best_distance = T(INFINITY);
    unsigned int best_index = 0;

    for(unsigned int i=0;i<kmeans.size();i++){

      T distance = T(0.0);
      
      for(unsigned int j=0;j<kmeans[i].size();j++){
	distance += (x[j] - kmeans[i][j])*(x[j] - kmeans[i][j]);
      }

      if(distance < best_distance){
	best_distance = distance;
	best_index = i;
      }
    }

    return best_index;
  }
  
  template <typename T>
  unsigned int KMeans<T>::getClusterIndex(const std::vector<T>& x) const
    throw(std::logic_error)
  {
    if(kmeans.size() <= 0)
      throw std::logic_error("KMeans: No clustering available");

    T best_distance = T(INFINITY);
    unsigned int best_index = 0;

    for(unsigned int i=0;i<kmeans.size();i++){

      T distance = T(0.0);
      
      for(unsigned int j=0;j<kmeans[i].size();j++){
	distance += (x[j] - kmeans[i][j])*(x[j] - kmeans[i][j]);
      }

      if(distance < best_distance){
	best_distance = distance;
	best_index = i;
      }
    }

    return best_index;
  }
  
  // calculates squared distance
  template <typename T>
  T KMeans<T>::calc_distance(const std::vector<T>& u,
			     const std::vector<T>& v) const
  {
    unsigned int i;
    const unsigned int max = u.size();
    T e, error = T(0.0);
    
    for(i=0;i<max;i++){
      e = (u[i] - v[i]);
      error += e*e;
    }
    
    return error;
  }
  
  
  template <typename T>
  T KMeans<T>::calc_distance(const std::vector<T>& u,
			     const whiteice::math::vertex<T>& v) const
  {
    unsigned int i;
    const unsigned int max = u.size();
    T e, error = T(0.0);
    
    for(i=0;i<max;i++){
      e = (u[i] - v[i]);
      error += e*e;
    }
    
    return error;
  }
  
  
  template <typename T>
  bool KMeans<T>::save(const std::string& filename) throw()
  {
    whiteice::conffile configuration;
    std::vector<int> ints;
    std::vector<float> floats;
    std::vector<std::string> strings;
    
    if(kmeans.size() <= 0) return false;
    
    // version number
    ints.clear();
    ints.push_back(1);
    if(!configuration.set("KMEANS_VERSION", ints)) return false;

    const unsigned int dimension = kmeans[0].size();
    
    // number of vectors and their dimensions    
    ints.clear();
    ints.push_back(kmeans.size()); // K
    ints.push_back(dimension); // dimension
    if(!configuration.set("KMEANS_SIZES", ints)) return false;
    
    // sets kmeans vector data
    char buf[100];    
    floats.resize(dimension);
    
    unsigned int counter = 0;
    typename std::vector<std::vector<T> >::iterator i;
    typename std::vector<T>::iterator j;
    std::vector<float>::iterator l;
    
    for(i=kmeans.begin();i!=kmeans.end();i++, counter++){
      sprintf(buf, "KMEANS_VECTOR%d", counter);            
      
      l = floats.begin();
      for(j = i->begin();j!=i->end();j++,l++)
	math::convert( (*l), (*j) );
      
      
      if(!configuration.set(buf,floats)){
	return false;
      }
    }
    
    return configuration.save(filename);
  }
  

  // loads k-means clustering data from a file
  template <typename T>
  bool KMeans<T>::load(const std::string& filename) throw()
  {
    whiteice::conffile configuration;
    std::vector<int> ints;
    std::vector<float> floats;
    std::vector<std::string> strings;
    
    kmeans.clear();
    
    if(!configuration.load(filename))
      return false;

    // version number
    ints.clear();
    if(!configuration.get("KMEANS_VERSION", ints)) return false;
    if(ints.size() != 1) return false;
    
    // version number check
    if(ints[0] != 1) return false;

    
    // number of vectors and their dimensions        
    ints.clear();
    if(!configuration.get("KMEANS_SIZES", ints)) return false;
    if(ints.size() != 2) return false;
    
    kmeans.resize(ints[0]); // K
    for(unsigned int i=0;i<kmeans.size();i++)
      kmeans[i].resize(ints[1]);
    
    // sets kmeans vector data
    char buf[100];    
    floats.clear();
    
    unsigned int counter = 0;
    typename std::vector<std::vector<T> >::iterator i;
    typename std::vector<T>::iterator j;
    std::vector<float>::iterator l;
    
    for(i=kmeans.begin();i!=kmeans.end();i++, counter++){
      sprintf(buf, "KMEANS_VECTOR%d", counter);
      
      if(!configuration.get(buf, floats)){	
	return false;
      }
      
      j = i->begin();
      for(l = floats.begin();l!=floats.end();j++,l++){
	*j = T(*l);
      }
    }
    
    return true;
  }
  
  
  
  // calculates aprox error from the latter part of data
  template <typename T>    
  T KMeans<T>::second_half_error(const std::vector< std::vector<T> >& data) const
  {
    T e, error = T(0.0);
    
    if(data.size() <= 1000){
      for(unsigned int i=(data.size()/2);i<data.size();i++){
	unsigned int j = 0;
	
	e = calc_distance(kmeans[j], data[i]);
	
	for(unsigned j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[i]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(data.size());
    }
    else{
      for(unsigned int i=0;i<1000;i++){
	unsigned int index = (rng.rand() % (data.size()/2)) + (data.size()/2);
	unsigned int j=0;
	
	e = calc_distance(kmeans[j], data[index]);
	
	for(unsigned int j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[index]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(1000.0);
    }
    
    return error;
  }
  
  
  // calculates aprox error from the latter part of data
  template <typename T>    
  T KMeans<T>::second_half_error(const std::vector< whiteice::math::vertex<T> >& data) const
  {
    T e, error = T(0.0);
    
    if(data.size() <= 1000){
      for(unsigned int i=(data.size()/2);i<data.size();i++){
	unsigned int j = 0;
	
	e = calc_distance(kmeans[j], data[i]);
	
	for(unsigned j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[i]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(data.size());
    }
    else{
      for(unsigned int i=0;i<1000;i++){
	unsigned int index = (rng.rand() % (data.size()/2)) + (data.size()/2);
	unsigned int j=0;
	
	e = calc_distance(kmeans[j], data[index]);
	
	for(unsigned int j=1;j<kmeans.size();j++){
	  T tmp = calc_distance(kmeans[j], data[index]);
	  if(tmp < e)
	    e = tmp;
	}
	
	error += e;
      }
      
      error /= T(1000.0);
    }
    
    return error;
  }

  
  
  
  
  template <typename T>
  T& KMeans<T>::rate() throw(){
    return learning_rate;
  }
  
  
  template <typename T>
  const T& KMeans<T>::rate() const throw(){
    return learning_rate;
  }
  
  
  template <typename T>
  bool& KMeans<T>::getOptimalMode() throw(){
    return goodmode;
  }
  
  
  template <typename T>
  const bool& KMeans<T>::getOptimalMode() const throw(){
    return goodmode;
  }
  
  
  template <typename T>
  unsigned int& KMeans<T>::numSamplingSteps() throw(){
    return samplesetsize;
  }
  
  
  template <typename T>
  const unsigned int& KMeans<T>::numSamplingSteps() const throw(){
    return samplesetsize;
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class KMeans< float >;  
  template class KMeans< double >;  
  template class KMeans< math::blas_real<float> >;
  template class KMeans< math::blas_real<double> >;
  
}
  
#endif



