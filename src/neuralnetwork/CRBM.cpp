
#include "CRBM.h"
#include "norms.h"
#include <random>

namespace whiteice
{
  
  // creates 1x1 network, used to load some useful network later
  template <typename T>
  CRBM<T>::CRBM()
  {
    v.resize(2);
    h.resize(2);
    W.resize(2,2);
    
    initializeWeights();
  }
  
  template <typename T>
  CRBM<T>::CRBM(const CRBM<T>& rbm)
  {
    this->v = rbm.v;
    this->h = rbm.h;
    this->W = rbm.W;
  }
  
  
  // creates 2-layer: V * H network
  template <typename T>
  CRBM<T>::CRBM(unsigned int visible, unsigned int hidden)
    
  {
    if(visible == 0 || hidden == 0)
      throw std::invalid_argument("invalid network architecture");
    
    // the last term is always constant: 1 (one)
    v.resize(visible + 1);
    h.resize(hidden + 1);
    W.resize(hidden + 1, visible + 1);
    
    initializeWeights();    
  }
  
  template <typename T>
  CRBM<T>::~CRBM()
  {
    // nothing to do
  }
  
  template <typename T>  
  CRBM<T>& CRBM<T>::operator=(const CRBM<T>& rbm)
  {
    this->v = rbm.v;
    this->h = rbm.h;
    this->W = rbm.W;
    
    return (*this);
  }
  
  template <typename T>
  math::vertex<T> CRBM<T>::getVisible() const
  {
    math::vertex<T> t = v;
    t.resize(v.size() - 1);
    
    return t;
  }
  
  template <typename T>
  bool CRBM<T>::setVisible(const math::vertex<T>& v)
  {
    if(this->v.size() != (v.size() + 1))
      return false;
    
    for(unsigned int j=0;j<v.size();j++)
      this->v[j] = v[j];
    
    this->v[v.size()] = T(1.0);
    
    return true;
  }
  
  template <typename T>
  math::vertex<T> CRBM<T>::getHidden() const
  {
    math::vertex<T> t = h;
    t.resize(h.size() - 1);
    
    return t;
  }
  
  
  template <typename T>
  bool CRBM<T>::setHidden(const math::vertex<T>& h)
  {
    if(this->h.size() != (h.size() + 1))
      return false;
    
    for(unsigned int j=0;j<h.size();j++)
      this->h[j] = h[j];
    
    this->h[h.size()] = T(1.0);
    
    return true;
  }
  
  
  template <typename T>
  bool CRBM<T>::reconstructData(unsigned int iters)
  {
    if(iters == 0) return false;
    
    math::matrix<T> Wt = W;
    Wt.transpose();
    
    std::default_random_engine gen;
    std::normal_distribution<double> normal_rng(0.0, 0.01); // N(0,1);
    
    while(iters > 0){
      h = W*v;
      
      // 1. hidden units: calculates sigma(a_j)
      for(unsigned int j=0;j<(h.size()-0);j++){
	T aj = T(2.0)/(T(1.0) + math::exp(-h[j])) - T(1.0);
	// T r = T(rand())/T((float)RAND_MAX);
	
	T r = T((float)rand())/T((float)RAND_MAX);
	if(aj > r) h[j] = T(1.0); // discretization step
	else       h[j] = T(0.0);
	
	h[j] = aj;
      }
      
      h[h.size()-1] = T(1.0); // bias term is always one
      
      iters--;
      if(iters <= 0) return true;
    
      v = Wt*h;
      
      // 1. visible units: calculates sigma(a_j)
      for(unsigned int j=0;j<(v.size()-0);j++){
	T aj = T(2.0)/(T(1.0) + math::exp(-v[j])) - T(1.0);
	// T r = T(rand())/T(RAND_MAX);
	
	//if(aj > r) v[j] = T(1.0); // discretization step
	//else       v[j] = T(0.0);
	
	v[j] = aj;
      }
      
      v[v.size()-1] = T(1.0); // bias term is always one
      
      iters--;
      if(iters <= 0) return true;
    }
    
    return true;
  }
  
  
  template <typename T>
  math::matrix<T> CRBM<T>::getWeights() const
  {
    return W;
  }
  
  template <typename T>
  bool CRBM<T>::initializeWeights(){ // initialize weights to small values
    
    for(unsigned int j=0;j<W.ysize();j++)
      for(unsigned int i=0;i<W.xsize();i++)
	W(j,i) = randomValue();
    
    v[v.size()-1] = T(1.0);
    h[h.size()-1] = T(1.0);

    return true;
  }
  
  
  template <typename T>
  T CRBM<T>::learnWeights(const std::vector< math::vertex<T> >& samples)
  {
    const unsigned int L = 10; // CD-10 algorithm
    const T lambda = T(0.01);
    
    math::matrix<T> P(W);
    math::matrix<T> N(W);
    math::matrix<T> originalW = W;
    math::matrix<T> Wt = W;
    Wt.transpose();
    
    std::default_random_engine gen;
    std::normal_distribution<double> normal_rng(0.0, 0.01); // N(0,0.01);
    
    
    for(unsigned int i=0;i<samples.size();i++){
      const unsigned int index = rand() % samples.size();
      
      if(samples[index].size() + 1 != v.size())
	throw std::invalid_argument("CRBM: Input data has incorrect dimension.");

      // v = samples[index];
      v.importData(&(samples[index][0]), samples[index].size());
      v[v.size()-1] = T(1.0);
      
      
      {
	h = W*v;
	
	// 1. hidden units: calculates sigma(a_j)
	for(unsigned int j=0;j<(h.size()-0);j++){
	  T aj = T(2.0)/(T(1.0) + math::exp(-h[j] + T(normal_rng(gen)))) - T(1.0); // [-1, 1]
	  
#if 1
	  T r = T((float)rand())/T((float)RAND_MAX);
	  if(aj > r) h[j] = T(1.0); // discretization step
	  else       h[j] = T(0.0);
#else
	  h[j] = aj;
#endif
	}
	
	h[h.size()-1] = T(1.0);
	
	for(unsigned int y=0;y<h.size();y++)
	  for(unsigned int x=0;x<v.size();x++)
	    P(y,x) = h[y]*v[x];

	
	for(unsigned int l=0;l<L;l++){
	  v = Wt*h;
	  
	  // 1. visible units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(v.size()-0);j++){
	    T aj = T(2.0)/(T(1.0) + math::exp(-v[j] + T(normal_rng(gen)))) - T(1.0); // [-1, 1]
	    
#if 0
	    T r = T((float)rand())/T((float)RAND_MAX);	    
	    if(aj > r) v[j] = T(1.0); // discretization step
	    else       v[j] = T(0.0);
#else
	    v[j] = aj;
#endif
	  }
	  
	  v[v.size()-1] = T(1.0);
	  
	  
	  h = W*v;
	  
	  // 2. hidden units: calculates sigma(a_j)
	  for(unsigned int j=0;j<(h.size()-0);j++){
	    T aj = T(2.0)/(T(1.0) + math::exp(-h[j] + T(normal_rng(gen)))) - T(1.0); // [-1, 1]
	    
#if 1
	    T r = T((float)rand())/T((float)RAND_MAX);
	    if(aj > r) h[j] = T(1.0); // discretization step
	    else       h[j] = T(0.0);
#else
	    h[j] = aj;
#endif
	  }
	  
	  h[h.size()-1] = T(1.0);
	  
	}
	
	
	for(unsigned int y=0;y<h.size();y++)
	  for(unsigned int x=0;x<v.size();x++)
	    N(y,x) = h[y]*v[x];
      }
      
      // updates weights according to CD rule
      W += lambda*(P - N);
      Wt = W;
      Wt.transpose();
    }
    
    
    originalW -= W;
    
    return (math::frobenius_norm(originalW)/originalW.size());
  }
  
  ////////////////////////////////////////////////////////////
  
#define CRBM_VERSION_CFGSTR          "CRBM_CONFIG_VERSION"
#define CRBM_ARCH_CFGSTR             "CRBM_ARCH"
#define CRBM_WEIGHTS_CFGSTR          "CRBM_WEIGHTS"
  
  template <typename T>
  bool CRBM<T>::load(const std::string& filename) 
  {
    try{
      whiteice::conffile configuration;
      
      std::vector<std::string> strings;
      std::vector<float> floats;
      std::vector<int> ints;
      
      if(!configuration.load(filename))
	return false;
      
      // checks version
      {
	int versionid = 0;
	
	if(!configuration.get(CRBM_VERSION_CFGSTR, ints))
	  return false;
	
	if(ints.size() != 1)
	  return false;
	
	versionid = ints[0];
	
	ints.clear();
	
	if(versionid != 1000) // v1.0 datafile
	  return false;
      }
      
      // loads architecture
      std::vector<int> arch;
      {
	if(!configuration.get(CRBM_ARCH_CFGSTR,ints))
	  return false;
	
	if(ints.size() < 2)
	  return false;
	
	for(unsigned int i=0;i<ints.size();i++)
	  if(ints[i] <= 0) return false;
	
	arch = ints;
      }
      
      // tries to load weight matrix V
      math::matrix<T> V;
      {
	if(!configuration.get(CRBM_WEIGHTS_CFGSTR, floats))
	  return false;
	
	if(floats.size() != (unsigned int)(arch[0]*arch[1]))
	  return false;
	
	V.resize(arch[1], arch[0]);
	
	for(unsigned int i=0;i<(unsigned int)(arch[1]*arch[0]);i++)
	  V[i] = T(floats[i]);
      }
      
      
      // if everything went ok then activate the changes
      v.resize(arch[0]);
      h.resize(arch[1]);
      W = V;
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception: "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      return false;
    }
  }
  
  template <typename T>
  bool CRBM<T>::save(const std::string& filename) const 
  {
    try{
      whiteice::conffile configuration;
      
      std::vector<std::string> strings;
      std::vector<float> floats;
      std::vector<int> ints;
      
      // sets version
      {
	int versionid = 1000; // v1.0 datafile
	
	ints.clear();
	ints.push_back(versionid);
	
	if(!configuration.set(CRBM_VERSION_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }
      
      // sets architecture
      {
	ints.clear();
	ints.push_back(v.size());
	ints.push_back(h.size());
	
	if(!configuration.set(CRBM_ARCH_CFGSTR,ints))
	  return false;
	
	ints.clear();
      }
      
      // sets weight matrix W
      {
	floats.clear();
	floats.resize(v.size()*h.size());

	for(unsigned int i=0;i<(unsigned int)(v.size()*h.size());i++){
	  float f;
	  if(math::convert(f, W[i]) == false)
	    return false; // cannot convert to floats meaningfully => failure in save
	  floats[i] = f;
	}
	
	if(!configuration.set(CRBM_WEIGHTS_CFGSTR, floats))
	  return false;
      }
      
      // if everything went ok saves the conf file to disk
      if(!configuration.save(filename))
	return false;
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception: "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      return false;
    }
  }
  
  
  template <typename T>
  T CRBM<T>::randomValue()
  {
    T r = T((float)rand())/T((float)RAND_MAX);
    
    r = T(0.02)*r - T(0.01);
    
    return r;
  }
  
  
  
  
  template class CRBM< float >;
  template class CRBM< double >;  
  template class CRBM< math::blas_real<float> >;
  template class CRBM< math::blas_real<double> >;
  
};
