

#include "bayesian_nnetwork.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <exception>
#include <stdexcept>
#include <typeinfo>
#include <set>

#include "nnetwork.h"
#include "dataset.h"
#include "dinrhiw_blas.h"



namespace whiteice
{

  template <typename T>
  bayesian_nnetwork<T>::bayesian_nnetwork()
  {
    
  }

  
  template <typename T>
  bayesian_nnetwork<T>::~bayesian_nnetwork()
  {
    for(unsigned int i=0;i<nnets.size();i++){
      if(this->nnets[i]){
	delete this->nnets[i];
	this->nnets[i] = NULL;
      }
    }

    nnets.clear();
  }


  // number of samples in BNN
  template <typename T>
  unsigned int bayesian_nnetwork<T>::getNumberOfSamples() const throw(){
	  return nnets.size();
  }

  
  /*
   * imports and exports samples of p(w) to and from nnetwork
   */
  template <typename T>
  bool bayesian_nnetwork<T>::importSamples(const std::vector<unsigned int>& arch,
					   const std::vector< math::vertex<T> >& weights,
					   const typename nnetwork<T>::nonLinearity nl)
  {
    if(weights.size() <= 0) return false;
    
    std::vector< nnetwork<T>* > nnnets;
    nnnets.resize(weights.size());
    
    for(unsigned int i=0;i<nnnets.size();i++){
      nnnets[i] = new nnetwork<T>(arch);
      nnnets[i]->setNonlinearity(nl);
      if(nnnets[i]->importdata(weights[i]) == false){
	for(unsigned int j=0;j<=i;j++){
	  delete nnnets[i];
	  nnnets[i] = NULL;
	}
	
	return false;
      }
    }


    // remove old data
    for(unsigned int i=0;i<this->nnets.size();i++)
      if(this->nnets[i]){
	delete this->nnets[i];
	this->nnets[i] = NULL;
      }
    
    nnets.clear();

    this->nnets = nnnets; // copies new pointers over old data

    return true;
  }


  template <typename T>
  bool bayesian_nnetwork<T>::importNetwork(const nnetwork<T>& net)
  {
    std::vector< math::vertex<T> > weight;
    weight.resize(1);

    std::vector<unsigned int> arch;

    net.getArchitecture(arch);
    if(net.exportdata(weight[0]) == false)
      return false;

    typename nnetwork<T>::nonLinearity nl = 
      net.getNonlinearity();

    return importSamples(arch, weight, nl);
  }
  

  template <typename T>
  bool bayesian_nnetwork<T>::exportSamples(std::vector<unsigned int>& arch,
					   std::vector< math::vertex<T> >& weights,
					   typename nnetwork<T>::nonLinearity& nl,
					   int latestN)
  {
    if(nnets.size() <= 0) return false;
    if(latestN > (signed)nnets.size()) return false;
    if(latestN <= 0) latestN = nnets.size();

    nnets[0]->getArchitecture(arch);
    nl = nnets[0]->getNonlinearity();

    weights.resize(latestN);

    for(unsigned int i=(nnets.size() - latestN);i<nnets.size();i++)
      if(nnets[i]->exportdata(weights[i]) == false){
	weights.clear();
	return false;
      }
    
    return true;
  }

  template <typename T>
  bool bayesian_nnetwork<T>::setNonlinearity(typename nnetwork<T>::nonLinearity nl){
    for(unsigned int i=0;i<nnets.size();i++)
      if(nnets[i])
	if(nnets[i]->setNonlinearity(nl) == false)
	  return false;
    
    return true;
  }

  template <typename T>
  typename nnetwork<T>::nonLinearity bayesian_nnetwork<T>::getNonlinearity(){ // returns sigmoid if there are no nnetworks
    if(nnets.size() > 0)
      if(nnets[0])
	return nnets[0]->getNonlinearity();

    return nnetwork<T>::sigmoidNonLinearity;
  }
  
  
  template <typename T>
  bool bayesian_nnetwork<T>::downsample(unsigned int N)
  {
    if(N == 0) return false;
    if(N >= nnets.size())
      return true;
    
    std::set<unsigned int> samples;
    for(unsigned int i=0;i<nnets.size();i++)
      samples.insert(i);
    
    std::vector< nnetwork<T>* > nn;
    
    while(nn.size() < N){ 
      auto iter = samples.begin();
      int i = rand() % samples.size();
      while(i > 0){ iter++; i--; }
      const int index = *iter;
      
      samples.erase(iter);
      nn.push_back(nnets[index]);
    }
    
    // deletes rest of the nnetworks
    {
      auto iter = samples.begin();
      
      while(iter != samples.end()){
	const int index = (*iter);
	delete nnets[index];
	iter++;
      }
    }
    
    nnets.clear();
    nnets = nn;
    
    return true;
  }
  
  
  // calculates E[f(input,w)] and Var[f(x,w)] for given input
  template <typename T>
  bool bayesian_nnetwork<T>::calculate(const math::vertex<T>& input,
				       math::vertex<T>& mean,
				       math::matrix<T>& covariance,
				       unsigned int SIMULATION_DEPTH, // for recurrent use of nnetworks..
				       int latestN)
  {
    if(nnets.size() <= 0) return false;
    if(latestN > (signed)nnets.size()) return false;
    if(latestN <= 0) latestN = nnets.size();

    if(SIMULATION_DEPTH > 1){
      if(nnets[0]->output_size() + input.size() != nnets[0]->input_size())
	return false;
    }
    else{
      if(input.size() != nnets[0]->input_size())
	return false;
    }

    const unsigned int D = nnets[0]->output_size();
    mean.resize(D);
    covariance.resize(D,D);
    
    mean.zero();
    covariance.zero();

    if(latestN <= (signed)D)
    	covariance.identity(); // regularizer term for small datasize

#pragma omp parallel shared(mean, covariance)
    {
    	math::matrix<T> cov;
    	math::vertex<T> m;

    	m.resize(D);
    	cov.resize(D,D);
    	m.zero();
    	cov.zero();

    	T ninv  = T(1.0f/latestN);

#pragma omp for nowait schedule(dynamic)
    	for(unsigned int i=(nnets.size() - latestN);i<nnets.size();i++){
	        nnets[i]->input().zero();
		nnets[i]->output().zero();
	        nnets[i]->input().write_subvertex(input, 0); // writes input section

		for(unsigned int d=0;d<SIMULATION_DEPTH;d++){
		  if(SIMULATION_DEPTH > 1)
		    nnets[i]->input().write_subvertex(nnets[i]->output(), input.size());
		  nnets[i]->calculate();       // recurrent calculations if needed
		}
		
    		math::vertex<T> out = nnets[i]->output();

    		m += ninv*out;
    		cov += ninv*out.outerproduct();
    	}

#pragma omp critical
    	{
    		mean += m;
    		covariance += cov;
    	}

    }

    covariance -= mean.outerproduct(); // should divide by N-1 but we ignore this in order to have a result for N = 1

    return true;
  }


  template <typename T>
  unsigned int bayesian_nnetwork<T>::outputSize() const throw()
  {
    if(nnets.size() <= 0) return 0;

    return nnets[0]->output().size();
  }

  template <typename T>
  unsigned int bayesian_nnetwork<T>::inputSize() const throw()
  {
    if(nnets.size() <= 0) return 0;

    return nnets[0]->input().size();
  }


#define FNN_VERSION_CFGSTR          "FNN_CONFIG_VERSION"
#define FNN_NUMWEIGHTS_CFGSTR       "FNN_NUM_WEIGHTS"
#define FNN_ARCH_CFGSTR             "FNN_ARCH"
#define FNN_WEIGHTS_CFGSTR          "FNN_WEIGHTS%d"  
#define FNN_NONLINEARITY_CFGSTR     "FNN_NONLINEARITY"  
  
  // stores and loads bayesian nnetwork to a text file
  // (saves all samples into files)
  template <typename T>
  bool bayesian_nnetwork<T>::load(const std::string& filename) throw()
  {
    try{
      // whiteice::conffile configuration;
      whiteice::dataset<T> configuration;
      math::vertex<T> data;
      // unsigned int cluster = 0;
      
      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      
      if(configuration.load(filename) == false)
	return false;
      
      int versionid = 0;
      
      // checks version
      {
	//if(!configuration.get(FNN_VERSION_CFGSTR, ints))
	//return false;
	data = configuration.accessName(FNN_VERSION_CFGSTR, 0);
	ints.resize(data.size());
	for(unsigned int i=0;i<data.size();i++){
	  math::convert(ints[i], data[i]);
	}
	
	
	if(ints.size() != 1)
	  return false;
	
	versionid = ints[0];
	
	ints.clear();
      } 
      
      if(versionid != 3500) // v3.5 datafile
	return false;
      
      std::vector<unsigned int> arch;
      
      // gets architecture
      {
	//if(!configuration.get(FNN_ARCH_CFGSTR,ints))
	// return false;
	data = configuration.accessName(FNN_ARCH_CFGSTR, 0);
	ints.resize(data.size());
	for(unsigned int i=0;i<data.size();i++){
	  math::convert(ints[i], data[i]);
	}
	
	if(ints.size() < 2)
	  return false;
	
	arch.resize(ints.size());
	
	for(unsigned int i=0;i<ints.size();i++){
	  if(ints[i] <= 0) return false;
	  arch[i] = (unsigned int)ints[i];
	}
      }

      // gets nonlinearity of nnetworks
      typename whiteice::nnetwork<T>::nonLinearity nl =
	whiteice::nnetwork<T>::sigmoidNonLinearity;
      {
	data = configuration.accessName(FNN_NONLINEARITY_CFGSTR, 0);
	ints.resize(data.size());
	
	for(unsigned int i=0;i<data.size();i++){
	  math::convert(ints[i], data[i]);
	}
	
	if(ints.size() != 1)
	  return false;
	
	if(ints[0] == 0)
	  nl = whiteice::nnetwork<T>::sigmoidNonLinearity;
	else if(ints[0] == 1)
	  nl = whiteice::nnetwork<T>::halfLinear;
	else if(ints[0] == 2)
	  nl = whiteice::nnetwork<T>::pureLinear;
	else
	  return false; // bad data
      }

      
      
      // reads number of samples information
      int numberOfSamples = 0;
      
      {
	data = configuration.accessName(FNN_NUMWEIGHTS_CFGSTR, 0);
	ints.resize(data.size());
	for(unsigned int i=0;i<data.size();i++){
	  math::convert(ints[i], data[i]);
	}
	
	if(ints.size() != 1)
	  return false;
	
	numberOfSamples = ints[0];
	
	ints.clear();
      }
      
      if(numberOfSamples <= 0)
	return false;
      
      
      std::vector< nnetwork<T>* > nets;
      
      nets.resize(numberOfSamples);
      
      // weights: we just import a big vertex vector
      for(unsigned int index=0;index<nets.size();index++)
      {
	nets[index] = new nnetwork<T>(arch);
	nets[index]->setNonlinearity(nl); // sets non-linearity
	
	math::vertex<T> w;
	
	w = configuration.accessName(FNN_WEIGHTS_CFGSTR, index);
	
#if 0
	char buffer[80];
	
	sprintf(buffer, FNN_WEIGHTS_CFGSTR, index);
	
	if(!configuration.get(buffer, floats)){
	  for(unsigned int j=0;j<=index;j++)
	    delete nets[j];
	  return false;
	}

	w.resize(floats.size());
	
	for(unsigned int i=0;i<w.size();i++){
	  w[i] = T(floats[i]);
	}
#endif
	
	
	if(nets[index]->importdata(w) == false){
	  for(unsigned int j=0;j<=index;j++)
	    delete nets[j];
	  return false;
	}
	
	floats.clear();
      }
      
      for(unsigned int i=0;i<nnets.size();i++)
	delete nnets[i]; // deletes old networks
      
      nnets = nets; // saves the loaded nnetworks
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      
      return false;
    }
  }

// #define FNN_VERSION_CFGSTR          "FNN_CONFIG_VERSION"
// #define FNN_NUMWEIGHTS_CFGSTR       "FNN_NUM_WEIGHTS"
// #define FNN_ARCH_CFGSTR             "FNN_ARCH"
// #define FNN_WEIGHTS_CFGSTR          "FNN_WEIGHTS%d"

  
  template <typename T>
  bool bayesian_nnetwork<T>::save(const std::string& filename) const throw()
  {
    try{
      if(nnets.size() <= 0) return false;
      
      // whiteice::conffile configuration;
      whiteice::dataset<T> configuration;
      math::vertex<T> data;

      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      

      // writes version information
      {
	// version number = integer/1000
	ints.push_back(3500); // 3.500

    configuration.createCluster(FNN_VERSION_CFGSTR, ints.size());
    data.resize(ints.size());
    for(unsigned int i=0;i<ints.size();i++){
    	data[i] = ints[i];
    }
    configuration.add(configuration.getCluster(FNN_VERSION_CFGSTR), data);

	//if(!configuration.set(FNN_VERSION_CFGSTR, ints))
	//  return false;
	
	ints.clear();
      }

      std::vector<unsigned int> arch;

      nnets[0]->getArchitecture(arch);
      
      // writes architecture information
      {
	for(unsigned int i=0;i<arch.size();i++)
	  ints.push_back(arch[i]);

    configuration.createCluster(FNN_ARCH_CFGSTR, ints.size());
    data.resize(ints.size());
    for(unsigned int i=0;i<ints.size();i++){
    	data[i] = ints[i];
    }
    configuration.add(configuration.getCluster(FNN_ARCH_CFGSTR), data);

	//if(!configuration.set(FNN_ARCH_CFGSTR, ints))
	//  return false;
	
	ints.clear();
      }

      // writes number of samples information
      {
	ints.push_back(nnets.size());

	configuration.createCluster(FNN_NUMWEIGHTS_CFGSTR, ints.size());
	data.resize(ints.size());
	for(unsigned int i=0;i<ints.size();i++){
	  data[i] = ints[i];
	}
	configuration.add(configuration.getCluster(FNN_NUMWEIGHTS_CFGSTR), data);
	
	ints.clear();
      }

      // writes non-linearity information (assumes all have same nonlin)
      {
	typename whiteice::nnetwork<T>::nonLinearity nl =
	  whiteice::nnetwork<T>::sigmoidNonLinearity;

	if(nnets.size() > 0){
	  nl = nnets[0]->getNonlinearity();

	  if(nl == whiteice::nnetwork<T>::sigmoidNonLinearity)
	    ints.push_back(0);
	  else if(nl == whiteice::nnetwork<T>::halfLinear)
	    ints.push_back(1);
	  else if(nl == whiteice::nnetwork<T>::pureLinear)
	    ints.push_back(2);
	  else
	    return false;
	}

	configuration.createCluster(FNN_NONLINEARITY_CFGSTR, ints.size());
	data.resize(ints.size());
	for(unsigned int i=0;i<ints.size();i++){
	  data[i] = ints[i];
	}
	configuration.add(configuration.getCluster(FNN_NONLINEARITY_CFGSTR), data);
	
	ints.clear();
      }

// weights: we just convert everything to a big vertex vector and write it
      math::vertex<T> w;
      if(nnets[0]->exportdata(w) == false)
    	  return false;

      configuration.createCluster(FNN_WEIGHTS_CFGSTR, w.size());

      for(unsigned int index=0;index<nnets.size();index++)
      {
	// char buffer[80];
	math::vertex<T> w;
	
	if(nnets[index]->exportdata(w) == false)
	  return false;
	

#if 0
	for(unsigned int i=0;i<w.size();i++){
	  float f;
	  math::convert(f, w[i]);
	  floats.push_back(f);
	}

	sprintf(buffer, FNN_WEIGHTS_CFGSTR, index);
	
	if(!configuration.set(buffer, floats))
	  return false;


	floats.clear();
#endif


    configuration.add(configuration.getCluster(FNN_WEIGHTS_CFGSTR), w);

      }      
      
      
      return configuration.save(filename);
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      
      return false;
    }
    
  }

  
  template class bayesian_nnetwork< float >;
  template class bayesian_nnetwork< double >;  
  template class bayesian_nnetwork< math::blas_real<float> >;
  template class bayesian_nnetwork< math::blas_real<double> >;
  
};
