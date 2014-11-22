

#include "bayesian_nnetwork.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <exception>
#include <stdexcept>
#include <typeinfo>

#include "nnetwork.h"
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
    for(unsigned int i=0;i<nnets.size();i++)
      if(this->nnets[i]) delete this->nnets[i];

    nnets.clear();
  }

  
  /*
   * imports and exports samples of p(w) to and from nnetwork
   */
  template <typename T>
  bool bayesian_nnetwork<T>::importSamples(const std::vector<unsigned int>& arch,
					   const std::vector< math::vertex<T> >& weights)
  {
    if(weights.size() <= 0) return false;
    
    std::vector< nnetwork<T>* > nnets;
    nnets.resize(weights.size());
    
    for(unsigned int i=0;i<nnets.size();i++){
      nnets[i] = new nnetwork<T>(arch);
      if(nnets[i]->importdata(weights[i]) == false){
	for(unsigned int j=0;j<=i;j++)
	  delete nnets[i];
	
	return false;
      }
    }


    // remove old data
    for(unsigned int i=0;i<this->nnets.size();i++)
      if(this->nnets[i]) delete this->nnets[i];

    this->nnets = nnets; // copies new pointers over old data

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

    return importSamples(arch, weight);
  }
  

  template <typename T>
  bool bayesian_nnetwork<T>::exportSamples(std::vector<unsigned int>& arch,
					   std::vector< math::vertex<T> >& weights)
  {
    if(nnets.size() <= 0) return false;

    nnets[0]->getArchitecture(arch);

    weights.resize(nnets.size());

    for(unsigned int i=0;i<nnets.size();i++)
      if(nnets[i]->exportdata(weights[i]) == false){
	weights.clear();
	return false;
      }
    
    return true;
  }
  
  
  // calculates E[f(input,w)] and Var[f(x,w)] for given input
  template <typename T>
  bool bayesian_nnetwork<T>::calculate(const math::vertex<T>& input,
				       math::vertex<T>& mean,
				       math::matrix<T>& covariance)
  {
    if(nnets.size() <= 0) return false;

    const unsigned int D = nnets[0]->output_size();
    mean.resize(D);
    covariance.resize(D,D);
    
    mean.zero();
    covariance.zero();

    T ninv = T(1.0f/nnets.size());

    if(nnets.size() <= D)
      covariance.identity(); // regularizer term for small datasize

    for(unsigned int i=0;i<nnets.size();i++){
      nnets[i]->input() = input;
      nnets[i]->calculate();
      math::vertex<T> out = nnets[i]->output();

      mean += ninv*out;
      covariance += ninv*out.outerproduct();
    }

    covariance -= mean.outerproduct();

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
  
  
  // stores and loads bayesian nnetwork to a text file
  // (saves all samples into files)
  template <typename T>
  bool bayesian_nnetwork<T>::load(const std::string& filename) throw()
  {
    try{
      whiteice::conffile configuration;
      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;

      if(configuration.load(filename) == false)
	return false;

      int versionid = 0;
      
      // checks version
      {
	if(!configuration.get(FNN_VERSION_CFGSTR, ints))
	  return false;
	
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
	if(!configuration.get(FNN_ARCH_CFGSTR,ints))
	  return false;
	  
	if(ints.size() < 2)
	  return false;

	arch.resize(ints.size());
	
	for(unsigned int i=0;i<ints.size();i++){
	  if(ints[i] <= 0) return false;
	  arch[i] = (unsigned int)ints[i];
	}
      }

      
      // reads number of samples information
      int numberOfSamples = 0;
      
      {
	if(!configuration.get(FNN_NUMWEIGHTS_CFGSTR, ints))
	  return false;

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
	
	char buffer[80];
	math::vertex<T> w;

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

  
  template <typename T>
  bool bayesian_nnetwork<T>::save(const std::string& filename) const throw()
  {
    try{
      if(nnets.size() <= 0) return false;
      
      whiteice::conffile configuration;

      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      
      // writes version information
      {
	// version number = integer/1000
	ints.push_back(3500); // 3.500
	if(!configuration.set(FNN_VERSION_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }

      std::vector<unsigned int> arch;

      nnets[0]->getArchitecture(arch);
      
      // writes architecture information
      {
	for(unsigned int i=0;i<arch.size();i++)
	  ints.push_back(arch[i]);

	if(!configuration.set(FNN_ARCH_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }

      // writes number of samples information
      {
	ints.push_back(nnets.size());

	if(!configuration.set(FNN_NUMWEIGHTS_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }

// weights: we just convert everything to a big vertex vector and write it
      for(unsigned int index=0;index<nnets.size();index++)
      {
	char buffer[80];
	math::vertex<T> w;
	
	if(nnets[index]->exportdata(w) == false)
	  return false;
	
	for(unsigned int i=0;i<w.size();i++){
	  float f;
	  math::convert(f, w[i]);
	  floats.push_back(f);
	}

	sprintf(buffer, FNN_WEIGHTS_CFGSTR, index);
	
	if(!configuration.set(buffer, floats))
	  return false;
	
	floats.clear();
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
