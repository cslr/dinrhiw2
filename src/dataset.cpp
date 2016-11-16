/*
 * data set implementation
 *
 * FIXME: save()/load() 
 *        assumes sizeof(float) == 32bits and sizeof(int) == 32bits
 *
 * FIXME: dataset load()/save() mysteriously FAILs on mingw/win32
 */

#ifndef dataset_cpp
#define dataset_cpp

#include <stdexcept>
#include <exception>
#include <vector>
#include <typeinfo>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "eig.h"
#include "dataset.h"
#include "blade_math.h"


/**************************************************/

namespace whiteice
{
  
  // static member definition
  template <typename T> 
  const char* dataset<T>::FILEID_STRING = "whiteice::dataset";
  
  
  template <typename T>
  dataset<T>::dataset()
  {
    clusters.clear();
    namemapping.clear();
  }
  
  
  // creates dataset with given dimensionality
  // data is set of vectors
  template <typename T>
  dataset<T>::dataset(unsigned int dimension) throw(std::out_of_range)
  {
    // dimension can be zero, this means
    // default cluster cannot be used
    
    clusters.resize(1);
    clusters[0].cname  = std::string("default");
    clusters[0].cindex = 0;
    clusters[0].preprocessings.clear();
    clusters[0].data_dimension = dimension;
    namemapping["default"] = 0;
  }
  
  
  template <typename T>
  dataset<T>::dataset(const dataset<T>& d)
  {
    clusters.resize(d.clusters.size());

    for(unsigned int i=0;i<clusters.size();i++){
      clusters[i] = d.clusters[i];
      namemapping[clusters[i].cname] = i;
    }
    
  }
  
  
  template <typename T>
  dataset<T>::~dataset() throw(){
    clusters.clear();
    namemapping.clear();
  }
  
  
  ////////////////////////////////////////////////////////////
  // cluster code
  
  template <typename T>
  bool dataset<T>::createCluster(const std::string& name, const unsigned int dimension)
  {
    if(name.length() <= 0)
      return false;
    
    if(namemapping.find(name) != namemapping.end())
      return false;
    
    unsigned int n = clusters.size();
    clusters.resize(clusters.size()+1);
    clusters[n].cname = name;
    clusters[n].cindex = n;
    clusters[n].data_dimension = dimension;
    clusters[n].preprocessings.clear();
    namemapping[name] = n;
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::getClusterNames(std::vector<std::string>& names) const
  {
    names.clear();
    
    typename std::vector<cluster>::const_iterator i;
    i = clusters.begin();
    
    while(i != clusters.end()){
      names.push_back(i->cname);
      i++;
    }
    
    return true;
  }
  
  
  template <typename T>
  unsigned int dataset<T>::getCluster(const std::string& name) const
  {
	    typename std::map<std::string, unsigned int>::const_iterator i;
	    i = namemapping.find(name);

	    if(i == namemapping.end())
	      return (unsigned int)(-1);

	    return (i->second);
  }
  
  
  template <typename T>
  std::string dataset<T>::getName(unsigned int index) const {
    if(index >= clusters.size()) return "";
    
    return clusters[index].cname;
  }


  template <typename T>
  bool dataset<T>::setName(const unsigned int index, const std::string& name) {
    if(index >= clusters.size()) return false;
    
    if(name.size() > 0)
      clusters[index].cname = name;
    else
      return false;
    
    return true;
  }
  
  
  template <typename T>
  unsigned int dataset<T>::getNumberOfClusters() const
  {
    return clusters.size();
  }
  
  
  template <typename T>
  bool dataset<T>::removeCluster(std::string& name)
  {
    typename std::map<std::string, unsigned int>::iterator i = 
      namemapping.find(name);
    
    if(i == namemapping.end())
      return false;
    
    return removeCluster(i->second);
  }
  
  
  template <typename T>
  bool dataset<T>::removeCluster(unsigned int index)
  {
    if(index >= clusters.size())
      return false;
    
    // not very good implementation, ok because
    // number of clusters is usually small 
    // ( < 10, typically 1 or 2)
    
    // removes cluster
    typename std::vector<cluster>::iterator i = clusters.begin();
    for(unsigned int j=0;j<index;j++,i++);
    clusters.erase(i);
    
    // recalculates *->cindex and namemapping entries
    
    namemapping.clear();
    
    i = clusters.begin();
    index = 0;
    
    while(i != clusters.end()){
      i->cindex = index;
      namemapping[i->cname] = index;
      index++;
      i++;
    }
    
    return true;
  }
  
  
  
  // adds data examples
  template <typename T>
  bool dataset<T>::add(const math::vertex<T>& input, bool nopreprocess) throw(){
    return add(0, input, nopreprocess);
  }
  
  
  template <typename T>
  bool dataset<T>::add(const std::vector<math::vertex<T> >& inputs, bool nopreprocess) throw()
  {
    return add(0, inputs, nopreprocess);
  }
  
  
  template <typename T>  
  bool dataset<T>::add(const std::string& input, bool nopreprocess) throw(){
    return add(0, input, nopreprocess);
  }
  
  
  template <typename T>  
  bool dataset<T>::add(const std::vector<std::string>& inputs, bool nopreprocess) throw()
  {
    return add(0, inputs, nopreprocess);
  }
  
  
  
  // adds data to clusters
  template <typename T>
  bool dataset<T>::add(unsigned int index,
		       const math::vertex<T>& input, bool nopreprocess) throw()
  {
    if(index >= clusters.size()) 
      // slow.. (internal calls need only single check)
      return false;
    
    try{ 
      if(input.size() != clusters[index].data_dimension)
	return false;
      
      clusters[index].data.push_back(input);
      unsigned int n = clusters[index].data.size() - 1;
      
      if(nopreprocess == false){
	// preprocesses if it is needed
	if(!preprocess(index, clusters[index].data[n])){
	  clusters[index].data.resize(n);
	  return false;
	}
      }
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "unexpected exception: "
		<< e.what() << std::endl;
      
      return false;
    }
    
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::add(unsigned int index,
		       const std::vector<math::vertex<T> >& inputs, bool nopreprocess) throw()
  {
    if(index >= clusters.size())
      return false;
    
    if(inputs.size() <= 0)
      return true;
    
    {
      const unsigned int DD = clusters[index].data_dimension;
      
      for(unsigned int i=0;i<inputs.size();i++)
	if(inputs[i].size() != DD)
	  return false;
    }

    typename std::vector<math::vertex<T> >::const_iterator i;
    
    {
      unsigned int last = clusters[index].data.size();
      
      for(i=inputs.begin();i!=inputs.end();i++,last++){
	clusters[index].data.push_back(*i);
	if(nopreprocess == false)
	  preprocess(index, clusters[index].data[last]);
      }
    }
    
    return true;
  }

  
  template <typename T>
  bool dataset<T>::add(unsigned int index, const std::vector<T>& input,
		       bool nopreprocess) throw()
  {
    math::vertex<T> v(input.size());
    
    for(unsigned int i=0;i<v.size();i++)
      v[i] = input[i];
    
    return this->add(index, v, nopreprocess);
  }
  
  

  
  
  template <typename T>
  bool dataset<T>::add(unsigned int index,
		       const std::string& input, bool nopreprocess) throw()
  {
    if(index >= clusters.size()) // slow.. (internal calls)
      return false;
    
    if(clusters[index].data_dimension != input.length())
      return false;
    
    try{
      math::vertex<T> vec;
      vec.resize(clusters[index].data_dimension);
      
      for(unsigned int i=0;i<input.length();i++)
	vec[i] = T(input[i]);
      
      return add(index, vec);
    }
    catch(std::exception& e){ return false; }
    
  }
  
  
  template <typename T>
  bool dataset<T>::add(unsigned int index,
		       const std::vector<std::string>& inputs,
		       bool nopreprocess) throw()
  {
    typename std::vector<std::string >::const_iterator i;
    
    for(i=inputs.begin();i!=inputs.end();i++){
      if(!add(index, *i)) return false;
    }
    
    return true;
  }
  
  // creates empty dataset
  template <typename T>
  bool dataset<T>::clear()
  {
    clusters.clear();
    namemapping.clear();
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::clearData(unsigned int index)
  {
    if(index >= clusters.size())
      return false;
    
    clusters[index].data.clear();
    return true;
  }
  
  
  template <typename T> // reduces size of data
  bool dataset<T>::resize(unsigned int index,
			  unsigned int nsize) throw()
  {
    if(index >= clusters.size())
      return false;
    
    if(clusters[index].data.size() < nsize)
      return false;
    
    clusters[index].data.resize(nsize);
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::clearAll(unsigned int index)
  {
    if(index >= clusters.size())
      return false;
    
    clusters[index].data.clear();
    
    clusters[index].preprocessings.clear();
    
    clusters[index].Rxx.resize(1,1);
    clusters[index].Wxx.resize(1,1);
    clusters[index].invWxx.resize(1,1);
    clusters[index].mean.resize(1);
    clusters[index].variance.resize(1);

    clusters[index].ICA.resize(1,1);
    
    return true;
  }

  
  /*
   * downsamples all clusters jointly down to samples samples.
   * fails if clusters have different sizes or samples == 0 or
   * it is larger that given samples. 
   */
  template <typename T>
  bool dataset<T>::downsampleAll(unsigned int samples) throw()
  {
    if(clusters.size() <= 0) return false;

    const unsigned int N = clusters[0].data.size();

    for(unsigned int i=0;i<clusters.size();i++){
      if(clusters[i].data.size() != N)
	return false;
    }

    if(samples >= N){
      return false;
    }
    else if(samples == 0){
      for(unsigned int i=0;i<clusters.size();i++){
	clusters[i].data.clear();
      }
      return true;
    }
    
    std::vector<dataset<T>::cluster> d;
    d.resize(clusters.size());

    for(unsigned int i=0;i<samples;i++){
      unsigned int index = rand() % N;
      
      for(unsigned int j=0;j<clusters.size();j++){
	d[j].data.push_back(clusters[j].data[index]);
      }
    }

    for(unsigned int j=0;j<clusters.size();j++)
      clusters[j].data = d[j].data;

    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::removeBadData()
  {
    if(clusters.size() <= 0)
      return true; // nothing to do
    
    // checks all cluster sizes are equal
    const unsigned int N = clusters[0].data.size();

    for(unsigned int i=0;i<clusters.size();i++){
      if(clusters[i].data.size() != N)
	return false;
    }
    
    
    std::vector<dataset<T>::cluster> d;
    d = clusters;
    
    for(unsigned int k=0;k<clusters.size();k++){
      d[k].data.clear();
    }
    
    for(unsigned int i=0;i<N;i++){
      bool bad_data = false;
      
      for(unsigned int k=0;k<clusters.size();k++){
	const auto& v = clusters[k].data[i];
	
	// std::cout << "test:  " << v << std::endl;
	// std::cout << "vsize: " << v.size() << std::endl;
	
	for(unsigned int d=0;d<v.size();d++){
	  // std::cout << v[d] << std::endl;
	  // std::cout << whiteice::math::tohex(v[d]) << std::endl;
	  
	  if(whiteice::math::isnan(v[d])){
	    // printf("NaN detected\n");
	    bad_data = true;
	  }
	  else if(whiteice::math::isinf(v[d])){
	    bad_data = true;
	  }
	}
	
	// std::cout << "bad data status: " << bad_data << std::endl;
      }
      
      // if(bad_data) printf("BAD DATA DETECTED\n");
      
      
      if(bad_data == false){
	for(unsigned int k=0;k<clusters.size();k++){
	  d[k].data.push_back(clusters[k].data[i]);
	}
      }
    }
    
    clusters = d;
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::getData(unsigned int index, std::vector< math::vertex<T> >& data) const throw(std::out_of_range)
  {
    if(index >= clusters.size())
      throw std::out_of_range("cluster index too big");
    
    data = clusters[index].data;
    
    return true;
  }
  
  
  // iterators for dataset
  template <typename T>
  typename dataset<T>::iterator dataset<T>::begin(unsigned int index) throw(std::out_of_range)
  {
    if(index >= clusters.size())
      throw std::out_of_range("cluster index too big.");
    
    return clusters[index].data.begin();
  }
  
  template <typename T>
  typename dataset<T>::iterator dataset<T>::end(unsigned int index) throw(std::out_of_range)
  {
    if(index >= clusters.size())
      throw std::out_of_range("cluster index too big.");
    
    return clusters[index].data.end();
  }
  
  template <typename T>
  typename dataset<T>::const_iterator dataset<T>::begin(unsigned int index) const throw(std::out_of_range)
  {
    if(index >= clusters.size())
      throw std::out_of_range("Cluster index too big.");
    
    return clusters[index].data.begin();
  }
  
  template <typename T>
  typename dataset<T>::const_iterator dataset<T>::end(unsigned int index) const throw(std::out_of_range)
  {
    if(index >= clusters.size())
      throw std::out_of_range("Cluster index too big.");
    
    return clusters[index].data.end();
  }
  
  
  
  template <typename T>
  bool dataset<T>::load(const std::string& filename) throw()
  {
    // loads datasets which have been saved in
    // dataset format = 1 (older dataset format 0 is not supported)
    //
    //  FILEID_STRING : char[]
    //  version : INT
    //  cnum    : INT  number of clusters
    //  namelen : INT  length of names section in bytes (includes padding)
    //  names   : [list of NULL terminated strings] (cnum names)
    //  padding : (=> address dividable by 4)
    //  clusters: [list of CLUSTER]
    //
    // CLUSTER:
    //  datasize: INT  number of data vectors
    //  dimen.  : INT
    //  NORMFLAG: INT  normalization flags
    //  softmax : float
    //  [mean]  : float[DIM]
    //  [var]   : float[DIM]
    //  [Rxx]   : float[DIM]x[DIM]
    //  [ICA]   : float[DIM]x[DIM]
    //  data    : float[datasize]x[DIM]
    
    if(filename.length() <= 0)
      return false;
    
    FILE* fp = (FILE*)fopen(filename.c_str(), "rb");
    
    if(fp == 0) return false;
    if(feof(fp) || ferror(fp))
      return false;
    
    // printf("L: %d\n", (int)ftell(fp));
    
    unsigned int version = 0xFFFFFFFF;
    unsigned int cnum = 0;
    
    // checks file id string
    {
      char line[128] = "";

      // printf("F: %d\n", (int)strlen(FILEID_STRING)+1);
      const size_t s = strlen(FILEID_STRING)+1;
      
      if(fread(line, 1, s, fp) != s)
      {
	fclose(fp);
	return false;
      }

      if(strcmp(line, FILEID_STRING) != 0){
	fclose(fp);
	return false;
      }
    }

    // printf("L: %d\n", (int)ftell(fp));
    
    if(fread(&version, 4, 1, fp) != 1){
      fclose(fp);
      return false;
    }
    
    if(fread(&cnum, 4, 1, fp) != 1){
      fclose(fp);
      return false;
    }
    
    if(version != 1){
      fclose(fp);
      return false;
    }
    
    
    clusters.resize(cnum);

    // reads names
    if(cnum > 0){
      // gets names section length
      
      unsigned int namesSectionSize = 0;
      
      if(fread(&namesSectionSize, 4, 1, fp) != 1){
	clusters.resize(0);
	fclose(fp);
	return false;
      }
      
      
      char* names = (char*)malloc(namesSectionSize);
      if(names == 0){
	clusters.resize(0);
	fclose(fp);
	return false;
      }
      
      if(fread(names, namesSectionSize, 1, fp) != 1){
	clusters.resize(0);
	free(names);
	fclose(fp);
	return false;
      }
      
      char *ptr, *endptr;
      ptr = names;
      endptr = &(names[namesSectionSize]);
      
      unsigned int index = 0;
      while(ptr < endptr && index <clusters.size()){
	char* begin = ptr;
	
	while(ptr < endptr && *ptr != '\0') ptr++;
	
	// found null
	if(*ptr == '\0'){
	  clusters[index].cname = begin; // copies string
	  index++;
	}
	
	ptr++;
      }
      
      free(names);
      
      if(index != clusters.size()){
	clusters.resize(0);
	fclose(fp);
	return false;
      }
      
    }
    
    //////////////////////////////////////////////////////////////////////
    // reads in each cluster
    
    for(unsigned int i=0;i<clusters.size();i++){
      unsigned int datasize = 0;
      unsigned int data_dimension = 0;
      unsigned int flags = 0;
      float d32_softmax = 0.0f;
      
      // reads basic cluster information
      
      if(fread(&datasize, 4, 1, fp) != 1){
	clusters.resize(0);
	fclose(fp);
	return false;
      }

      if(fread(&data_dimension, 4, 1, fp) != 1){
	clusters.resize(0);
	fclose(fp);
	return false;	
      }

      if(fread(&flags, 4, 1, fp) != 1){
	clusters.resize(0);
	fclose(fp);
	return false;	
      }
      
      if(fread(&d32_softmax, 4, 1, fp) != 1){
	clusters.resize(0);
	fclose(fp);
	return false;	
      }

      
      clusters[i].data_dimension = data_dimension;
      clusters[i].softmax_parameter = T(d32_softmax);
      
      clusters[i].preprocessings.clear();
      
      if(flags & 0x02)
	clusters[i].preprocessings.push_back(dnSoftMax);
      if(flags & 0x01)
	clusters[i].preprocessings.push_back(dnMeanVarianceNormalization);
      if(flags & 0x04)
	clusters[i].preprocessings.push_back(dnCorrelationRemoval);
      if(flags & 0x08)
	clusters[i].preprocessings.push_back(dnLinearICA);	
      
      
      
      //////////////////////////////////////////////////////////////////////
      // reads cluster statistics

      float* buffer = (float*)calloc(clusters[i].data_dimension, 4);
      
      if(flags & 0x01){
	clusters[i].mean.resize(clusters[i].data_dimension);
	clusters[i].variance.resize(clusters[i].data_dimension);
	
	if(fread(buffer, 4, clusters[i].mean.size(), fp) != 
	   clusters[i].mean.size())
	  {
	    clusters.resize(0);
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
	
	
	for(unsigned int j=0;j<clusters[i].mean.size();j++)
	  clusters[i].mean[j] = T(buffer[j]);
	
	
	if(fread(buffer, 4, clusters[i].variance.size(), fp) != 
	   clusters[i].variance.size())
	  {
	    clusters.resize(0);
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
	
	for(unsigned int j=0;j<clusters[i].mean.size();j++)
	  clusters[i].variance[j] = T(buffer[j]);
	
      }
      
      if(flags & 0x04){
	clusters[i].Rxx.resize(clusters[i].data_dimension,
			       clusters[i].data_dimension);
	
	for(unsigned int a=0;a<data_dimension;a++){
	  
	  if(fread(buffer, 4, data_dimension, fp) != data_dimension){
	    clusters.resize(0);
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
	  
	  for(unsigned int b=0;b<data_dimension;b++)
	    clusters[i].Rxx(a,b) = T(buffer[b]);
	}
	
	
	// recalculates Wx and invWx vectors
	math::matrix<T> D(clusters[i].Rxx);
	math::matrix<T> V, Vt, invD;
	
	if(symmetric_eig(D, V) == false){
	  clusters.resize(0);
	  fclose(fp);
	  free(buffer);
	  return false;
	}
	
	invD = D;
	
	for(unsigned int j=0;j<D.ysize();j++){
	  T d = invD(j,j);

	  if(d > T(10e-8)){
	      invD(j,j) = whiteice::math::sqrt(T(1.0)/whiteice::math::abs(d));
	      D(j,j)    = whiteice::math::sqrt(whiteice::math::abs(d));
	  }
	  else{
	    invD(j,j) = T(0.0f);
	    D(j,j)    = T(0.0f);
	  }
	  
	}
	
	Vt = V;
	Vt.transpose();
	
	clusters[i].Wxx = V * invD * Vt;
	clusters[i].invWxx = V * D * Vt;
	
      }


      if(flags & 0x08){
	clusters[i].ICA.resize(clusters[i].data_dimension,
			       clusters[i].data_dimension);
	
	for(unsigned int a=0;a<data_dimension;a++){
	  
	  if(fread(buffer, 4, data_dimension, fp) != data_dimension){
	    clusters.resize(0);
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
	  
	  for(unsigned int b=0;b<data_dimension;b++)
	    clusters[i].ICA(a,b) = T(buffer[b]);
	}

	clusters[i].invICA = clusters[i].ICA;
	if(clusters[i].invICA.inv() == false)
	  return false; // calculating inverse of ICA failed.
      }      

      //////////////////////////////////////////////////////////////////////
      // reads cluster data
      
      clusters[i].data.resize(datasize);
      
      for(unsigned int a=0;a<clusters[i].data.size();a++){
	
	if(fread(buffer, 4, clusters[i].data_dimension, fp) != 
	   clusters[i].data_dimension)
	{
	  clusters.resize(0);
	  fclose(fp);
	  free(buffer);
	  return false;
	}
	
	clusters[i].data[a].resize(data_dimension);
	for(unsigned int b=0;b<data_dimension;b++)
	  clusters[i].data[a][b] = T(buffer[b]);
      }
      
      
      free(buffer);
    }
    
    
    if(ferror(fp)){
      clusters.resize(0);
      fclose(fp);      
      return false;
    }
    
    
    fclose(fp);
    
    
    // sets up rest of data structures
    namemapping.clear();
    
    for(unsigned int i=0;i<clusters.size();i++){
      clusters[i].cindex = i;
      namemapping[clusters[i].cname] = i;
    }
    
    
    return true;
  }
  
  
  
  
  template <typename T>
  bool dataset<T>::save(const std::string& filename) const throw()
  {
    // dataset is saved as binary file in following format.
    // all data is either 32bit unsigned integers or 32bit floats
    // COMPLEX NUMBERS ARE NOT CURRENTLY SUPPORTED!
    // 
    //  FILEID_STRING : char[]
    //  version : INT
    //  cnum    : INT  number of clusters
    //  namelen : INT  length of names list in bytes (incl. padding)
    //  names   : [list of NULL terminated strings] (cnum names)
    //  padding : (address divisable by 4)
    //  clusters: [list of CLUSTER]
    //
    // CLUSTER:
    //  datasize: INT  number of data vectors
    //  dimen.  : INT
    //  NORMFLAG: INT  normalization flags
    //  softmax : float
    //  [mean]  : float[DIM]
    //  [var]   : float[DIM]
    //  [Rxx]   : float[DIM]x[DIM]
    //  [ICA]   : float[DIM]x[DIM]
    //  data    : float[datasize]x[DIM]
    
    if(filename.length() <= 0)
      return false;
    
    FILE* fp = (FILE*)fopen(filename.c_str(), "wb");
     
    if(fp == 0) return false;
    if(ferror(fp)){ fclose(fp); return false; }

    const unsigned int version = 1; // 0 was initial version number for previous
    // dataset fileformat (not supported)
    const unsigned int cnum    = clusters.size();

    if(fwrite(FILEID_STRING, 1, strlen(FILEID_STRING)+1, fp) != strlen(FILEID_STRING)+1){
      fclose(fp);
      remove(filename.c_str());
      return false;
    }


    if(fwrite(&version, 4, 1, fp) != 1){
      fclose(fp);
      remove(filename.c_str());
      return false;
    }


    if(fwrite(&cnum, 4, 1, fp) != 1){
      fclose(fp);
      remove(filename.c_str());
      return false;
    }


    // writes names list
    if(clusters.size() > 0){
      unsigned int namesSectionSize = 0;

      for(unsigned int i=0;i<clusters.size();i++)
	namesSectionSize += (clusters[i].cname.length() + 1);

      // adds padding
      namesSectionSize = ((namesSectionSize+3)/4)*4;

      if(fwrite(&namesSectionSize, 4, 1, fp) != 1){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }

      char* names = (char*)calloc(1, namesSectionSize);
      if(names == 0){	
	fclose(fp);
	remove(filename.c_str());
	return false;
      }

      char *ptr;
      ptr = names;
      
      for(unsigned int i=0;i<clusters.size();i++){
	for(unsigned int j=0;j<(clusters[i].cname.length()+1);j++, ptr++){
	  *ptr = clusters[i].cname[j];
	}
      }
      
      
      if(fwrite(names, namesSectionSize, 1, fp) != 1){
	fclose(fp);
	free(names);
	remove(filename.c_str());
	return false;
      }

      free(names);
    }


    //////////////////////////////////////////////////////////////////////
    // writes each cluster

    for(unsigned int i=0;i<clusters.size();i++){
      const unsigned int datasize = clusters[i].data.size();
      const unsigned int data_dimension = clusters[i].data_dimension;

      unsigned int flags = 0;
      // flags are (order is always same):
      // bit 0 = dnMeanVarianceNormalization
      // bit 1 = dnSoftMax
      // bit 2 = dnCorrelationRemoval
      // bit 3 = dnLinearICA

      for(unsigned int j=0;j<clusters[i].preprocessings.size();j++){
	if(clusters[i].preprocessings[j] == dnMeanVarianceNormalization)
	  flags |= 0x01;
	else if(clusters[i].preprocessings[j] == dnSoftMax)
	  flags |= 0x02;
	else if(clusters[i].preprocessings[j] == dnCorrelationRemoval)
	  flags |= 0x04;
	else if(clusters[i].preprocessings[j] == dnLinearICA)
	  flags |= 0x08;
      }

      float d32_softmax = 0.0f;
      math::convert(d32_softmax, clusters[i].softmax_parameter);

      // writes basic cluster information

      if(fwrite(&datasize, 4, 1, fp) != 1){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }

      if(fwrite(&data_dimension, 4, 1, fp) != 1){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }

      if(fwrite(&flags, 4, 1, fp) != 1){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }

      if(fwrite(&d32_softmax, 4, 1, fp) != 1){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }


      //////////////////////////////////////////////////////////////////////
      // writes cluster statistics
      
      
      float* buffer = (float*)calloc(4, clusters[i].data_dimension);

      if(buffer == 0){
	fclose(fp);
	remove(filename.c_str());
	return false;
      }
      
      
      if(flags & 0x01){
	for(unsigned int j=0;j<clusters[i].mean.size();j++)
	  math::convert(buffer[j], clusters[i].mean[j]);
	
	if(fwrite(buffer, 4, clusters[i].mean.size(), fp) != 
	   clusters[i].mean.size())
	  {
	    fclose(fp);
	    remove(filename.c_str());
	    free(buffer);
	    return false;
	  }
	
	
	for(unsigned int j=0;j<clusters[i].variance.size();j++)
	  math::convert(buffer[j], clusters[i].variance[j]);
	
	if(fwrite(buffer, 4, clusters[i].variance.size(), fp) != 
	   clusters[i].variance.size())
	  {
	    remove(filename.c_str());
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
      }
      
      
      if(flags & 0x04){
	for(unsigned int a=0;a<data_dimension;a++){
	  
	  for(unsigned int b=0;b<data_dimension;b++)
	    math::convert(buffer[b], clusters[i].Rxx(a,b));
	  
	  if(fwrite(buffer, 4, data_dimension, fp) != data_dimension){
	    remove(filename.c_str());
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
	}
      }


      if(flags & 0x08){
	for(unsigned int a=0;a<data_dimension;a++){
	  
	  for(unsigned int b=0;b<data_dimension;b++)
	    math::convert(buffer[b], clusters[i].ICA(a,b));
	  
	  if(fwrite(buffer, 4, data_dimension, fp) != data_dimension){
	    remove(filename.c_str());
	    fclose(fp);
	    free(buffer);
	    return false;
	  }
	}
      }
      
      
      //////////////////////////////////////////////////////////////////////
      // writes cluster data
      
      for(unsigned int a=0;a<clusters[i].data.size();a++){
	
	for(unsigned int b=0;b<data_dimension;b++)
	  math::convert(buffer[b], clusters[i].data[a][b]);
	
	if(fwrite(buffer, 4, data_dimension, fp) != data_dimension){
	  remove(filename.c_str());
	  fclose(fp);
	  free(buffer);
	  return false;
	}
      }
      
      free(buffer);
      
    }
    
    
    if(ferror(fp))
      return false;
    
    fclose(fp);
    
    return true;
  }
  

  template <typename T>
  bool dataset<T>::exportAscii(const std::string& filename) const throw()
  {
    if(filename.length() <= 0)
      return false;
    
    FILE* fp = (FILE*)fopen(filename.c_str(), "wt");
    
    if(fp == 0) return false;
    if(ferror(fp)){ fclose(fp); return false; }
    
    const unsigned int BUFSIZE = 2048;
    
    char* buffer = (char*)malloc(BUFSIZE*sizeof(char));
    if(buffer == NULL){ fclose(fp); return false; }
    
    for(unsigned int index = 0;index < clusters.size();index++){
      snprintf(buffer, BUFSIZE, "# cluster %d: %d datapoints %d dimensions\n",
	       index, (int)clusters[index].data.size(), clusters[index].data_dimension);
      fputs(buffer, fp);
      
      // dumps data in this cluster to ascii format
      for(auto d : clusters[index].data){
	
	this->invpreprocess(index, d); // removes possible preprocessing from data
	
	if(clusters[index].data_dimension > 0){
	  float value = 0.0f;
	  whiteice::math::convert(value, d[0]);
	  snprintf(buffer, BUFSIZE, "%+f", value);
	  fputs(buffer, fp);
	  
	  for(unsigned int i=1;i<d.size();i++){
	    float value = 0.0f;
	    whiteice::math::convert(value, d[i]);
	    snprintf(buffer, BUFSIZE, " %+f", value);
	    fputs(buffer, fp);
	  }
	  
	  fputs("\n", fp);
	}
      }
    }
    
    free(buffer);
    if(ferror(fp)){ fclose(fp); return false;}
    
    fclose(fp);
    return true;
  }

  
  /*
   * imports space, "," or ";" separated floating point numbers as vectors into cluster 0
   * which will be overwritten. Ignores the first line which may contain headers and
   * reads at most LINES of vertex data or unlimited amount of data (if set to 0).
   */
  template <typename T>
  bool dataset<T>::importAscii(const std::string& filename, unsigned int LINES) throw()
  {
    std::vector< math::vertex<T> > import;

    FILE* fp = fopen(filename.c_str(), "rt");
    if(fp == 0 || ferror(fp)){
      if(fp) fclose(fp);
      return false;
    }

    const unsigned int BUFLEN = 50000;
    char* buffer = (char*)malloc(BUFLEN);
    if(buffer == NULL){
      fclose(fp);
      return false;
    }
    
    
    // import format is
    // <file> = (<line>"\n")*
    // <line> = <vector> = "%f %f %f %f ... "
    // where separators are either spaces, ";" or "," numbers are assumed to have form -12.3101

    unsigned int lines = 0;

    while(!feof(fp)){
      if(LINES){
	if(lines >= LINES)
	  break; // we stop after we have processed LINES lines
      }

      if(fgets(buffer, BUFLEN, fp) != buffer){
	break; // silently fails if we cannot read new line
      }
      
      // intepretes buffer as a vector
      math::vertex<T> line;
      char* s = buffer;
      unsigned int index = 0;

      while(*s == ' ' || *s == ',' || *s == ';') s++;
    
      while(*s != '\n' && *s != '\0' && *s != '\r'){
	char* prev = s;
	double v = strtod(s, &s);
	if(s == prev){
	  break; // no progress
	}
	
	if(isnan(v) || isinf(v))
	  break; // bad data
      
	line.resize(index+1);
	line[index] = v;
	index++;
	
	while(*s == ' ' || *s == ',' || *s == ';') s++;
      }

      if(import.size() > 0 && line.size() > 0){
	if(line.size() != import[0].size()){ // number of dimensions must match for all lines
	  fclose(fp);
	  free(buffer);
	  return false; // we just give up if there is strange/bad file
	}
      }

      if(line.size() > 0){
	import.push_back(line);
	lines++;
      }
    }
    
    free(buffer);
    fclose(fp);

    if(import.size() <= 0)
      return false;

    // clears cluster 0 and adds new data
    if(this->getNumberOfClusters() > 0){
      this->clearAll(0);

      typename std::map<std::string, unsigned int>::const_iterator i;
      i = namemapping.find(clusters[0].cname);
      
      std::string name = "data import";
      clusters[0].data_dimension = import[0].size();
      clusters[0].cname = name;
      clusters[0].cindex = 0;
      
      if(i != namemapping.end())
	namemapping.erase(i);
      
      namemapping[name] = 0;
      clusters[0].data = import;

      return true;
    }
    else{
      if(this->createCluster("data import", import[0].size()) == false)
	return false;

      if(this->add(0, import) == false){
	this->removeCluster(0);
	return false;
      }
      
      return true;
    }
  }
  
  
  // accesses zero cluster
  template <typename T>
  const math::vertex<T>& dataset<T>::operator[](unsigned int index) const
    throw(std::out_of_range)
  {
    if(clusters.size() == 0){
      throw std::out_of_range("dataset: cluster zero doesn't exist.");
    }
    else if(index >= clusters[0].data.size()){
      throw std::out_of_range("dataset: index out of range");
    }
    
    return clusters[0].data[index];
  }
  
  template <typename T>
  const math::vertex<T>& dataset<T>::access(unsigned int cluster, unsigned int data) const throw(std::out_of_range)
  {
	    if(cluster >= clusters.size())
	      throw std::out_of_range("cluster index out of range");

	    if(data >= clusters[cluster].data.size())
	      throw std::out_of_range("data index out of range");

	    return clusters[cluster].data[data];
  }
  
  
  template <typename T>
  const math::vertex<T>& dataset<T>::accessName(const std::string& clusterName, unsigned int dataElem) throw(std::out_of_range)
  {
            typename std::map<std::string, unsigned int>::const_iterator i;
	    i = namemapping.find(clusterName);

	    if(i == namemapping.end())
	      throw std::out_of_range("dataset: cannot find cluster name");

	    const unsigned int cluster = i->second;

	    if(dataElem >= clusters[cluster].data.size())
	      throw std::out_of_range("data index out of range");

	    return clusters[cluster].data[dataElem];
  }


  template <typename T>
  unsigned int dataset<T>::size(unsigned int index) const throw()  // dataset size  
  {
    if(index >= clusters.size())
      return (unsigned int)(-1);
    
    return clusters[index].data.size();
  }
  
  template <typename T>
  bool dataset<T>::clear(unsigned int index) throw()  // data set clear  
  {
    if(index >= clusters.size())
      return false;
    
    clusters[index].data.clear();
    return true;
  }
  
  template <typename T>
  unsigned int dataset<T>::dimension(unsigned int index) const throw()  // dimension of data vectors
  {
    if(index >= clusters.size())
      return (unsigned int)(-1);
    
    return clusters[index].data_dimension;
  }
  
  
  template <typename T>
  bool dataset<T>::getPreprocessings(unsigned int cluster,
				     std::vector<data_normalization>& preprocessings) const throw()
  {
    if(cluster >= clusters.size())
      return false;
    
    preprocessings.clear();
    preprocessings = clusters[cluster].preprocessings;
    
    return true;
  }
  
  
  // data preprocessing
  template <typename T>
  bool dataset<T>::preprocess(unsigned int index, enum data_normalization norm) throw()
  {
    if(index >= clusters.size())
      return false;
    
    try{
      if(norm == dnMeanVarianceNormalization){
	if(is_normalized(index, dnMeanVarianceNormalization))
	  return true;
	
	// sets up mean&variance vectors
	{
	  clusters[index].mean.resize(clusters[index].data_dimension);
	  clusters[index].variance.resize(clusters[index].data_dimension);
	  
	  clusters[index].mean.zero();
	  clusters[index].variance.zero();
	}
	
	// calculates mean & variance
	// (calculates first E[X] and E[X^2]
	// -> mean,variance (almost correct)
	{
	  typename std::vector<math::vertex<T> >::iterator i;
	  
	  i = clusters[index].data.begin();
	  
	  while(i != clusters[index].data.end()){
	    clusters[index].mean += (*i);
	    
	    for(unsigned int k=0;k<clusters[index].data_dimension;k++)
	      clusters[index].variance[k] += ((*i)[k]) * ((*i)[k]);
	    
	    i++;
	  }
	  
	  // calculates E[X] and sqrt(Var[X]) = sqrt(E[X^2] - E[X]**2)
	  
	  if(clusters[index].data.size() > 0){
	    clusters[index].mean /= T(clusters[index].data.size());
	    clusters[index].variance /= T(clusters[index].data.size());
	  }
	  
	  // not needed
	  //mean.resize(data_dimension);
	  //variance.resize(data_dimension);
	  
	  
	  {
	    unsigned int k = 0;
	    
	    while(k < clusters[index].data_dimension){
	      if(k < clusters[index].data_dimension){
		clusters[index].variance[k] -= (clusters[index].mean[k])*(clusters[index].mean[k]);
			    
		if(clusters[index].variance[k] < 0.0)
		  clusters[index].variance[k] = T(0.0);
		
		clusters[index].variance[k]  = whiteice::math::sqrt(clusters[index].variance[k]);
		k++;
	      }
	      else{
		break;
	      }
	    }
	    
	  }
	  
	  
	}
	
	{
	  typename std::vector< math::vertex<T> >::iterator i;
	  
	  i = clusters[index].data.begin();
	  while(i != clusters[index].data.end()){
	    mean_variance_removal(index, *i);
	    i++;
	  }
	}
	
	clusters[index].preprocessings.push_back(dnMeanVarianceNormalization);
	return true;
      }
      else if(norm == dnSoftMax){
	const T r = T(0.80);

	// softmax code here requires that we have something like N(0,1) distributed data
	if(is_normalized(index, dnMeanVarianceNormalization) == false){
	  if(preprocess(index, dnMeanVarianceNormalization) == false){
	    return false;
	  }
	}
	
	if(is_normalized(index, dnSoftMax))
	  return true;
	
	clusters[index].softmax_parameter = r;
	
	// soft max
	{
	  typename std::vector< math::vertex<T> >::iterator i;
	  i = clusters[index].data.begin();
	  
	  while(i != clusters[index].data.end()){
	    soft_max(index, *i);
	    i++;
	  }
	}
	
	clusters[index].preprocessings.push_back(dnSoftMax);
	return true;
      }
      else if(norm == dnCorrelationRemoval){
	
	if(is_normalized(index, dnCorrelationRemoval))
	  return true;
	
	if(is_normalized(index, dnMeanVarianceNormalization) == false)
	  if(preprocess(index, dnMeanVarianceNormalization) == false)
	    return false;
	
	// we can use autocorrelation because mean is already zero
	if(autocorrelation(clusters[index].Rxx, clusters[index].data) == false){
		clusters[index].Rxx.resize(clusters[index].data_dimension, clusters[index].data_dimension);
		clusters[index].Rxx.identity();
	}
	
	// std::cout << "Rxx = " << clusters[index].Rxx << std::endl;
	
	math::matrix<T> V, Vt, invD, D(clusters[index].Rxx);
	
	if(symmetric_eig(D, V) == false){
		D.resize(clusters[index].data_dimension, clusters[index].data_dimension);
		D.identity();
		V.resize(clusters[index].data_dimension, clusters[index].data_dimension);
		V.identity();
	}
	
	// std::cout << "typeinfo = " << typeid(T).name() << std::endl;
	// std::cout << "D = " << D << std::endl;
	// std::cout << "V = " << V << std::endl;
	
	invD = D;
	
	for(unsigned int i=0;i<invD.ysize();i++){
	  T d = invD(i,i);
	  
	  if(d > T(10e-8)){
	    invD(i,i) = whiteice::math::sqrt(T(1.0)/whiteice::math::abs(d));
	    D(i,i)    = whiteice::math::sqrt(whiteice::math::abs(d));
	  }
	  else{
	    invD(i,i) = T(0.0f);
	    D(i,i)    = T(0.0f);
	  }
	  
	}
	
	
	// std::cout << "invD = " << invD << std::endl;
	
	Vt = V;
	Vt.transpose();
	
	clusters[index].Wxx = V * invD * Vt;
	clusters[index].invWxx = V * D * Vt;

	// std::cout << "Wxx      = " << clusters[index].Wxx << std::endl;
	// std::cout << "inv(Wxx) = " << clusters[index].invWxx << std::endl;
	
	typename std::vector< math::vertex<T> >::iterator i;
	i = clusters[index].data.begin();
	
	while(i != clusters[index].data.end()){	  
	  whiten(index, *i);
	  i++;
	}
	
	
	clusters[index].preprocessings.push_back(dnCorrelationRemoval);
	return true;
      }
      else if(norm == dnLinearICA)
      {
	if(is_normalized(index, dnLinearICA))
	  return true; // already preprocessed

	if(clusters[index].data.size() <= 1)
	  return false; // cannot work

	// we must have correlation removal so do it first
	if(is_normalized(index, dnCorrelationRemoval) == false)
	  if(preprocess(index, dnCorrelationRemoval) == false)
	    return false;

	if(math::ica(clusters[index].data, clusters[index].ICA) == false){
	  std::cout << "Calculating independent component analysis failed."
		    << std::endl;
	  return false;
	}
	else{
	  // calcualtes inverse of ICA
	  clusters[index].invICA = clusters[index].ICA;
	  if(clusters[index].invICA.inv() == false){
	    std::cout << "Cannot compute inverse of ICA." << std::endl;
	    std::cout << "ICA = " << clusters[index].ICA << std::endl;
	    return false; // calculating of ICA preprocessing failed.
	  }
	    
	}

	typename std::vector< math::vertex<T> >::iterator i;
	i = clusters[index].data.begin();
	
	while(i != clusters[index].data.end()){
	  ica(index, (*i));
	  i++;
	}
	
	clusters[index].preprocessings.push_back(dnLinearICA);
	return true;
      }
      else return false;
    }
    catch(std::exception& e){
      std::cout << "dataset::preprocess: fatal error: " << e.what() << std::endl;
      return false;
    }
  }
  
  
  template <typename T>
  bool dataset<T>::preprocess(enum data_normalization norm) throw(){
    return preprocess(0, norm);
  }
  
  
  
  // inverse preprocess everything, calculates new preprocessing parameters
  // and preprocesses everything with parameter data from the whole dataset
  // (dataset may grow after preprocessing)
  template <typename T>
  bool dataset<T>::repreprocess(unsigned int index) throw()
  { 
    if(index >= clusters.size())
      return false;
    
    // saves preprocesings
    std::vector<enum data_normalization > old_preprocessings;
    typename std::vector<enum data_normalization >::iterator i;
    
    for(i=clusters[index].preprocessings.begin();
	i!=clusters[index].preprocessings.end();i++)
      old_preprocessings.push_back(*i);
    
    // inverse preprocess whole data
    if(!invpreprocess(index, clusters[index].data))
      return false;  
    
    clusters[index].preprocessings.clear();
    
    // apply preprocessings to whole data again
    for(i = old_preprocessings.begin();i!=old_preprocessings.end();i++){
      if(!preprocess(index, *i))
	return false;
    }
    
    return true;
  }
  
  
  // converts data with same preprocessing as with dataset vectors
  template <typename T>
  bool dataset<T>::preprocess(unsigned int index, math::vertex<T>& vec) const throw()
  {
    if(index >= clusters.size()) // this is slow (optimize internal calls)
      return false;
    
    typename std::vector<enum data_normalization>::const_iterator i;
    
    for(i=clusters[index].preprocessings.begin();i!=clusters[index].preprocessings.end();i++){
      
      if(*i == dnCorrelationRemoval){
    	  whiten(index, vec);
      }
      else if(*i == dnMeanVarianceNormalization){
    	  mean_variance_removal(index, vec);
      }
      else if(*i == dnSoftMax){
    	  soft_max(index, vec);
      }
      else return false;
    }
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::preprocess(unsigned int index,
			      std::vector< math::vertex<T> >& group) const throw()
  {
    if(index >= clusters.size())
      return false;
    
    typename std::vector< math::vertex<T> >::iterator i;
    
    for(i=group.begin();i!=group.end();i++){
      if(!preprocess(index, *i)) return false;
    }
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::preprocess(math::vertex<T>& vec) const throw(){
    return preprocess(0, vec);
  }
  
  template <typename T>
  bool dataset<T>::preprocess(std::vector< math::vertex<T> >& group) const throw(){
    return preprocess(0, group);
  }
  
  
  // inverse preprocess given data vector
  template <typename T>
  bool dataset<T>::invpreprocess(unsigned int index, math::vertex<T>& vec) const throw()
  {
    if(index >= clusters.size()) // this is slow (optimize internal calls)
      return false;

    
    typename std::vector<enum data_normalization>::const_reverse_iterator i;
    
    for(i=clusters[index].preprocessings.rbegin();
	i!=clusters[index].preprocessings.rend();i++){
      if(*i == dnLinearICA){
	inv_ica(index, vec);
      }
      else if(*i == dnCorrelationRemoval){
	inv_whiten(index, vec);
      }
      else if(*i == dnMeanVarianceNormalization){
	inv_mean_variance_removal(index, vec);
      }
      else if(*i == dnSoftMax){
	inv_soft_max(index, vec);
      }
      else return false;
    }

    return true;  
  }
  
  
  template <typename T>
  bool dataset<T>::invpreprocess(unsigned int index,
				 std::vector<math::vertex<T> >& group) const throw()
  {
    if(index >= clusters.size())
      return false;
    
    typename std::vector< math::vertex<T> >::iterator i;
    
    for(i=group.begin();i!=group.end();i++)
      if(!invpreprocess(index, *i)) return false;
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::invpreprocess(math::vertex<T>& vec) const throw(){
    return invpreprocess(0, vec);
  }
  
  
  template <typename T>
  bool dataset<T>::invpreprocess(std::vector< math::vertex<T> >& group) const throw(){
    return invpreprocess(0, group);
  }
  
  
  // removes all preprocessings from data
  template <typename T>
  bool dataset<T>::convert(unsigned int index) throw()
  {
    if(index >= clusters.size())
      return false;

    // inverse preprocess whole data
    if(!invpreprocess(index, clusters[index].data)) return false;
    clusters[index].preprocessings.clear();
    
    return true;
  }
  
  
  template <typename T>
  bool dataset<T>::convert(unsigned int index,
			   std::vector<enum data_normalization> plist)
  {
    if(index >= clusters.size())
      return false;
    
    // checks that list has only legitime values
    // and it doesn't have duplicates
    
    for(unsigned int i=0;i<plist.size();i++){
      if(plist[i] != dnMeanVarianceNormalization &&
	 plist[i] != dnSoftMax && 
	 plist[i] != dnCorrelationRemoval &&
	 plist[i] != dnLinearICA) return false;
      
      for(unsigned int j=0;j<plist.size();j++){
	if(i == j) continue;
	if(plist[i] == plist[j]) return false;
      }
    }
    
    // checks how far the both lists are same and 
    // invert only preprocessings that don't match
    
    {
      unsigned int i=0;
      while(i < plist.size() && i < clusters[index].preprocessings.size()){
	if(plist[i] != clusters[index].preprocessings[i])
	  break;
	else i++;
      }
      
      typename std::vector< math::vertex<T> >::iterator k;
      
      for(unsigned int j=i;j<clusters[index].preprocessings.size();j++){
	if(clusters[index].preprocessings[j] == dnMeanVarianceNormalization){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    inv_mean_variance_removal(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnSoftMax){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    inv_soft_max(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnCorrelationRemoval){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    inv_whiten(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnLinearICA){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    inv_ica(index, *k);
	}	
      }
      
      // preprocesses with plist[restoflist]
      
      for(unsigned int j=i;j<plist.size();j++){
	if(clusters[index].preprocessings[j] == dnMeanVarianceNormalization){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    mean_variance_removal(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnSoftMax){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    soft_max(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnCorrelationRemoval){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    whiten(index, *k);
	}
	else if(clusters[index].preprocessings[j] == dnLinearICA){
	  for(k=clusters[index].data.begin();k!=clusters[index].data.end();k++)
	    ica(index, *k);
	}
      }
      
      
      clusters[index].preprocessings = plist;
    }
    
    
    return true;
  }
  
  
  /**************************************************/
  
  // is data normalized with given operator?
  template <typename T>
  bool dataset<T>::is_normalized(unsigned int index,
				 enum data_normalization norm) const throw()
  {
    typename std::vector<enum data_normalization>::const_iterator i;
    
    for(i=clusters[index].preprocessings.begin();
	i!=clusters[index].preprocessings.end();i++){
      if(*i == norm) return true;
    }
    
    return false;
  }
  
  
  
  // sets variance to be something like 0.25**2 -> good input/outputs for nn
  template <typename T>
  void dataset<T>::mean_variance_removal(unsigned int index, math::vertex<T>& vec) const
  {
    // x = (x - mean)/sqrt(var) -> new var = 1.0
    
    vec -= clusters[index].mean;
    
    for(unsigned int i=0;i<vec.size();i++){
      if(clusters[index].variance[i] > T(10e-8))
	vec[i] /= (T(4.0)*clusters[index].variance[i]);
    }
    
  }
  
  
  template <typename T>
  void dataset<T>::inv_mean_variance_removal(unsigned int index,
					     math::vertex<T>& vec) const
  {
    // [x' * sqrt(var)] + mean
    
    for(unsigned int i=0;i<vec.size();i++){
      if(clusters[index].variance[i] > T(10e-8))
	vec[i] *= (T(4.0)*clusters[index].variance[i]);
    }
    
    vec += clusters[index].mean;
  }
  
  
  template <typename T>
  void dataset<T>::soft_max(unsigned int index,
			    math::vertex<T>& vec) const
  {
    const unsigned int N = vec.size();
    
    for(unsigned int i=0;i<N;i++)
      vec[i] = T(1.0) / (T(1.0) + whiteice::math::exp( -(vec[i]/clusters[index].softmax_parameter) ));
    
  }
  
  
  template <typename T>
  void dataset<T>::inv_soft_max(unsigned int index,
				math::vertex<T>& vec) const
  {
    const unsigned int N = vec.size();
    
    for(unsigned int i=0;i<N;i++)
      vec[i] = clusters[index].softmax_parameter * whiteice::math::log( T(vec[i]) / (T(1.0) - vec[i]) );
  }
  
  
  template <typename T>
  void dataset<T>::whiten(unsigned int index, math::vertex<T>& vec) const
  {
	  vec = clusters[index].Wxx * vec;
  }
  
  
  template <typename T>
  void dataset<T>::inv_whiten(unsigned int index,
			      math::vertex<T>& vec) const
  {
	  vec = clusters[index].invWxx * vec;
  }



  template <typename T>
  void dataset<T>::ica(unsigned int index,
		       math::vertex<T>& vec) const
  {
    vec = clusters[index].ICA * vec;
  }
  
  
  template <typename T>
  void dataset<T>::inv_ica(unsigned int index,
			   math::vertex<T>& vec) const
  {
    
    vec = clusters[index].invICA * vec;
  }  
  
  
  
  //////////////////////////////////////////////////////////////////////
  
  
  template class dataset< whiteice::math::blas_real<float> >;
  template class dataset< whiteice::math::blas_real<double> >;
  template class dataset< float >;
  template class dataset< double >;    
  
}
  
#endif
