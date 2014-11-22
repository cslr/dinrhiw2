
#ifndef neuralnetwork_cpp
#define neuralnetwork_cpp

#include "neuralnetwork.h"
#include "neuronlayer.h"
#include "conffile.h"
#include <assert.h>
#include <typeinfo>

#include "odd_sigmoid.h"
#include "identity_activation.h"

#include <stdio.h>


namespace whiteice
{
  
  template <typename T>
  neuralnetwork<T>::neuralnetwork()
  {
    // creates 1x1 network
    
    /* allocates / creates data */
    
    input_values.resize(1);
    output_values.resize(1);
    state.resize(1);
    this->layers.resize(1);
    
    /* initializes data */
    
    for(unsigned int i=0;i<1;i++){
      input_values[i] = 0;
      output_values[i] = 0;
      state[i] = 0;
    }
    
    for(unsigned int i=0;i<1;i++){
      this->layers[i] = new neuronlayer<T>(1, 1);
      this->layers[i]->input() = &state;
      this->layers[i]->output() = &state;
    }
    
    this->layers[0]->input() = &input_values;
    this->layers[0]->output() = &output_values;
    
    this->randomize();
    
    this->compressed = false;
  }
  
  
  template <typename T>
  neuralnetwork<T>::neuralnetwork(const neuralnetwork<T>& nn)
  {  
    input_values.resize(nn.input_values.size());  
    output_values.resize(nn.output_values.size());
    state.resize(nn.state.size());
    
    state = nn.state;
    
    input_values = nn.input_values;
    output_values = nn.output_values;
    
    this->layers.resize(nn.layers.size());
    
    for(unsigned int i=0;i<layers.size();i++){
      layers[i] = new neuronlayer<T>(*nn.layers[i]);
      
      layers[i]->input() = &state;
      layers[i]->output() = &state;      
    }
    
    this->layers[0]->input() = &input_values;
    this->layers[layers.size()-1]->output() = &output_values;
    
    this->compressed = nn.compressed;
  }
  
  
  
  // L*W rectangle network
  template <typename T>
  neuralnetwork<T>::neuralnetwork(unsigned int layers,
				  unsigned int width)
  {
    /* allocates / creates data */
    
    input_values.resize(width);
    output_values.resize(width);
    state.resize(width);
    this->layers.resize(layers);
    
    /* initializes data */
    
    for(unsigned int i=0;i<width;i++){
      input_values[i] = 0;
      output_values[i] = 0;
      state[i] = 0;
    }
    
    for(unsigned int i=0;i<layers;i++){
      this->layers[i] = new neuronlayer<T>(width, width);
      this->layers[i]->input() = &state;
      this->layers[i]->output() = &state;
    }
    

    // final layer has identity activation
    // (default) this means results
    // linear combination of the previous
    // layer's values
    {
      identity_activation<T> id;
      this->layers[this->layers.size()-1]->set_activation(id);
    }
    
    
    this->layers[0]->input() = &input_values;
    this->layers[layers-1]->output() = &output_values;
    
    this->randomize();
    
    this->compressed = false;
  }
  
  
  
  template <typename T>
  neuralnetwork<T>::neuralnetwork(const std::vector<unsigned int>& nn_structure,
				  bool compressed_network)
    throw(std::invalid_argument)
  {
    std::invalid_argument e("empty/bad network architecture");
    
    if(nn_structure.size() < 2)
      throw e;
    
    // finds out first, max, last width of network (for input, state and output vectors)
    
    std::vector<unsigned int>::const_iterator i = nn_structure.begin();
    
    unsigned int max = 0, first = 0, last = 0;
    first = *i;
    
    while(i != nn_structure.end()){
      
      if(*i == 0) throw e;
      else if(*i > max) max = *i;
      last = *i;
      
      i++;
    }
    
    input_values.resize(first);
    output_values.resize(last);
    state.resize(max);
    layers.resize(nn_structure.size()-1);
    
    // allocates layers and setups input/outputs for neuralnetwork
    
    typename std::vector< neuronlayer<T>* >::iterator j;
    typename std::vector<unsigned int>::const_iterator k;
    
    i = nn_structure.begin();
    k = nn_structure.begin();
    k++;
    
    j = layers.begin();
    
    // 'debugging'
    int temp = 0;
    
    while(k != nn_structure.end()){
      *j = new neuronlayer<T>(*i, *k);
      if(compressed_network) (*j)->compress();
	
      (*j)->input() = &state;
      (*j)->output() = &state;
      
      i++;
      j++;
      k++;
      temp++;
    }
    
    j--; /* last layer */
    (*layers.begin())->input() = &input_values;
    (*j)->output() = &output_values;
    
    
    // final layer has identity activation
    // (default) this means results
    // linear combination of the previous
    // layer's values
    {
      identity_activation<T> id;
      this->layers[layers.size()-1]->set_activation(id);
    }    
    
    this->randomize();
    
    this->compressed = compressed_network;
  }
  
  
  template <typename T>
  neuralnetwork<T>::~neuralnetwork()
  {
    typename std::vector< neuronlayer<T>* >::iterator i = 
      layers.begin();
    
    while(!layers.empty()){
      
      delete (*i);
      i = layers.erase(i);
    }
    
  }
  
  
  
  template <typename T>
  neuralnetwork<T>& neuralnetwork<T>::operator=(const neuralnetwork<T>& nn)
  {
    assert(0);	// implement me
  }
  
  // other stuff to create more complex nn structures
  /* accessing data / configuration */
  
  template <typename T>
  math::vertex<T>& neuralnetwork<T>::input() throw()
  {
    return input_values;
  }
  
  
  template <typename T>
  math::vertex<T>& neuralnetwork<T>::output() throw()
  {
    return output_values;
  }
  
  
  /*
   * calculates output of neural network
   */
  template <typename T>
  bool neuralnetwork<T>::calculate()
  {
    typename std::vector< whiteice::neuronlayer<T>* >::iterator i;
    
    for(i=layers.begin();i!=layers.end();i++){
      
      if((*i)->iscompressed())
	if((*i)->decompress() == false) return false;
      
      (*i)->calculate();
      
      if(compressed) (*i)->compress();
      
    }
    
    return true;
  }
  
  
  /* 
   * calculates output of neural network
   */
  template <typename T>
  bool neuralnetwork<T>::operator()()
  {
    return calculate();
  }
  
  /*
   * returns number of layers in nn
   */
  template <typename T>
  unsigned int neuralnetwork<T>::length() const // number of layers
  {
    return layers.size();
  }
  
  
  template <typename T>
  bool neuralnetwork<T>::randomize()
  {
    typename std::vector< neuronlayer<T>* >::iterator i;
    bool result = true;
    
    // 'debugging'
    int temp = 0;
    
    for(i=layers.begin();i!=layers.end();i++){
      if((*i)->iscompressed()) 
	if((*i)->decompress() == false) return false;
      
      if((*i)->randomize() == false)
	result = false;
      
      if(compressed) (*i)->compress();
      
      temp++;      
    }
    
    return result;
  }
  

  template <typename T>
  bool neuralnetwork<T>::setlearningrate(float rate)
  {
    typename std::vector< neuronlayer<T>* >::iterator i =
      layers.begin();
    
    while(i != layers.end()){
      (*i)->learning_rate() = rate;
      i++;
    }
    
    return true;
  }
  

  template <typename T>
  neuronlayer<T>& neuralnetwork<T>::operator[](unsigned int index) throw(std::out_of_range)
  {
    if(index >= layers.size()){
      std::out_of_range e("neuralnetwork<T>:: index too big");
      throw(e);
    }
    
    // TODO for code-cleaning:
    // does layers throw out_of_range etc. exception itself if index is too big!?!? - probably
    return (*layers[index]);
    
  }


  //////////////////////////////////////////////////////////////////////
  
#define FNN_VERSION_CFGSTR          "FNN_CONFIG_VERSION"
#define FNN_ARCH_CFGSTR             "FNN_ARCH"
#define FNN_LAYER_W_CFGSTR          "FNN_LWEIGHT%d"
#define FNN_LAYER_B_CFGSTR          "FNN_LBIAS%d"
#define FNN_AFUNS_CFGSTR            "FNN_AFUNCTIONS" // symbolic names for activation function
#define FNN_LAYER_AFUN_PARAM_CFGSTR "FNN_LAFPARAM%d"
#define FNN_MOMENS_CFGSTR           "FNN_MOMENTS"
#define FNN_LRATES_CFGSTR           "FNN_LRATES"
  
  
  // load & saves neuralnetwork data from file
  template <typename T>
  bool neuralnetwork<T>::load(const std::string& filename) throw()
  {
    if(compressed) return false;
    
    // floats must be castable to T
    try{
      whiteice::conffile configuration;
      
      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      
      if(!configuration.load(filename))
	return false;
      
      
      int versionid;
      
      // checks version
      {
	if(!configuration.get(FNN_VERSION_CFGSTR, ints))
	  return false;
	
	if(ints.size() != 1)
	  return false;
	
	versionid = ints[0];
	
	ints.clear();
      } 
      
      if(versionid != 100)
	return false;
      
      
      // gets architecture
      {
	if(!configuration.get(FNN_ARCH_CFGSTR,ints))
	  return false;
	  
	
	if(ints.size() < 2)
	  return false;
	
	input_values.resize(ints[0]);
	output_values.resize(ints[ints.size() - 1]);
	
	layers.resize(ints.size() - 2);
	
	if(layers.size() > 0){
	  for(unsigned int i=0,j=1;i<layers.size();i++,j++){
	    layers[i] = new neuronlayer<T>(ints[j-1],ints[j]);
	    layers[i]->input() = &state;
	    layers[i]->output() = &state;
	  }
	  
	  layers[0]->input() = &input_values;
	  layers[layers.size()-1]->output() = &output_values;
	}
	
	// resizes state
	for(unsigned int i=0;i<ints.size();i++)
	  if(state.size() < (unsigned)ints[i])
	    state.resize(ints[i]);
	
	ints.clear();
      }
      
      
      // gets layers weights
      {
	char str[100];

	for(unsigned int i=0;i<layers.size();i++){
	  sprintf(str, FNN_LAYER_W_CFGSTR, i);
	  
	  if(!configuration.get(str, floats))
	    return false;
	  
	  math::matrix<T>& W = layers[i]->weights();
	  
	  if(floats.size() != W.ysize()*W.xsize())
	    return false;
	  
	  unsigned int j = 0;
	  
	  for(unsigned int y=0;y<W.ysize();y++){
	    for(unsigned int x=0;x<W.xsize();x++,j++){
	      W(y,x) = floats[j];
	    }
	  }
	  
	  floats.clear();
	}
      }
      
            
      // gets biases
      {
	char str[100];
	
	for(unsigned int i=0;i<layers.size();i++){
	  sprintf(str, FNN_LAYER_B_CFGSTR, i);
	  
	  if(!configuration.get(str, floats))
	    return false;
	  
	  math::vertex<T>& b = layers[i]->bias();
	  
	  if(floats.size() != b.size())
	    return false;
	  
	  for(unsigned int x=0;x<b.size();x++)
	    b[x] = floats[x];
	  
	  floats.clear();
	}
      }
      


      
      // layer activation function parameters
      {
	// only odd_sigmoid and identity activation
	
	char str[100];
	
	for(unsigned int i=0;i<layers.size();i++){
	  
	  const activation_function<T>* a = 
	    layers[i]->get_activation();
	  
	  
	  try{ // tries to dynamic cast to a odd sigmoid
	    
	    const odd_sigmoid<T>* o = 
	      dynamic_cast< const odd_sigmoid<T>* >(a);
	    
	    if(o == NULL)
	      throw std::bad_cast();
	    
	    float f; if(!math::convert(f, o->get_scale())) return false;
	    floats.push_back(f);
	    
	    if(!math::convert(f, o->get_alpha())) return false;
	    floats.push_back(f);
	    
	  }
	  catch(std::exception& e){
	    
	    // tries to dynamic_cast to identity
	    try{
	      
	      const identity_activation<T>* ia =
		dynamic_cast< const identity_activation<T>* >(a);
	      
	      if(ia == 0)
		throw std::bad_cast();
	      
	      floats.push_back(1.0f); // fake param
	      
	    }
	    catch(std::exception& e){
	      
	      return false;
	    }
	  }	
	  
	  
	  sprintf(str, FNN_LAYER_AFUN_PARAM_CFGSTR, i);
	  

	  
	  floats.clear();
	}
      }

      
      // gets activation functions
      // gets activation function parameters
      {
	// only odd_sigmoid and identity activation
	
	char str[100];
	
	if(!configuration.get(FNN_AFUNS_CFGSTR, strings))
	  return false;
	
	if(strings.size() != layers.size())
	  return false;
	
	
	for(unsigned int i=0;i<layers.size();i++){
	  
	  identity_activation<T> ia;
	  odd_sigmoid<T>* os;
	  
	  sprintf(str, FNN_LAYER_AFUN_PARAM_CFGSTR, i);
	  if(!configuration.get(str, floats))
	    return false;
	  
	  if(strings[i] == "odd sigmoid"){
	    if(floats.size() != 2)
	      return false;
	    
	    os = new odd_sigmoid<T>(floats[0], floats[1]);
	    
	    if(!layers[i]->set_activation(*os))
	      return false;
	    
	    delete os;
	  }
	  else if(strings[i] == "identity"){
	    if(!layers[i]->set_activation(ia))
	      return false;
	  }
	  else
	    return false;
	  
	  
	  floats.clear();
	}

	strings.clear();
      }
      
      
      
      
      // gets layer moments
      {
	if(!configuration.get(FNN_MOMENS_CFGSTR, floats))
	  return false;
	
	if(floats.size() != layers.size())
	  return false;
	
	
	for(unsigned int i=0;i<layers.size();i++)
	  layers[i]->moment() = floats[i];
	
	floats.clear();
      }
      
      
      // gets layer learning rates
      {
	if(!configuration.get(FNN_LRATES_CFGSTR, floats))
	  return false;
	
	if(floats.size() != layers.size())
	  return false;
	
	for(unsigned int i=0;i<layers.size();i++)
	  layers[i]->learning_rate() = floats[i];
	
	floats.clear();
      }      
      
      
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
  bool neuralnetwork<T>::save(const std::string& filename) const throw()
  {
    if(compressed) return false;
    
    // T must be castable to floats
    try{
      whiteice::conffile configuration;
      
      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      
      // writes version information
      {
	// version number = integer/1000
	ints.push_back(100); // 0.100
	if(!configuration.set(FNN_VERSION_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }
      
      // writes architecture information
      {
	ints.push_back(input_values.size());
	
	if(layers.size() > 0)
	  for(unsigned int i=0;i<layers.size();i++){
	    ints.push_back( layers[i]->size() );
	  }
	
	ints.push_back(output_values.size());
	
	if(!configuration.set(FNN_ARCH_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }
      
      if(layers.size() <= 0){
	return configuration.save(filename);
      }
      
      // writes layer specific information
      
      std::cout << "WEIGHTS" << std::endl;
      
      // layer weights
      {
	char str[100];

	for(unsigned int i=0;i<layers.size();i++){
	  const math::matrix<T>& W = layers[i]->weights();
	  
	  for(unsigned int y=0;y<W.ysize();y++){
	    for(unsigned int x=0;x<W.xsize();x++){
	      std::cout << "(y,x) = (" << y << "," << x << ")" << std::endl;
	      float f; if(!math::convert(f, W(y,x))) return false;
	      floats.push_back(f);
	    }
	  }
	  
	  sprintf(str, FNN_LAYER_W_CFGSTR, i);
	  
	  if(!configuration.set(str, floats))
	    return false;
	  
	  floats.clear();
	}
      }
      
      std::cout << "BIASES" << std::endl;
      
      // layer biases
      {
	char str[100];
	
	for(unsigned int i=0;i<layers.size();i++){
	  const math::vertex<T>& b = layers[i]->bias();
	  
	  for(unsigned int x=0;x<b.size();x++){
	    float f; if(!math::convert(f, b[x])) return false;
	    floats.push_back(f);
	  }
	  
	  sprintf(str, FNN_LAYER_B_CFGSTR, i);
	  
	  if(!configuration.set(str, floats))
	    return false;
	  
	  floats.clear();
	}
      }
      
      std::cout << "LAYER FUNCS" << std::endl;
      
      // layer activation functions
      {
	// only odd_sigmoid and identity activation
	
	for(unsigned int i=0;i<layers.size();i++){
	  
	  const activation_function<T>* a = 
	    layers[i]->get_activation();
	  
	  try{ // tries to dynamic cast to a odd sigmoid
	    const odd_sigmoid<T>* ptr = dynamic_cast< const odd_sigmoid<T>* >(a);
	    
	    if(ptr == NULL) 
	      throw std::bad_cast();
	    
	    strings.push_back("odd sigmoid");
	  }
	  catch(std::exception& e){
	    
	    // tries to dynamic_cast to identity
	    try{
	      const identity_activation<T>* ptr =
		dynamic_cast< const identity_activation<T>* >(a);
	      
	      if(ptr == NULL)
		throw std::bad_cast();
	      
	      strings.push_back("identity");
	    }
	    catch(std::exception& e){
	      return false;
	    }
	  }
	  
	}
	
	
	if(!configuration.set(FNN_AFUNS_CFGSTR, strings))
	  return false;
	
	strings.clear();
      }
      
      
      std::cout << "FUNC PARAMS" << std::endl;
      
      
      // layer activation function parameters
      {
	// only odd_sigmoid and identity activation
	
	char str[100];
	
	for(unsigned int i=0;i<layers.size();i++){
	  
	  const activation_function<T>* a = 
	    layers[i]->get_activation();
	  
	  
	  try{ // tries to dynamic cast to a odd sigmoid
	    
	    const odd_sigmoid<T>* o = 
	      dynamic_cast< const odd_sigmoid<T>* >(a);

	    if(o == NULL)
	      throw std::bad_cast();
	    
	    
	    float f; if(!math::convert(f, o->get_scale())) return false;
	    floats.push_back(f);
	    
	    if(!math::convert(f, o->get_alpha())) return false;
	    floats.push_back(f);
	    
	  }
	  catch(std::exception& e){
	    
	    // tries to dynamic_cast to identity
	    try{
	      
	      const identity_activation<T>* ia =
		dynamic_cast< const identity_activation<T>* >(a);
	      
	      if(ia == NULL)
		throw std::bad_cast();
	      
	      ia = 0;
	      
	      floats.push_back(1.0f); // fake param
	      
	    }
	    catch(std::exception& e){
	      
	      return false;
	    }
	  }	
	  
	  
	  sprintf(str, FNN_LAYER_AFUN_PARAM_CFGSTR, i);
	  
	  if(!configuration.set(str, floats))
	    return false;
	  
	  floats.clear();
	}
      }
      
      std::cout << "MOMENTS" << std::endl;
      
      // layer moments
      {
	for(unsigned int i=0;i<layers.size();i++){
	  float f; if(!math::convert(f, layers[i]->moment())) return false;
	  floats.push_back( f );
	}
	
	if(!configuration.set(FNN_MOMENS_CFGSTR, floats))
	  return false;
	
	floats.clear();
      }
      
      std::cout << "LEARNING RATES" << std::endl;
      
      // layer learning rates
      {
	for(unsigned int i=0;i<layers.size();i++){
	  float f; if(!math::convert(f, layers[i]->learning_rate())) return false;
	  floats.push_back( f );
	}
	
	if(!configuration.set(FNN_LRATES_CFGSTR, floats))
	  return false;
	
	floats.clear();
      }
      
      std::cout << "END" << std::endl;
      
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
  
  

#undef FNN_VERSION_CFGSTR
#undef FNN_ARCH_CFGSTR
#undef FNN_LAYER_W_CFGSTR
#undef FNN_LAYER_B_CFGSTR
#undef FNN_AFUNS_CFGSTR
#undef FNN_LAYER_AFUN_PARAM_CFGSTR
#undef FNN_MOMENS_CFGSTR
#undef FNN_LRATES_CFGSTR
  
  
  //////////////////////////////////////////////////////////////////////
  
  
  template <typename T>
  bool neuralnetwork<T>::exportdata(math::vertex<T>& v) const throw()
  {
    if(compressed) return false;
    
    // params for each layer:
    // weights, biases, moment_factor, learning_factor
    // (later: activation fun type, its parameters)
    
    assert(layers.size() > 0);
    
    typename std::vector< neuronlayer<T>* >::const_iterator i =
      layers.begin();
    
    unsigned int j = 0;
    
    // calculates size of feature vector
    while(i != layers.end()){
      // doesn't do decompression because it isn't needed
      
      j += (*i)->weights().ysize() * (*i)->weights().xsize();
      j += (*i)->bias().size();
      j += 2;
      
      i++;
    }
    
    
    // v doesn't need to have to have exactly same size
    if(v.size() < j)
      v.resize(j);
    
    
    i = layers.begin();
    j = 0;
    
    
    while(i != layers.end()){
      if(compressed) (*i)->decompress();
      
      const math::matrix<T>& W = (*i)->weights();
      const math::vertex<T>& b = (*i)->bias();
      
      for(unsigned int y=0;y<W.ysize();y++)
	for(unsigned int x=0;x<W.xsize();x++,j++)
	  v[j] = W(y,x);
      
      for(unsigned int x=0;x<b.size();x++,j++)
	v[j] = b[x];

      // these are not needed, remove and/or replace with activation function params
      v[j] = (*i)->moment();
      j++;
      
      v[j] = (*i)->learning_rate();
      j++;
      
      
      if(compressed) (*i)->compress();
      
      i++;
    }
    
    return true;
  }
  
  
  template <typename T>
  bool neuralnetwork<T>::importdata(const math::vertex<T>& v) throw()
  {
    assert(layers.size() > 0);
    
    typename std::vector< neuronlayer<T>* >::iterator i =
      layers.begin();
    
    
    unsigned int j = 0;
    
    // calculates correct size of feature vector for this NN
    {
      while(i != layers.end()){		
	
	// doesn't do decompression because it isn't needed
	
	j += (*i)->weights().xsize() * (*i)->weights().ysize();
	j += (*i)->bias().size();
	j += 2;
	
	
	i++;
      }

      
      // v doesn't need to have to have exactly same size
      if(v.size() < j)
	return false;
    }
    
    
    i = layers.begin();
    j = 0;
    
    
    while(i != layers.end()){
      if(compressed) (*i)->decompress();
      
      math::matrix<T>& W = (*i)->weights();
      math::vertex<T>& b = (*i)->bias();
      
      for(unsigned int y=0;y<W.ysize();y++)
	for(unsigned int x=0;x<W.xsize();x++,j++)
	  W(y,x) = v[j];
      
      for(unsigned int x=0;x<b.size();x++,j++)
	b[x] = v[j];
      
      // these are not needed, remove and/or replace with activation function params
      (*i)->moment() = v[j]; j++;
      (*i)->learning_rate() = v[j]; j++;
      
      
      if(compressed) (*i)->compress();
      
      i++;
    }
    
    return true;
  }
  
  
  template <typename T>
  unsigned int neuralnetwork<T>::exportdatasize() const throw()
  {
    typename std::vector< neuronlayer<T>* >::const_iterator i =
      layers.begin();
    
    unsigned int j = 0;
    
    // calculates size of feature vector
    
    while(i != layers.end()){
      // doesn't do decompression because it isn't needed
      
      j += (*i)->weights().ysize() * (*i)->weights().xsize();
      j += (*i)->bias().size();
      j += 2;
      
      i++;
    }
    
    return j;
  }
  
  
  ////////////////////////////////////////////////////////////
  
  template <typename T>
  bool neuralnetwork<T>::compress() throw()
  {
    neuralnetwork<T>* nncopy = 0;
    
    try{      
      // tries to make copy of this (in case compression fails)      
      nncopy = new neuralnetwork<T>(*this);
      
      typename std::vector< neuronlayer<T>* >::iterator i =
	layers.begin();
      
      while(i != layers.end()){
	if((*i)->iscompressed() == false)
	  if((*i)->compress() == false){
	    // compression failed,
	    // get good data from the copy
	    
	    i = layers.begin();
	    while(i != layers.end()){ delete *i; i++; }
	      
	    this->layers = nncopy->layers;
	    
	    nncopy->layers.resize(0);
	    delete nncopy;
	    
	    return false; 
	  }
	
	i++;
      }
      
      delete nncopy;
      
      compressed = true;
      
      return true;
    }
    catch(std::exception& e){ 
      if(nncopy != 0) delete nncopy;
      return false;
    }
  }
  
  
  
  template <typename T>
  bool neuralnetwork<T>::decompress() throw()
  {
    neuralnetwork<T>* nncopy = 0;
    
    try{
      // tries to make copy of this (in case compression fails)      
      nncopy = new neuralnetwork<T>(*this);
      
      typename std::vector< neuronlayer<T>* >::iterator i =
	layers.begin();
      
      while(i != layers.end()){
	if((*i)->iscompressed())
	  if((*i)->decompress() == false){
	    // most probably reason: out of memory,
	    // gets old data from the nncopy
	    
	    i = layers.begin();
	    while(i != layers.end()){ delete *i; i++; }
	    
	    this->layers = nncopy->layers;
	    
	    nncopy->layers.resize(0);
	    delete nncopy;
	    
	    return false;
	  }
	
	i++;
      }
      
      delete nncopy;
      
      compressed = false;
      
      return true;
    }
    catch(std::exception& e){
      if(nncopy != 0) delete nncopy;
      return false;
    }
  }
  

  template <typename T>
  bool neuralnetwork<T>::iscompressed() const throw()
  {
    return compressed;
  }
  
  
  template <typename T>
  float neuralnetwork<T>::ratio() const throw()
  {
    if(compressed == false) return 1.0f;
    
    typename std::vector< neuronlayer<T>* >::const_iterator i =
      layers.begin();
    
    float total = 0.0f, sum = 0.0f;
    
    while(i != layers.end()){
      
      sum += (*i)->ratio() * 
	( (*i)->weights().ysize() * (*i)->weights().xsize() );
      
      total += 
	( (*i)->weights().ysize() * (*i)->weights().xsize() );
      
      i++;
    }
    
    return (sum / total);
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class neuralnetwork< float >;
  template class neuralnetwork< double >;  
  template class neuralnetwork< math::blas_real<float> >;
  template class neuralnetwork< math::blas_real<double> >;
  
}


#endif


