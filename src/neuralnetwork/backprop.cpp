
#include "backprop.h"



namespace whiteice
{
  
  template <typename T>
  backprop<T>::backprop()
  {
    latestError = T(0.0);
    input = 0;
    output = 0;
    nn = 0;
    
  }
  
  
  template <typename T>
  backprop<T>::backprop(nnetwork<T>* nn,
			const dataset<T>* data)
  {
    this->nn = nn;
    latestError = T(0.0);
    input = data;
    output = 0;
  }
  
  
  template <typename T>
  backprop<T>::backprop(nnetwork<T>* nn,
			const dataset<T>* in,
			const dataset<T>* out)
  {
    this->nn = nn;
    latestError = T(0.0);
    input = in;
    output = out;
  }
  

  
  template <typename T>
  backprop<T>::~backprop()
  {
    nn = 0;
    input = 0;
    output = 0;
  }
  

  //////////////////////////////////////////////////
  
  template <typename T>
  bool backprop<T>::improve(unsigned int niters)
  {
    T sum_error = T(0.0f);
    whiteice::math::vertex<T> v;
    bool singleDataset = false;
    
    if(input == 0 || nn == 0)
      return false;
    
    if(output == 0)
      singleDataset = true;
    
    if(singleDataset){
      if(input->getNumberOfClusters() < 2)
	return false;
      
      if(input->size(0) == 0)
	return true;
      
      if(input->size(0) != input->size(1))
	return false;
      
      v.resize(input->dimension(1));
    }
    else{
      if(input->getNumberOfClusters() < 1 ||
	 output->getNumberOfClusters() < 1)
	return false;
      
      if(input->size(0) == 0)
	return true;
      
      if(input->size(0) != output->size(0))
	return false;
      
      v.resize(output->dimension(0));
    }
    
    
    if(singleDataset){
      for(unsigned int e=0;e<niters;e++){
	for(unsigned int i=0;i<input->size(0);i++){
	  unsigned int index = rand() % (input->size(0));
	  
	  nn->input() = input->access(0, index);
	  /*
	    std::cout << " IN : " << input->access(0, index) << std::endl;
	    std::cout << " IN : " << nn->input() << std::endl;
	  */
	  
	  nn->calculate(true);
	  
	  // std::cout << "OUT: " << nn->output() << std::endl;
	  
	  v = input->access(1, index);
	  v -= nn->output();
	  
	  // inner product
	  for(unsigned int j=0;j<v.size();j++)
	    sum_error += (v[j]*v[j]) / T(niters*(input->size(0)));
	  
	  nn->backprop(v);
	}
      }
      
      latestError = sum_error;
    }
    else{
      for(unsigned int e=0;e<niters;e++){
	for(unsigned int i=0;i<input->size();i++){
	  unsigned int index = rand() % (input->size(0));
	  
	  nn->input() = (*input)[index];
	  nn->calculate(true);
	  
	  v = (*output)[index] - nn->output();
	  
	  for(unsigned int j=0;j<v.size();j++)
	    sum_error += v[j]*v[j]; // inner product
	  
	  nn->backprop(v);
	}
      }
      
      sum_error /= T(niters*(input->size()));
      latestError = sum_error;
    }
    
    
    return true;
  }
  
  
  template <typename T>
  void backprop<T>::setData(nnetwork<T>* nn,
			    const dataset<T>* data)
  {
    this->nn = nn;
    input = data;
    output = 0;
  }
  
  
  template <typename T>
  void backprop<T>::setData(nnetwork<T>* nn,
			    const dataset<T>* in,
			    const dataset<T>* out)
  {
    this->nn = nn;
    input = in;
    output = out;
  }
  
  
  template <typename T>
  T backprop<T>::getError() {
    return latestError;
    
  }

  template <typename T>
  T backprop<T>::getCurrentError() {
    return latestError;
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class backprop< float >;
  template class backprop< double >;
  template class backprop< math::atlas_real<float> >;
  template class backprop< math::atlas_real<double> >;
  
};
