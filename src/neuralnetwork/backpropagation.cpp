
#ifndef backpropagation_cpp
#define backpropagation_cpp

#include "nn_iterative_correction.h"
#include "backpropagation.h"
#include "vertex.h"
#include "matrix.h"
#include "dataset.h"

#include <vector>
#include <typeinfo>


namespace whiteice
{
  
  
  template <typename T>
  backpropagation<T>::backpropagation(){
    latestError = T(0.0);
    nnetwork = 0;
    input  = 0;
    output = 0;
  }
  
  
  template <typename T>
  backpropagation<T>::backpropagation(neuralnetwork<T>* nn,
				      const dataset<T>* input,
				      const dataset<T>* output)
  {
    setData(nn, input, output);
  }
  
  
  template <typename T>
  backpropagation<T>::backpropagation(neuralnetwork<T>* nn,
				      const dataset<T>* data)
  {
    setData(nn, data);
  }
  

  template <typename T>
  backpropagation<T>::~backpropagation(){ }


  /*
   * backprop error correction
   */
  template <typename T>
  bool backpropagation<T>::calculate(neuralnetwork<T>& nn,
				     const math::vertex<T>& correct_output) const
  {
    
    typename std::vector<neuronlayer<T>*>::iterator i;
    i = nn.layers.end();
    i--;
    
    math::vertex<T> lgrad, temp; // local gradients
    lgrad.resize(nn.state.size()); // make it big enough for all layers
    temp.resize(nn.state.size());
    
    // calculates initial gradients
    
    for(unsigned int j=0;j<nn.output_values.size();j++){
      // lgrad = (d - o) * activation_derivate
      lgrad[j] = (correct_output[j] - nn.output_values[j]);
      lgrad[j] *= (*i)->F->derivate( (*i)->state[j] );
    }
    
    // iterates from backward to forward
    
    if(typeid(T) == typeid(math::blas_real<float>)){
    
      while(i != nn.layers.begin()){
	temp = lgrad; // creates copy of current gradient
	
	// calculates the next gradient (1)
	// 
	// Wt = (*i)->W;
	// Wt.transpose();
	// 
	
	temp.resize((*i)->W.ysize());
	lgrad.resize((*i)->W.xsize());
	
	
	// lgrad = Wt * temp;
	// 
	cblas_sgemv(CblasRowMajor, CblasTrans, (*i)->W.ysize(), (*i)->W.xsize(),
		    1.0f, ((float*)((*i)->W.data)), (*i)->W.xsize(),
		    (float*)(temp.data), 1, 
		    0.0f, (float*)(lgrad.data), 1);
		    

	
	// updates weights & biases
	// 
	// (*i)->W += ((*i)->learning_factor) * temp.outerproduct( (*i)->saved_input );
	// (*i)->b += ((*i)->learning_factor) * temp;
	// 
	cblas_sger(CblasRowMajor, (*i)->W.ysize(), (*i)->W.xsize(),
		   *((float*)(&((*i)->learning_factor))), (float*)temp.data, 1,
		   (float*)((*i)->saved_input.data), 1,
		   (float*)((*i)->W.data), (*i)->W.xsize());
	
	cblas_saxpy((*i)->b.size(), *((float*)(&((*i)->learning_factor))),
		    (float*)(temp.data), 1,
		    (float*)((*i)->b.data), 1);
	
	i--;
	
	// calculates the next gradient (2)
	
	for(unsigned int j=0;j<lgrad.size();j++)
	  lgrad[j] *= (*i)->F->derivate((*i)->state[j]);
	
      }
    
      // calculates first layer
      // (no need to calculate next gradient)
    
      cblas_sger(CblasRowMajor, (*i)->W.ysize(), (*i)->W.xsize(),
		 *((float*)(&((*i)->learning_factor))), (float*)(lgrad.data), 1,
		 (float*)((*i)->saved_input.data), 1,
		 (float*)((*i)->W.data), (*i)->W.xsize());
      
      cblas_saxpy((*i)->b.size(), *((float*)(&((*i)->learning_factor))),
		  (float*)(temp.data), 1,
		  (float*)((*i)->b.data), 1);
      
    }
    else if(typeid(T) == typeid(math::blas_complex<float>)){
      
      math::blas_complex<float> one(1.0f);
      math::blas_complex<float> zero(0.0f);
      math::blas_complex<float> lr(1.0f);
      
      while(i != nn.layers.begin()){
	temp = lgrad; // creates copy of current gradient
	
	temp.resize((*i)->W.ysize());
	lgrad.resize((*i)->W.xsize());
	
	// calculates the next gradient (1)
	
	cblas_cgemv(CblasRowMajor, CblasTrans, (*i)->W.ysize(), (*i)->W.xsize(),
		    (float*)&one, ((float*)((*i)->W.data)), (*i)->W.xsize(),
		    (float*)(temp.data), 1, (float*)&zero,
		    (float*)(lgrad.data), 1);
	
	// updates weights
	
	lr = (*i)->learning_factor;
	
	cblas_cgeru(CblasRowMajor, (*i)->W.ysize(), (*i)->W.xsize(),
		    (float*)&lr, (float*)(temp.data), 1, (float*)((*i)->saved_input.data), 1,
		    (float*)((*i)->W.data), (*i)->W.xsize());
	
	cblas_caxpy((*i)->b.size(), (float*)&lr,
		    (float*)(temp.data), 1,
		    (float*)((*i)->b.data), 1);
	
	i--;
	
	// calculates the next gradient (2)
	
	for(unsigned int j=0;j<lgrad.size();j++)
	  lgrad[j] *= (*i)->F->derivate((*i)->state[j]);
	
      }
    
      // calculates first layer
      // (no need to calculate next gradient)
      
      lr = (*i)->learning_factor;
	    
      cblas_cgeru(CblasRowMajor, (*i)->W.ysize(), (*i)->W.xsize(),
		  (float*)&lr, (float*)(lgrad.data), 1, (float*)((*i)->saved_input.data), 1, 
		  (float*)((*i)->W.data), (*i)->W.xsize());
      
      cblas_caxpy((*i)->b.size(), (float*)&lr,
		  (float*)(temp.data), 1,
		  (float*)((*i)->b.data), 1);
      
      
    }
    else if(typeid(T) == typeid(math::blas_real<double>)){
      
      while(i != nn.layers.begin()){
	temp = lgrad; // creates copy of current gradient
	
	temp.resize((*i)->W.ysize());
	lgrad.resize((*i)->W.xsize());
	
	// calculates the next gradient (1)
	
	cblas_dgemv(CblasRowMajor, CblasTrans, (*i)->W.ysize(), (*i)->W.xsize(),
		    1.0, ((double*)((*i)->W.data)), (*i)->W.xsize(),
		    (double*)(temp.data), 1, 0.0,
		    (double*)(lgrad.data), 1);
	
	// updates weights
	
	cblas_dger(CblasRowMajor, (*i)->W.ysize(), (*i)->W.xsize(),
		   *((double*)(&((*i)->learning_factor))), (double*)(temp.data), 1,
		   (double*)((*i)->saved_input.data), 1, 
		   (double*)((*i)->W.data), (*i)->W.xsize());
	
	cblas_daxpy((*i)->b.size(), *((double*)(&((*i)->learning_factor))),
		    (double*)(temp.data), 1,
		    (double*)((*i)->b.data), 1);
	
	i--;
	
	// calculates the next gradient (2)
	
	for(unsigned int j=0;j<lgrad.size();j++)
	  lgrad[j] *= (*i)->F->derivate((*i)->state[j]);
	
      }
    
      // calculates first layer
      // (no need to calculate next gradient)
      
      cblas_dger(CblasRowMajor, (*i)->W.ysize(), (*i)->W.xsize(),
		 *((double*)&((*i)->learning_factor)), (double*)(lgrad.data), 1,
		 (double*)((*i)->saved_input.data), 1,
		 (double*)((*i)->W.data), (*i)->W.xsize());
      
      cblas_daxpy((*i)->b.size(), *((double*)(&((*i)->learning_factor))),
		  (double*)(temp.data), 1,
		  (double*)((*i)->b.data), 1);
      
    }
    else if(typeid(T) == typeid(math::blas_complex<double>)){
      
      math::blas_complex<double> one(1.0f);
      math::blas_complex<double> zero(0.0f);
      math::blas_complex<double> lr(1.0f);
      
      while(i != nn.layers.begin()){
	temp = lgrad; // creates copy of current gradient
	
	temp.resize((*i)->W.ysize());
	lgrad.resize((*i)->W.xsize());
	
	// calculates the next gradient (1)
	
	cblas_zgemv(CblasRowMajor, CblasTrans, (*i)->W.ysize(), (*i)->W.xsize(),
		    (double*)&one, ((double*)((*i)->W.data)), (*i)->W.xsize(),
		    (double*)(temp.data), 1, (double*)&zero,
		    (double*)(lgrad.data), 1);
	
	// updates weights & biases
	
	lr = (*i)->learning_factor;
	
	cblas_zgeru(CblasRowMajor, (*i)->W.ysize(), (*i)->W.xsize(),
		    (double*)&lr, (double*)(temp.data), 1,
		    (double*)((*i)->saved_input.data), 1, 
		    (double*)((*i)->W.data), (*i)->W.xsize());
	
	cblas_zaxpy((*i)->b.size(), (double*)&lr,
		    (double*)(temp.data), 1,
		    (double*)((*i)->b.data), 1);
	
	i--;
	
	
	// calculates the next gradient (2)
	
	for(unsigned int j=0;j<lgrad.size();j++)
	  lgrad[j] *= (*i)->F->derivate((*i)->state[j]);
	
      }
    
      // calculates first layer
      // (no need to calculate next gradient)
      
      lr = (*i)->learning_factor;
	    
      cblas_zgeru(CblasRowMajor, (*i)->W.ysize(), (*i)->W.xsize(),
		  (double*)&lr, (double*)lgrad.data, 1, 
		  (double*)((*i)->saved_input.data), 1,
		  (double*)((*i)->W.data), (*i)->W.xsize());
      
      cblas_zaxpy((*i)->b.size(), (double*)&lr,
		  (double*)(temp.data), 1,
		  (double*)((*i)->b.data), 1);
      
    }
    else{ // generic implementation
      
      math::matrix<T> Wt;
      
      unsigned int counter = 0;
      
      while(i != nn.layers.begin()){
	// std::cout << "lgrad " << counter << " = "<< lgrad << std::endl;
	// std::cout << "W = "<< (*i)->W << std::endl;
	temp = lgrad; // creates copy of current gradient
	
	// updates weights & biases
	(*i)->W += ((*i)->learning_factor) * temp.outerproduct( (*i)->saved_input );
	(*i)->b += ((*i)->learning_factor) * temp;
	
	
	// calculates the next gradient (1)
	
	Wt = (*i)->W;
	Wt.transpose();
	
	temp.resize(Wt.xsize());
	lgrad.resize(Wt.ysize());
	
	lgrad = Wt * temp;
	
	i--;
	
	// calculates the next gradient (2)
	
	for(unsigned int j=0;j<lgrad.size();j++)
	  lgrad[j] *= (*i)->F->derivate((*i)->state[j]);
	
	counter++;
      }
      
      // std::cout << "lgrad " << counter << " = "<< lgrad << std::endl;
      // std::cout << "W = "<< (*i)->W << std::endl;

    
      // calculates first layer
      // (no need to calculate next gradient)
      
      (*i)->W += ((*i)->learning_factor) * lgrad.outerproduct( (*i)->saved_input );
      (*i)->b += ((*i)->learning_factor) * lgrad;
      
      counter++;
    }
    
    
    return true;

    
#if 0
    typename std::vector<T> gradients[2];
    unsigned int l;
    
    gradients[0].resize(nn.state.size());
    gradients[1].resize(nn.state.size());
    
    typename std::vector<T>::iterator i = gradients[1].begin();
    typename std::vector<T>::const_iterator j = correct_output.begin();
    typename std::vector<T>::iterator k = nn.output().begin();
    
    /* 
     * calculates initial gradients
     */
    
    const unsigned int L = nn.length();
    
    for(unsigned int m = 0;j!=correct_output.end();i++,j++,k++,m++){
      
      *i = (*j - *k) * nn[L-1][m].activation_derivate_field();
    }
    
    /*
     * backpropagates weight changes/error
     */		
    
    /* calculates next gradient layer, calculation uses soon to be updated weights */
    for(l=L-1;l>0;l--){
      
      unsigned int index;
      
      for(i = gradients[0].begin(), index = 0;index<nn[l].input_size();i++,index++){
	T sum = 0;
	
	unsigned int m;
	
	for(j=gradients[1].begin(), m=0;m<nn[l].size();j++,m++){
	  sum += (*j) * nn[l][m][index];  /* local_grad * weight__index_to_m */
	}
	
	*i = nn[l-1][index].activation_derivate_field() * sum;
      }
      
      /* updates weights in layer l */
      // using moments
      
      i = gradients[1].begin();
      
      for(unsigned int n = 0; n < nn[l].size(); n++, i++){
	
	
	for(unsigned int m = 0; m < nn[l-1][n].input_size(); m++){ // normal weights
	  
	  T delta = nn[l].learning_rate() * (*i) * nn[l-1][m].activation_field()
	    + nn[l].moment() * nn[l][n].delta(m);
	  
	  nn[l][n][m] += delta;
	  nn[l][n].delta(m) = delta;
	}
	
	// biasses
	
	nn[l][n].bias() += nn[l].learning_rate() * (*i); // input is one
      }				
      
      gradients[1] = gradients[0];
    }
    
    // first layer
    
    i = gradients[1].begin();
    
    for(unsigned int n = 0; n < nn[l].size(); n++, i++){
      
      
      for(unsigned int m = 0; m < nn.input().size(); m++){ // normal weights
	nn[l][n][m] += nn[l].learning_rate() * (*i) * nn.input()[m];
      }
      
      // biasses
      
      nn[l][n].bias() += nn[l].learning_rate() * (*i); // input is one
    }
    
    return true;
#endif
    
  }
  
  
  template <typename T>
  bool backpropagation<T>::operator() (neuralnetwork<T>& nn,
				       const math::vertex<T>& correct_output) const
  {
    return calculate(nn, correct_output);
  }
  
  
  template <typename T> // creates copy of object
  nn_iterative_correction<T>* backpropagation<T>::clone() const
  {
    return static_cast< nn_iterative_correction<T>* >(new backpropagation<T>(*this));
  }
  
  
  // forced learning
  // this goes through the material once and 'forces'
  // neural network to improve for each point till
  // it has learnt the one point well
  // enough. This is local minimization but it seems
  // usually give reasonable good starting configuration
  // for more global standard backpropagation.
  // (somewhat related to simulated annealing:
  //  mix forced_improvements() with normal backpropagation)
  template <typename T>
  bool backpropagation<T>::forced_improve()
  {
    T sum_error = T(0.0f);
    whiteice::math::vertex<T> v;
    bool singleDataset = false;
    
    if(output == 0)
      singleDataset = true;
    
    if(nnetwork == 0 || input == 0)
      return false;
    
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
      if(input->size(0) == 0)
	return true;
      
      if(input->size(0) != output->size(0))
	return false;    
    
      v.resize((*output)[0].size());
    }
    
    
    if(singleDataset){
      const unsigned int N = input->size(0);
      
      for(unsigned int i=0;i<N;i++){
	unsigned int counter = 0;
	do{
	  sum_error = T(0.0f);	
	  
	  nnetwork->input() = input->access(0, i);
	  nnetwork->calculate();
	  
	  v = nnetwork->output();
	  v -= input->access(1, i);
	  
	  for(unsigned int j=0;j<v.size();j++)
	    sum_error += v[j]*v[j]; // inner product
	  
	  if(!calculate(*nnetwork, input->access(1,i)))
	    return false;
	  
	  counter++;
	}
	while(sum_error > 0.0001f && counter < 1000);
      }
      
    }
    else{
      const unsigned int N = input->size(0);
      
      for(unsigned int i=0;i<N;i++){
	unsigned int counter = 0;
	do{
	  sum_error = T(0.0f);	
	  
	  nnetwork->input() = (*input)[i];
	  nnetwork->calculate();
	  
	  v = nnetwork->output();
	  v -= (*output)[i];
	  
	  for(unsigned int j=0;j<v.size();j++)
	    sum_error += v[j]*v[j]; // inner product
	  
	  if(!calculate(*nnetwork, (*output)[i]))
	    return false;
	  
	  counter++;
	}
	while(sum_error > 0.0001f && counter < 1000);
      }
      
    }
    
    return true;
  }
  
  
  template <typename T>
  bool backpropagation<T>::improve(unsigned int niters)
  {
    T sum_error = T(0.0f);
    whiteice::math::vertex<T> v;
    bool singleDataset = false;
    
    if(input == 0 || nnetwork == 0)
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
	  
	  nnetwork->input() = input->access(0, index);
	  nnetwork->calculate();
	  
	  v = nnetwork->output();
	  v -= input->access(1, index);
	  
	  for(unsigned int j=0;j<v.size();j++)
	    sum_error += v[j]*v[j]; // inner product
	  
	  if(!calculate(*nnetwork, input->access(1, index)))
	    return false;
	}
      }
      
      sum_error /= T(niters*(input->size(0)));
      latestError = sum_error;
    }
    else{
      for(unsigned int e=0;e<niters;e++){
	for(unsigned int i=0;i<input->size(0);i++){
	  unsigned int index = rand() % (input->size(0));
	  
	  nnetwork->input() = (*input)[index];
	  nnetwork->calculate();
	  
	  v = nnetwork->output();
	  v -= output->access(0, index);
	  
	  for(unsigned int j=0;j<v.size();j++)
	    sum_error += v[j]*v[j]; // inner product
	  
	  if(!calculate(*nnetwork, output->access(0, index)))
	    return false;
	}
      }
      
      sum_error /= T(niters*(input->size(0)));
      latestError = sum_error;
    }
    
    return true;
  }
  
  
  // sets neural network for improve() and
  // getCurrentError() and getError() calls
  template <typename T>
  void backpropagation<T>::setData(neuralnetwork<T>* nn,
				   const dataset<T>* input,
				   const dataset<T>* output)
  {
    latestError  = T(0.0f);
    nnetwork     = nn;
    this->input  = input;
    this->output = output;
  }
  
  
  template <typename T>
  void backpropagation<T>::setData(neuralnetwork<T>* nn, 
				   const dataset<T>* data)
  {
    latestError  = T(0.0f);
    nnetwork     = nn;
    this->input  = data;
    this->output = 0;
  }
	     
  
  
  template <typename T>
  T backpropagation<T>::getError(){
    return latestError;
  }
  
  template <typename T>
  T backpropagation<T>::getCurrentError(){
    return latestError;
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class backpropagation< float >;
  template class backpropagation< double >;
  template class backpropagation< math::blas_real<float> >;
  template class backpropagation< math::blas_real<double> >;
  
  
}


#endif


