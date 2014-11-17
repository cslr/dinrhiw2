
#include <stdexcept>
#include <typeinfo>
#include "nnPSO.h"


#ifndef nnPSO_cpp
#define nnPSO_cpp


namespace whiteice
{
  
  
  template <typename T>
  nnPSO<T>::nnPSO(neuralnetwork<T>* nn,
		  const dataset<T>* input,
		  const dataset<T>* output,
		  unsigned int swarmsize)
  {
    pso = 0;
    nn_error = 0;
    
    nn_error = new nnPSO_optimized_function<T>(nn,
					       input,
					       output);
    
    std::vector<typename PSO<T>::range> range;
    range.resize(nn_error->dimension());
    
    for(unsigned int i=0;i<nn_error->dimension();i++){
      range[i].min = -1.0;
      range[i].max = +1.0;
    }
    
    pso = new PSO<T>(*nn_error, range);
    
    this->firsttime = true;
    this->swarmsize = swarmsize;        
  }



  template <typename T>
  nnPSO<T>::nnPSO(neuralnetwork<T>* nn,
		  const dataset<T>* data,
		  unsigned int swarmsize)
  {
    pso = 0;
    nn_error = 0;
    
    nn_error = new nnPSO_optimized_function<T>(nn, data);
    
    std::vector<typename PSO<T>::range> range;
    range.resize(nn_error->dimension());
    
    for(unsigned int i=0;i<nn_error->dimension();i++){
      range[i].min = -1.0;
      range[i].max = +1.0;
    }
    
    pso = new PSO<T>(*nn_error, range);
    
    this->firsttime = true;
    this->swarmsize = swarmsize;        
  }
  
  
  template <typename T>
  nnPSO<T>::nnPSO(optimized_function<T>* nnfun,
		  unsigned int swarmsize)
  {
    nn_error = nnfun;
    
    std::vector<typename PSO<T>::range> range;
    range.resize(nn_error->dimension());
    
    for(unsigned int i=0;i<nn_error->dimension();i++){
      range[i].min = -1.0;
      range[i].max = +1.0;
    }
    
    pso = new PSO<T>(*nn_error, range);
    
    this->firsttime = true;
    this->swarmsize = swarmsize;        
  }
  
  
  template <typename T>
  nnPSO<T>::~nnPSO()
  {
    if(nn_error) delete nn_error;
    if(pso) delete pso;
  }
  
  
  template <typename T>
  bool nnPSO<T>::improve(unsigned int niters)
  {
    if(firsttime){
      if(pso->minimize(niters, swarmsize)){
	firsttime = false;
	return true;
      }
      else
	return false;
    }
    else
      return pso->improve(niters);
  }
  
  
  template <typename T>
  T nnPSO<T>::getCurrentError()
  {
    // calculates error
    math::vertex<T> x;
    pso->getCurrentBest(x);
    
    return nn_error->calculate(x);
  }
  
  
  template <typename T>
  T nnPSO<T>::getError()
  {
    // calculates error
    math::vertex<T> x;
    pso->getBest(x);
    
    return nn_error->calculate(x);
  }
  
  
  template <typename T>
  bool nnPSO<T>::getSolution(math::vertex<T>& v)
  {
    pso->getBest(v);
    
    return true;
  }
  
  
  template <typename T>
  T nnPSO<T>::getCurrentValidationError()
  {
    if(typeid(nn_error) == typeid(nnPSO_optimized_function<T>*)){
      // calculates error
      math::vertex<T> x;
      pso->getCurrentBest(x);
    
      return ((nnPSO_optimized_function<T>*)nn_error)->getValidationError(x);
    }
    else
      return T(-1);
  }
  
  
  template <typename T>
  bool nnPSO<T>::verbosity(bool v) throw()
  {
    this->verbose = v;
    if(pso) pso->verbosity(v);
    return true;
  }
  
  
  template <typename T>
  const math::vertex<T>& nnPSO<T>::sample()
  {
    return (pso->sample()).value;
  }
  
  
  template <typename T>
  bool nnPSO<T>::enableOvertraining() throw()
  {
    if(typeid(nn_error) != typeid(nnPSO_optimized_function<T>*))
      return false;
    
    ((nnPSO_optimized_function<T>*)nn_error)->enableUseAllData();
    return true;
  }
  
  
  template <typename T>
  bool nnPSO<T>::disableOvertraining() throw()
  {
    if(typeid(nn_error) != typeid(nnPSO_optimized_function<T>*))
      return false;
    
    ((nnPSO_optimized_function<T>*)nn_error)->disableUseAllData();
    return true;
  }
  
  
  
  ////////////////////////////////////////////////////////////
  // to be optimized function
  
  
  template <typename T>
  nnPSO_optimized_function<T>::nnPSO_optimized_function(neuralnetwork<T>* nn,
							const dataset<T>* input,
							const dataset<T>* output)
  {
    this->testnet = new neuralnetwork<T>(*nn);
    this->input = input;
    this->output = output;
    
    {
      math::vertex<T> v;
      
      if(this->testnet->exportdata(v) == false)
	throw std::logic_error("bad neural network parameter/copy");
      
      this->fvector_dimension = v.size();
	
    }
    
    if(input->size() != output->size())
      throw std::invalid_argument("nnPSO: |input| != |output|");
  }

  
  template <typename T>
  nnPSO_optimized_function<T>::nnPSO_optimized_function(neuralnetwork<T>* nn,
							const dataset<T>* data)
  {
    this->testnet = new neuralnetwork<T>(*nn);
    this->input = data;
    this->output = 0;
    
    {
      math::vertex<T> v;
      
      if(this->testnet->exportdata(v) == false)
	throw std::logic_error("bad neural network parameter/copy");
      
      this->fvector_dimension = v.size();
	
    }
    
    if(input->size(0) != input->size(1))
      throw std::invalid_argument("nnPSO: |input| != |output|");
  }
  
  
  template <typename T>
  nnPSO_optimized_function<T>::nnPSO_optimized_function(const nnPSO_optimized_function<T>& nnpsof)
  {
    this->testnet = new neuralnetwork<T>(*nnpsof.testnet);
    this->input  = nnpsof.input;
    this->output = nnpsof.output;
    
    this->fvector_dimension = nnpsof.fvector_dimension;
    
    if(output == 0){
      if(input->size(0) != input->size(1))
	throw std::invalid_argument("nnPSO: |input| != |output|");
    }
    else{
      if(input->size() != output->size())
	throw std::invalid_argument("nnPSO: |input| != |output|");
    }
  }
  
  
  template <typename T>
  nnPSO_optimized_function<T>::~nnPSO_optimized_function()
  {
    if(this->testnet)
      delete this->testnet;
  }
  
  
  // calculates value of function
  template <typename T>
  T nnPSO_optimized_function<T>::operator() (const math::vertex<T>& x) const {
    return calculate(x);
  }
  
  
  // calculates value/error
  template <typename T>
  T nnPSO_optimized_function<T>::calculate(const math::vertex<T>& x) const
  {    
    T error = T(0.0);
    
    // uses even samples to calculate error    
    
    typename dataset<T>::const_iterator i = input->begin(0);
    typename dataset<T>::const_iterator j;
    if(output != 0) j = output->begin();
    else j = input->begin(1);
    
    
    math::vertex<T> e;
    T counter = T(0.0);
    
    if(testnet->importdata(x) == false)
      goto bigerror;
    
    while(i != input->end(0)){
      testnet->input() = *i;
      
      if(!testnet->calculate())
	goto bigerror;
      
      e = testnet->output();
      e -= *j;
      
      error += (e * e)[0];
      
      counter += T(1.0);
      
      i++;
      j++;
      
      if(!useAllSamples){
	i++;
	j++;
      }
    }
    
    if(counter > T(0.0))
      error /= counter;
    
    return error;
    
  bigerror: // "exception handlers with goto"
    std::cout << "calculate error" << std::endl;
    
    error = T(10.0);
    for(unsigned int i=0;i<100;i++)
      error *= error;
    
    return error; // very big number
  }
  
  
  template <typename T>
  void nnPSO_optimized_function<T>::calculate(const math::vertex<T>& x, T& y) const {
    y = calculate(x);
  }
  
  
  template <typename T>
  unsigned int nnPSO_optimized_function<T>::dimension() const throw()
  {
    return (this->fvector_dimension);
  }
  
  
  // creates copy of object
  template <typename T>
  function<math::vertex<T>,T>* nnPSO_optimized_function<T>::clone() const
  {
    return new nnPSO_optimized_function<T>(*this);
  }
  
  
  template <typename T>
  bool nnPSO_optimized_function<T>::getUseAllData() const throw()
  { return useAllSamples; }
  
  
  template <typename T>
  void nnPSO_optimized_function<T>::enableUseAllData() throw()
  { useAllSamples = true; }
  
  
  template <typename T>
  void nnPSO_optimized_function<T>::disableUseAllData() throw()
  { useAllSamples = false; }
  
  
  template <typename T>
  T nnPSO_optimized_function<T>::getValidationError(const math::vertex<T>& x) const
  {
    T error = T(0.0);
    
    // uses odd samples to calculate error    
    
    typename dataset<T>::const_iterator i = input->begin(0);
    typename dataset<T>::const_iterator j;
    if(output != 0) j = output->begin();
    else j = input->begin(1);
    
    
    math::vertex<T> e;
    T counter = T(0.0);
    
    if(testnet->importdata(x) == false)
      goto bigerror;
    
    i++;
    j++;
    
    while(i != input->end(0)){
      testnet->input() = *i;
      
      if(!testnet->calculate())
	goto bigerror;
      
      e = testnet->output();
      e -= *j;
      
      error += (e * e)[0];
      
      counter += T(1.0);
      
      i++;
      j++;
      
      if(!useAllSamples){
	i++;
	j++;
      }
    }
    
    if(counter > T(0.0))
      error /= counter;
    
    return error;
    
  bigerror: // "exception handlers with goto"
    std::cout << "calculate error" << std::endl;
    
    error = T(10.0);
    for(unsigned int i=0;i<100;i++)
      error *= error;
    
    return error; // very big number
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  
  template <typename T>
  bool nnPSO_optimized_function<T>::hasGradient() const throw(){
    return false;
  }
  
  // gets gradient at given point (faster)
  template <typename T>
  math::vertex<T> nnPSO_optimized_function<T>::grad(math::vertex<T>& x) const{
    return x;
  }
  
  
  template <typename T>
  void nnPSO_optimized_function<T>::grad(math::vertex<T>& x, math::vertex<T>& y) const{
    return;
  }
  
  
  template <typename T>
  bool nnPSO_optimized_function<T>::hasHessian() const throw(){
    return false;
  }
  
  
  // gets gradient at given point (faster)
  template <typename T>
  math::matrix<T> nnPSO_optimized_function<T>::hessian(math::vertex<T>& x) const{
    return math::matrix<T>(1,1);
    
  }
  
  
  template <typename T>
  void nnPSO_optimized_function<T>::hessian(math::vertex<T>& x, math::matrix<T>& y) const{
    return;
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class nnPSO<float>;
  template class nnPSO<double>;
  template class nnPSO< math::atlas_real<float> >;
  template class nnPSO< math::atlas_real<double> >;
  template class nnPSO_optimized_function<float>;
  template class nnPSO_optimized_function<double>;
  template class nnPSO_optimized_function< math::atlas_real<float> >;
  template class nnPSO_optimized_function< math::atlas_real<double> >;
  
  
};


#endif
