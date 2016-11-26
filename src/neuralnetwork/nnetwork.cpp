// TODO:
//   - refactor gradient etc code 
// 

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
  nnetwork<T>::nnetwork()
  {
    stochasticActivation = false;
    nonlinearity = sigmoidNonLinearity;
    
    arch.resize(2);    
    
    arch[0] = 1;
    arch[1] = 1;

    maxwidth = 0;
    size = 0;
    for(unsigned int i=0;i<arch.size();i++){
      if(i > 0)
	size += (arch[i-1] + 1)*arch[i];
      
      if(arch[i] > maxwidth)
	maxwidth = arch[i];
    }

    data.resize(size);

    randomize();
    
    state.resize(maxwidth);
    temp.resize(maxwidth);
    lgrad.resize(maxwidth);
    
    inputValues.resize(1);
    outputValues.resize(1);
    
  }
  
  
  template <typename T>
  nnetwork<T>::nnetwork(const nnetwork<T>& nn)
  {
    inputValues.resize(nn.inputValues.size());
    outputValues.resize(nn.outputValues.size());

    hasValidBPData = nn.hasValidBPData;
    maxwidth = nn.maxwidth;
    size = nn.size;

    stochasticActivation = nn.stochasticActivation;

    arch   = nn.arch;
    bpdata = nn.bpdata;
    data   = nn.data;
    state  = nn.state;
    temp   = nn.temp;
    lgrad  = nn.lgrad;
    nonlinearity = nn.nonlinearity;
  }
  
  
  template <typename T>
  nnetwork<T>::nnetwork(const std::vector<unsigned int>& nnarch,
			const typename nnetwork<T>::nonLinearity nl)
    throw(std::invalid_argument)
  {
    if(nnarch.size() < 2)
      throw std::invalid_argument("invalid network architecture");

    maxwidth = 0;
    stochasticActivation = false;
    nonlinearity = nl; // sigmoidNonLinearity; is the default
    
    for(unsigned int i=0;i<nnarch.size();i++){
      if(nnarch[i] <= 0)
	throw std::invalid_argument("invalid network architecture");
      if(nnarch[i] > maxwidth)
	maxwidth = nnarch[i];
    }

    // sets up architecture
    arch = nnarch;
    
    unsigned int memuse = 0;
    
    for(unsigned int i=0;i<arch.size();i++){
      if(i > 0) 
	memuse += (arch[i-1] + 1)*arch[i];
    }
    
    size = memuse;
    
    data.resize(size);
    
    // intializes all layers (randomly)
    randomize();

    state.resize(maxwidth);
    temp.resize(maxwidth);
    lgrad.resize(maxwidth);
     
    inputValues.resize(arch[0]);
    outputValues.resize(arch[arch.size()-1]);
   
    hasValidBPData = false;
  }
  
  
  template <typename T>
  nnetwork<T>::~nnetwork()
  {
    
  }

  
  template <typename T>
  nnetwork<T>& nnetwork<T>::operator=(const nnetwork<T>& nn)
  {
    hasValidBPData = nn.hasValidBPData;
    arch = nn.arch;
    maxwidth = nn.maxwidth;
    size = nn.size;

    stochasticActivation = nn.stochasticActivation;
    nonlinearity = nn.nonlinearity;

    data = nn.data;
    bpdata = nn.bpdata;

    state = nn.state;
    temp = nn.temp;
    lgrad = nn.lgrad;
    
    inputValues.resize(nn.inputValues.size());
    outputValues.resize(nn.outputValues.size());
    
    return (*this);
  }

  
  ////////////////////////////////////////////////////////////

  // returns input and output dimensions of neural network
  template <typename T>
  unsigned int nnetwork<T>::input_size() const throw(){
    if(arch.size() > 0) return arch[0];
    else return 0;
  }
  
  template <typename T>
  unsigned int nnetwork<T>::output_size() const throw(){
    unsigned int index = arch.size()-1;
    
    if(arch.size() > 0) return arch[index];
    else return 0;
  }

  template <typename T>
  unsigned int nnetwork<T>::gradient_size() const throw()
  {
    return size; // number of parameters in neural network
  }


  template <typename T>
  void nnetwork<T>::getArchitecture(std::vector<unsigned int>& nn_arch) const
  {
    nn_arch = this->arch;
  }


  /*
   * calculates output for input
   * if gradInfo = true also saves needed
   * information for calculating gradient (bpdata)
   */
  template <typename T>
  bool nnetwork<T>::calculate(bool gradInfo, bool collectSamples)
  {
	    if(!inputValues.exportData(&(state[0])))
	      return false;

	    if(collectSamples)
	      samples.resize(arch.size()-1);


	    // unsigned int* a = &(arch[0]);
	    unsigned int aindex = 0;

	    if(gradInfo){ // saves bpdata

	      if(bpdata.size() <= 0){
		unsigned int bpsize = 0;

		for(unsigned int i=0;i<arch.size();i++)
		  bpsize += arch[i];

		bpdata.resize(bpsize);
	      }

	      T* bpptr = &(bpdata[0]);
	      T* dptr = &(data[0]);


	      while(aindex+1 < arch.size()){
		// copies layer input x to bpdata
		memcpy(bpptr, &(state[0]), arch[aindex]*sizeof(T));
		bpptr += arch[aindex];

		if(collectSamples){
		  math::vertex<T> x;
		  x.resize(arch[aindex]);
		  memcpy(&(x[0]), &(state[0]), arch[aindex]*sizeof(T));
		  samples[aindex].push_back(x);
		}

		// gemv(a[1], a[0], dptr, state, state); // s = W*s
		// gvadd(a[1], state, dptr + a[0]*a[1]); // s += b;

		// s = b + W*s
		gemv_gvadd(arch[aindex+1], arch[aindex], dptr, &(state[0]), &(state[0]),
			   arch[aindex+1], &(state[0]), dptr + arch[aindex]*arch[aindex+1]);

	      	// s = g(v)

		if(aindex+2 < arch.size()){ // not the last layer
		  // f(x) = b * ( (1 - Exp[-ax]) / (1 + Exp[-ax]) )
		  //      = b * ( 2 / (1 + Exp[-ax]) - 1)

		  for(unsigned int i=0;i<arch[aindex+1];i++){
		    state[i] = nonlin(state[i], aindex + 1, i);
		  }
		}


		dptr += (arch[aindex] + 1)*arch[aindex+1];

		aindex++; // next layer
	      }

	    }
	    else{
	      T* dptr = &(data[0]);

	      while(aindex+1 < arch.size()){

		if(collectSamples){
		  math::vertex<T> x;
		  x.resize(arch[aindex]);
		  memcpy(&(x[0]), &(state[0]), arch[aindex]*sizeof(T));
		  samples[aindex].push_back(x);
		}

		// gemv(a[1], a[0], dptr, state, state); // s = W*s
		// gvadd(a[1], state, dptr + a[0]*a[1]); // s += b;

		// s = b + W*s
		gemv_gvadd(arch[aindex+1], arch[aindex], dptr, &(state[0]), &(state[0]),
			   arch[aindex+1], &(state[0]), dptr + arch[aindex]*arch[aindex+1]);

		// s = g(v)

		if(aindex+2 < arch.size()){ // not the last layer
		  // f(x)  = a * (1 - Exp[-bx]) / (1 + Exp[-bx])
		  // f'(x) = (0.5*a*b) * ( 1 + f(x)/a ) * ( 1 - f(x)/a )

		  for(unsigned int i=0;i<arch[aindex+1];i++){
		    state[i] = nonlin(state[i], aindex + 1, i);
		  }
		}

		dptr += (arch[aindex] + 1)*arch[aindex+1];
		aindex++; // next layer
	      }

	    }


	    if(!outputValues.importData(&(state[0]))){
	      std::cout << "Failed to import data to vertex from memory." << std::endl;
	      return false;
	    }


	    hasValidBPData = gradInfo;

	    return true;
  }
  
  
  // simple thread-safe version [parallelizable version of calculate: don't calculate gradient nor collect samples]
  template <typename T>
  bool nnetwork<T>::calculate(const math::vertex<T>& input, math::vertex<T>& output) const
  {
	  std::vector<T> state;
	  state.resize(maxwidth);

	  if(input.size() != arch[0])
		  return false; // input vector has wrong dimension

	  if(!input.exportData(&(state[0])))
		  return false;

	  // unsigned int* a = &(arch[0]);
	  unsigned int aindex = 0;

	  {
		  const T* dptr = &(data[0]);

		  while(aindex+1 < arch.size()){
			  // gemv(a[1], a[0], dptr, state, state); // s = W*s
			  // gvadd(a[1], state, dptr + a[0]*a[1]); // s += b;

			  // s = b + W*s
			  gemv_gvadd(arch[aindex+1], arch[aindex], dptr, &(state[0]), &(state[0]),
				     arch[aindex+1], &(state[0]), dptr + arch[aindex]*arch[aindex+1]);

			  // s = g(v)

			  if(aindex+2 < arch.size()){ // not the last layer
				  // f(x)  = a * (1 - Exp[-bx]) / (1 + Exp[-bx])
				  // f'(x) = (0.5*a*b) * ( 1 + f(x)/a ) * ( 1 - f(x)/a )

				  for(unsigned int i=0;i<arch[aindex+1];i++){
				    state[i] = nonlin(state[i], aindex + 1, i);
				  }
			  }

			  dptr += (arch[aindex] + 1)*arch[aindex+1];
			  aindex++; // next layer
		  }
	  }

	  output.resize(arch[arch.size()-1]); // resizes output to have correct size

	  if(!output.importData(&(state[0]))){
		  std::cout << "Failed to import data to vertex from memory." << std::endl;
	      return false;
	  }

	  // hasValidBPData = false;

	  return true;
  }


  template <typename T> // number of layers
  unsigned int nnetwork<T>::length() const {
    return arch.size();
  }
  
  
  template <typename T>
  bool nnetwork<T>::randomize()
  {
    // memset(data.data(), 0, sizeof(T)*data.size()); // debugging..
    
    {
      unsigned int start = 0;
      unsigned int end   = 0;
      
      for(unsigned int i=1;i<arch.size();i++){
	end   += (arch[i-1] + 1)*arch[i];
	
	// this initialization is as described in the paper of Xavier Glorot
	// Understanding the difficulty of training deep neural networks
	
	T var = math::sqrt(6.0f / (arch[i-1] + arch[i]));
	T scaling = T(2.2);

	  // T(3.0f*((float)rand())/((float)RAND_MAX)); // different scaling for different nonlins

	// std::cout << "random init 2 " << scaling << std::endl;
      
	var *= scaling;
	
	for(unsigned int j=start;j<end;j++){
	  T r = T( 2.0f*(((float)rand())/((float)RAND_MAX)) - 1.0f ); // [-1,1]
	  // data[j] = T(3.0f)*var*r; // asinh(x) requires aprox 3x bigger values before reaching saturation than tanh(x)
	  data[j] = var*r; // asinh(x) requires aprox 3x bigger values before reaching saturation than tanh(x)
	}

#if 0
	// sets bias terms to zero?
	for(unsigned int l=0;l<getLayers();l++){
	  whiteice::math::vertex<T> bias;
	  if(getBias(bias, l)){
	    bias.zero();
	    setBias(bias,l);
	  }
	}
	
	whiteice::math::matrix<T> W;
	for(unsigned int l=0;l<getLayers();l++){
	  if(getWeights(W, l)){
	    W.zero();
	    setWeights(W,l);
	  }
	}
#endif
	
	start += (arch[i-1] + 1)*arch[i];
      }

      assert(start == data.size()); // we have processed the whole memory area correctly
    }

    
    
    return true;
  }
  
  
  
  // calculates gradient of [ 1/2*(last_output - network(last_input|w))^2 ] => error*GRAD[function(x,w)]
  // uses values stored by previous computation, this function is very heavily optimized using direct memory
  // accesses to the neural network matrix and vector parameter data (memory block)
  // calculates gradient grad(error) = grad(right - output)
  template <typename T>
  bool nnetwork<T>::gradient(const math::vertex<T>& error,
			     math::vertex<T>& grad) const
  {
    if(!hasValidBPData)
      return false;
    
    T* lgrad = (T*)this->lgrad.data();
    T* temp  = (T*)this->temp.data();
  
    // initial (last layer gradient)
    // error[i] * NONLIN'(v) (last layer NONLIN = id(x), id(x)' = 1)
    // => last layer local gradient is error[i]
    // also for logistic/sigmoidal g(v), g'(v) = y*(1-y) this means
    // local field v doesn't have to be saved.
    //
    // in practice I use modified sigmoidal (scale + e(-bx) term)
    // so this is not that simple but outputs can be still used
  
    
    unsigned int counter = arch.size() - 1;
    const unsigned int *a = &(arch[arch.size()-1]);
    
    if(!error.exportData(lgrad))
      return false;
    
    const T* _bpdata = &(bpdata[0]);
    
    // goes to end of bpdata input lists
    for(unsigned int i=0;i<(arch.size() - 1);i++)
      _bpdata += arch[i];
  
    const T* _data = &(data[0]);
    
    // goes to end of data area
    for(unsigned int i=1;i<arch.size();i++){
      _data += ((arch[i-1])*(arch[i])) + arch[i];
    }
    
    grad.resize(size);
    unsigned int gindex = grad.size(); // was "unsigned gindex" ..
    
    
    while(counter > 1){
      // updates W and b in this layer
      
      // delta W = (lgrad * input^T)
      // delta b =  lgrad;
      
      const unsigned int rows = *a;
      const unsigned int cols = *(a - 1);
      
      _bpdata -= *(a - 1);
      _data -= (*a) * (*(a - 1)) + *a;
      gindex -= (*a) * (*(a - 1)) + *a;
      const T* dptr = _data;
      
      {
	for(unsigned int y=0;y<rows;y++){
	  const T* bptr = _bpdata;
	  for(unsigned int x=0;x<cols;x++,gindex++){
	    // dptr[x + y*cols] += rate * lgrad[y] * (*bptr);
	    grad[gindex] = -lgrad[y] * (*bptr);
	    bptr++;
	  }
	}
	
	dptr += rows*cols;
	
	
	for(unsigned int y=0;y<rows;y++,gindex++){
	  grad[gindex] = -lgrad[y];
	  // dptr[y] += rate * lgrad[y];
	}
	
	gindex -= rows*cols + rows;
      }
      

      
      // calculates next lgrad
      
      // for hidden layers: local gradient is:
      // grad[n] = diag(..g'(v[i])..)*(W^t * grad[n+1])

      // FIXME: THERE IS A BUG IN BPTR handling???
      const T* bptr = _bpdata;
    
      if(typeid(T) == typeid(whiteice::math::blas_real<float>)){
	
	cblas_sgemv(CblasRowMajor, CblasTrans,
		    rows, cols,
		    1.0f, (float*)_data, cols, (float*)lgrad, 1,
		    0.0f, (float*)temp, 1);
	
	for(unsigned int x=0;x<cols;x++){
	  temp[x] *= Dnonlin(*bptr, counter - 1, x);
	  bptr++;
	}
	
      }
      else if(typeid(T) == typeid(whiteice::math::blas_real<double>)){
	
	cblas_dgemv(CblasRowMajor, CblasTrans,
		    rows, cols,
		    1.0f, (double*)_data, cols, (double*)lgrad, 1,
		    0.0f, (double*)temp, 1);
	
	for(unsigned int x=0;x<cols;x++){
	  temp[x] *= Dnonlin(*bptr, counter - 1, x);
	  bptr++;
	}
	
      }
      else{
	for(unsigned int x=0;x<cols;x++){
	  T sum = T(0.0f);
	  for(unsigned int y=0;y<rows;y++)
	    sum += lgrad[y]*_data[x + y*cols];
	  
	  sum *= Dnonlin(*bptr, counter - 1, x);
	  
	  temp[x] = sum;
	  bptr++;
	}
      }
      
      // swaps memory pointers
      {
	T* ptr = temp;
	temp = lgrad;
	lgrad = ptr;
      }
      
      counter--;
      a--;
    }
    
    
    {
      // calculates the first layer's delta W and delta b 
      // 
      // delta W += (lgrad * input^T)
      // delta b += lgrad;
      
      const unsigned int rows = *a;
      const unsigned int cols = *(a - 1);
      
      _bpdata -= *(a - 1);
      _data -= (*a) * (*(a - 1)) + *a;
      gindex -= (*a) * (*(a - 1)) + *a;
      
      // assert(_bpdata == &(bpdata[0]));
      // assert(_data == &(data[0]));
      // assert(gindex == 0);
      
      for(unsigned int y=0;y<rows;y++){
	const T* bptr = _bpdata;
	for(unsigned int x=0;x<cols;x++,gindex++){
	  //_data[x + y*cols] += rate * lgrad[y] * (*bptr);
	  grad[gindex] = -lgrad[y] * (*bptr);
	  bptr++;
	}
      }
      
      
      _data += rows*cols;
    
      for(unsigned int y=0;y<rows;y++, gindex++){
	grad[gindex] = -lgrad[y];
	// _data[y] += rate*lgrad[y];
      }
      
    }
    
    
    return true;
  }


  // return true if nnetwork has stochastic sigmoid activations (clipped to 0 or 1)
  template <typename T>
  bool nnetwork<T>::stochastic(){
    return stochasticActivation;
  }

  template <typename T>
  void nnetwork<T>::setStochastic(bool stochastic){ // sets stochastic sigmoid activations
    stochasticActivation = stochastic;
  }
  
  
  template <typename T> // non-linearity used in neural network
  inline T nnetwork<T>::nonlin(const T& input, unsigned int layer, unsigned int neuron) const throw()
  {
    if(nonlinearity == sigmoidNonLinearity){
      // non-linearity motivated by restricted boltzman machines..
      T output = T(1.0) / (T(1.0) + math::exp(-input));
      
      if(stochasticActivation){
	T r = T(((double)rand())/((double)RAND_MAX));
	
	if(output > r){ output = T(1.0); }
	else{ output = T(0.0); }
      }

      return output;
    }
    else if(nonlinearity == halfLinear){
      // T output = T(0.0);
      
      if(neuron & 1){
#if 0
	// softmax function
	T a = T(5.0);
	output = log(T(1.0) + exp(a*input));
	return output;
#endif
#if 0
	// rectifier non-linearity
	output = T(0.0);
	if(input >= T(0.0)) output = input;
	return output;
#endif
	const T af = T(1.7159f);
	const T bf = T(0.6666f);
	
	T expbx = math::exp(-bf*input ); // numerically more stable (no NaNs)
	T output = af * ( T(2.0f) / (T(1.0f) + expbx) - T(1.0f) );
	return output;
      }
      else{
	return input; // half-the layers nodes are linear!
      }
    }
    else if(nonlinearity == pureLinear){
      return input; // all layers/neurons are linear..
    }
    else{
      assert(0);
    }
    

    
#if 0
    const T af = T(1.7159f);
    const T bf = T(0.6666f);
    
    T expbx = math::exp(-bf*input ); // numerically more stable (no NaNs)
    T output = af * ( T(2.0f) / (T(1.0f) + expbx) - T(1.0f) );
#endif
#if 0        
    T output = input;
    if(output > T(0.999f)) output = T(0.999f);
    else if(output < T(-0.999f)) output = T(-0.999f);
    output = math::atanh(output);
#endif
#if 0
    // T output = math::asinh(input);

    // tanh(x) - 0.5x non-linearity as proposed by a research paper [statistically better gradients]
    // T output = math::tanh(input) - T(0.5)*input;

    // rectifier non-linearity
    T output = T(0.0);
    if(input >= T(0.0))
      output = input;

    // T output = -math::exp(T(-0.5)*input*input);
#endif
#if 0
    // non-linearity motivated by restricted boltzman machines..
    T output = T(1.0) / (T(1.0) + math::exp(-input));

    if(stochasticActivation){
      T r = T(((double)rand())/((double)RAND_MAX));
      
      if(output > r){ output = T(1.0); }
      else{ output = T(0.0); }
    }
    
#endif
#if 0
    T output;

    if(neuron & 1){ // alternates between sigmoid and rectifier non-linearities within each layer
      // rectifier
      output = T(0.0);
      if(input >= T(0.0))
	output = input;
    }
    else{
      // non-linearity motivated by restricted boltzman machines..
      output = T(1.0) / (T(1.0) + math::exp(-input));
    }
#endif
#if 0
    T output;
    
    if(neuron & 1){
      // rectifier non-linearity
      output = T(0.0);
      if(input >= T(0.0))
	output = input;
    }
    else{
      return input; // half-the layers nodes are linear!
    }

    return output;
#endif    
  }
  
  template <typename T> // derivat of non-linearity used in neural network
  inline T nnetwork<T>::Dnonlin(const T& input, unsigned int layer, unsigned int neuron) const throw()
  {
    if(nonlinearity == sigmoidNonLinearity){
      // non-linearity motivated by restricted boltzman machines..
      T output = T(1.0) + math::exp(-input);
      output = math::exp(-input) / (output*output);
      return output;
    }
    else if(nonlinearity == halfLinear){
      if(neuron & 1){
#if 0
	// softmax
	T a = T(5.0);
	T output = a/(T(1.0) + exp(-a*input));
	return output;
#endif
#if 0
	// rectifier non-linearity
	T output = T(0.0);
	if(input >= T(0.0))
	  output = T(1.0);
	return output;
#endif
	const T af = T(1.7159f);
	const T bf = T(0.6666f);
	//const T af = T(1.0f);
	//const T bf = T(1.0f);
	
	T fxa = input/af;
	T output = (T(0.50f)*af*bf) * ((T(1.0f) + fxa)*(T(1.0f) - fxa));
	return output;
      }
      else{
	return 1.0; // half-the layers nodes are linear!
      }
    }
    else if(nonlinearity == pureLinear){
      return 1.0; // all layers/neurons are linear..
    }
    else{
      assert(0);
    }

    
#if 0
    const T af = T(1.7159f);
    const T bf = T(0.6666f);
    //const T af = T(1.0f);
    //const T bf = T(1.0f);
    
    T fxa = input/af;
    T output = (T(0.50f)*af*bf) * ((T(1.0f) + fxa)*(T(1.0f) - fxa));
#endif
#if 0    
    T output = input;
    if(output > T(0.999f)) output = T(0.999f);
    else if(output < T(-0.999f)) output = T(-0.999f);
    output = T(1.0f)/(T(1.0f) - output*output);
#endif
#if 0
    // T output = T(1.0f)/math::sqrt(input*input + T(1.0f));
    
    // T output = input*math::exp(T(-0.5)*input*input);

    // statistically better gradients?
    // T t = math::tanh(input);
    // T output = T(1.0f) - t*t - T(0.5); // statistically better non-linearity?

    // rectifier non-linearity
    T output = T(0.0);
    if(input >= T(0.0))
      output = 1.0;
    
#endif
#if 0
    // non-linearity motivated by restricted boltzman machines..
    T output = T(1.0) + math::exp(-input);
    
    output = math::exp(-input) / (output*output);
#endif
#if 0
    T output;
    
    if(neuron & 1){
      // rectifier non-linearity
      output = T(0.0);
      if(input >= T(0.0))
	output = 1.0;
    }
    else{
      // non-linearity motivated by restricted boltzman machines..
      output = T(1.0) + math::exp(-input);
      
      output = math::exp(-input) / (output*output);
    }
#endif
#if 0
    T output;
    
    if(neuron & 1){
      // rectifier non-linearity
      output = T(0.0);
      if(input >= T(0.0))
	output = 1.0;
    }
    else{
      return 1.0; // half-the layers nodes are linear!
    }
    return output;
#endif
  }

  
  template <typename T>
  inline T nnetwork<T>::inv_nonlin(const T& input, unsigned int layer, unsigned int neuron) const throw(){ // inverse of non-linearity used
#if 0
    const T af = T(1.7159f);
    const T bf = T(0.6666f);
    
    T output = T(2.0f)*af/(input + T(1.0f)) - T(1.0f);
    if(output < T(0.001)) output = T(0.001);
    output = -math::log(output)/bf;
    
#endif
#if 0
    T output = math::sinh(input); // sinh non-linearity.. (sinh()) non-linearity is maybe a bit better non-linearity..
#endif
    // THIS DO NOT WORK CURRENTLY
    
    T output = 0.0f; assert(0); // there is NO inverse function
    
    
    // T output;
    //
    //if(input > T(+0.999f))
    //  output = math::atanh(T(+0.999));
    //else if(input < T(-0.999f))
    //  output = math::atanh(T(-0.999));
    //else
    //  output = math::atanh(input);
      
    
    return output;
  }
  
  
  template <typename T>
  bool nnetwork<T>::gradient_value(const math::vertex<T>& input, math::matrix<T>& grad) const
  {
    const unsigned int L = getLayers();
    
    grad.resize(input_size(), input_size());
    grad.identity();
    
    math::vertex<T> x = input;

    math::matrix<T> A;
    math::vertex<T> a;
    
    for(unsigned int l=0;l<(L-1);l++){
      getWeights(A, l);
      getBias(a, l);
      
      grad = A*grad;
      
      x = A*x + a;
      
      for(unsigned int i=0;i<x.size();i++){
	x[i] *= Dnonlin(x[i], l, i);
      }
    }
    
    getWeights(A, L-1);
    getBias(a, L-1);
    
    grad = A*grad;
    
    return true;
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
#define FNN_VERSION_CFGSTR          "FNN_CONFIG_VERSION"
#define FNN_ARCH_CFGSTR             "FNN_ARCH"
#define FNN_VWEIGHTS_CFGSTR         "FNN_WEIGHTS"
#define FNN_LAYER_W_CFGSTR          "FNN_LWEIGHT%d"
#define FNN_LAYER_B_CFGSTR          "FNN_LBIAS%d"
#define FNN_AFUNS_CFGSTR            "FNN_AFUNCTIONS" // symbolic names for activation function
#define FNN_LAYER_AFUN_PARAM_CFGSTR "FNN_LAFPARAM%d"
#define FNN_MOMENS_CFGSTR           "FNN_MOMENTS"
#define FNN_LRATES_CFGSTR           "FNN_LRATES"

  // version 3.5 labels
#define FNN_NUMWEIGHTS_CFGSTR       "FNN_NUM_WEIGHTS"
#define FNN_WEIGHTS_CFGSTR          "FNN_WEIGHTS%d"

#define FNN_STOCHASTIC_CFGSTR       "FNN_STOCHASTIC"
#define FNN_NONLINEARITY_CFGSTR     "FNN_NONLINEAR"

  //////////////////////////////////////////////////////////////////////

  template <typename T>
  bool nnetwork<T>::save(const std::string& filename) const throw(){
    try{
      whiteice::conffile configuration;
      
      std::vector<int> ints;
      std::vector<float> floats;
      std::vector<std::string> strings;
      
      // writes version information
      {
	// version number = integer/1000
	ints.push_back(2000); // 2.000
	if(!configuration.set(FNN_VERSION_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }
      
      // writes architecture information
      {
	for(unsigned int i=0;i<arch.size();i++)
	  ints.push_back(arch[i]);

	if(!configuration.set(FNN_ARCH_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }
      
      // weights: we just convert everything to a big vertex vector and write it
      {
	math::vertex<T> w;
	
	if(this->exportdata(w) == false)
	  return false;
	
	for(unsigned int i=0;i<w.size();i++){
	  float f;
	  math::convert(f, w[i]);
	  floats.push_back(f);
	}
	
	if(!configuration.set(FNN_VWEIGHTS_CFGSTR, floats))
	  return false;
	
	floats.clear();
      }

      // stochastic activation
      {
	if(stochasticActivation)
	  ints.push_back(1); // true
	else
	  ints.push_back(0);
	
	if(!configuration.set(FNN_STOCHASTIC_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }

      // used non-linearity
      {
	if(nonlinearity == sigmoidNonLinearity){
	  ints.push_back(0);
	}
	else if(nonlinearity == halfLinear){
	  ints.push_back(1);
	}
	else if(nonlinearity == pureLinear){
	  ints.push_back(2);
	}
	else return false; // error!

	if(!configuration.set(FNN_NONLINEARITY_CFGSTR, ints))
	  return false;

	ints.clear();
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
  
  
  ///////////////////////////////////////////////////////////////////////////
  

  // load neuralnetwork data from file
  template <typename T>
  bool nnetwork<T>::load(const std::string& filename) throw(){
    try{
      whiteice::conffile configuration;
      std::vector<std::string> strings;
      std::vector<float> floats;
      std::vector<int> ints;
      
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
      
      if(versionid != 2000) // v2 datafile
	return false;
      
      
      unsigned int memuse;
      
      // gets architecture
      {
	if(!configuration.get(FNN_ARCH_CFGSTR,ints))
	  return false;
	  
	if(ints.size() < 2)
	  return false;
	
	for(unsigned int i=0;i<ints.size();i++)
	  if(ints[i] <= 0) return false;
	
	arch.resize(ints.size());
	
	for(unsigned int i=0;i<arch.size();i++)
	  arch[i] = ints[i];
	
	memuse = 0;
	maxwidth = arch[0];
	
	unsigned int i = 1;
	while(i < arch.size()){
	  memuse += (arch[i-1] + 1)*arch[i];
	  
	  if(arch[i] > maxwidth)
	    maxwidth = arch[i];
	  i++;
	}

	data.resize(memuse);

	state.resize(maxwidth);
	temp.resize(maxwidth);
	lgrad.resize(maxwidth);

	hasValidBPData = false;
	bpdata.resize(1);
	size = memuse;
	
	inputValues.resize(arch[0]);
	outputValues.resize(arch[arch.size() - 1]);
	
	ints.clear();
      }
      
      
      // gets layer weights & biases for the new nnetwork
      {
	math::vertex<T> w;
	
	if(!configuration.get(FNN_VWEIGHTS_CFGSTR, floats))
	  return false;
	
	w.resize(floats.size());
	
	for(unsigned int i=0;i<floats.size();i++)
	  w[i] = T(floats[i]);
	
	if(this->importdata(w) == false)
	  return false;
	
	floats.clear();
      }

      // stochastic activation
      {
	if(!configuration.get(FNN_STOCHASTIC_CFGSTR, ints))
	  return false;

	if(ints.size() != 1) return false;

	stochasticActivation = (bool)ints[0];
	
	ints.clear();
      }
      
      // used nonlinearity
      {
	if(!configuration.get(FNN_NONLINEARITY_CFGSTR, ints))
	  return false;

	if(ints.size() != 1) return false;

	if(ints[0] == 0){
	  nonlinearity = sigmoidNonLinearity;
	}
	else if(ints[0] == 1){
	  nonlinearity = halfLinear;
	}
	else if(ints[0] == 2){
	  nonlinearity = pureLinear;
	}
	else{
	  return false;
	}
	
	ints.clear();
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

  
  //////////////////////////////////////////////////////////////////////  //////////////////////////////////////////////////////////////////////
  
  
  // exports and imports neural network parameters to/from vertex
  template <typename T>
  bool nnetwork<T>::exportdata(math::vertex<T>& v) const throw(){
    v.resize(size);
    
    // NN exports TO vertex, vertex imports FROM NN
    return v.importData(&(data[0]), size, 0);
  }
  
  template <typename T>
  bool nnetwork<T>::importdata(const math::vertex<T>& v) throw(){
    if(v.size() != size)
      return false;
    
    // nn imports FROM vertex, vertex exports TO network
    if(v.exportData(&(data[0]), size, 0) == false)
      return false;

    // "safebox" (keeps data always within sane levels)
    for(unsigned int i=0;i<data.size();i++){
      if(whiteice::math::isnan(data[i])) data[i] = T(0.0);
      if(data[i] < T(-10000.0)) data[i] = T(-10000.0);
      else if(data[i] > T(10000.0)) data[i] = T(10000.0);
    }

    return true;
  }
  
  
  // number of dimensions used by import/export
  template <typename T>
  unsigned int nnetwork<T>::exportdatasize() const throw(){
    return size;
  }

  ////////////////////////////////////////////////////////////////////////////
  
  template <typename T>
  unsigned int nnetwork<T>::getLayers() const throw(){
    return (arch.size()-1); 
  }

  // number of neurons per layer
  template <typename T>
  unsigned int nnetwork<T>::getNeurons(unsigned int layer) const throw()
  {
    if(layer+1 >= arch.size()) return 0;
    return arch[layer+1];
  }

  template <typename T>
  bool nnetwork<T>::getBias(math::vertex<T>& b, unsigned int layer) const throw()
  {
    if(layer+1 >= arch.size()) return false;

    b.resize(arch[layer+1]);

    const T* dptr = &(data[0]);

    for(unsigned int i=0;i<layer;i++){
      dptr = dptr + arch[i]*arch[i+1];
      dptr = dptr + arch[i+1];
    }

    dptr = dptr + arch[layer]*arch[layer+1];

    for(unsigned int i=0;i<b.size();i++)
      b[i] = dptr[i];

    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::setBias(const math::vertex<T>& b, unsigned int layer) throw()
  {
    if(layer+1 >= arch.size()) return false;

    if(b.size() != arch[layer+1]) return false;

    T* dptr = &(data[0]);

    for(unsigned int i=0;i<layer;i++){
      dptr = dptr + arch[i]*arch[i+1];
      dptr = dptr + arch[i+1];
    }

    dptr = dptr + arch[layer]*arch[layer+1];

    for(unsigned int i=0;i<b.size();i++)
      dptr[i] = b[i];

    return true;
  }
  

  template <typename T>
  bool nnetwork<T>::getWeights(math::matrix<T>& w, unsigned int layer) const throw()
  {
    if(layer+1 >= arch.size()) return false;

    w.resize(arch[layer+1], arch[layer]);

    const T* dptr = &(data[0]);

    for(unsigned int i=0;i<layer;i++){
      dptr = dptr + arch[i]*arch[i+1];
      dptr = dptr + arch[i+1];
    }

    for(unsigned int j=0;j<arch[layer+1];j++)
      for(unsigned int i=0;i<arch[layer];i++)
	w(j,i) = dptr[j*arch[layer] + i];

    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::setWeights(const math::matrix<T>& w, unsigned int layer) throw()
  {
    if(layer+1 >= arch.size()) return false;

    if(w.ysize() != arch[layer+1] || w.xsize() != arch[layer])
      return false;

    T* dptr = &(data[0]);

    for(unsigned int i=0;i<layer;i++){
      dptr = dptr + arch[i]*arch[i+1];
      dptr = dptr + arch[i+1];
    }

    for(unsigned int j=0;j<arch[layer+1];j++)
      for(unsigned int i=0;i<arch[layer];i++)
	dptr[j*arch[layer] + i] = w(j,i);

    return true;
  }
  

  template <typename T>
  typename nnetwork<T>::nonLinearity nnetwork<T>::getNonlinearity() const throw()
  {
    return nonlinearity;
  }

  template <typename T>
  bool nnetwork<T>::setNonlinearity(nnetwork<T>::nonLinearity nl)
  {
    nonlinearity = nl;
    return true;
  }
  
  
  template <typename T>
  unsigned int nnetwork<T>::getSamplesCollected() const throw()
  {
    if(samples.size() > 0)
      return samples[0].size();
    else
      return 0;
  }
  
  template <typename T>
  bool nnetwork<T>::getSamples(std::vector< math::vertex<T> >& samples, unsigned int layer) const throw()
  {
    if(layer >= this->samples.size()) return false;
    samples = this->samples[layer];
    return true;
  }
  
  template <typename T>
  void nnetwork<T>::clearSamples() throw()
  {
    for(auto& s : samples)
      s.clear();
  }

  /////////////////////////////////////////////////////////////////////////////
  
  // y = W*x
  template <typename T>
  inline void nnetwork<T>::gemv(unsigned int yd, unsigned int xd, 
			 T* W, T* x, T* y)
  {
    // uses temporary space to handle correctly
    // the case when x and y vectors overlap (or are same)
    // T temp[yd]; [we have global temp to store results]
      
    for(unsigned int j=0;j<yd;j++){
      T sum = T(0.0f);
      for(unsigned int i=0;i<xd;i++)
	sum += W[i + j*xd]*x[i];
      
      temp[j] = sum;
    }
    
    memcpy(y, &(temp[0]), yd*sizeof(T));
  }
  
  
  // s += b;
  template <typename T>
  inline void nnetwork<T>::gvadd(unsigned int dim, T* s, T* b){
    for(unsigned int i=0;i<dim;i++)
      s[i] += b[i];
  }

  template <typename T>
  inline void nnetwork<T>::gemv_gvadd(unsigned int yd, unsigned int xd, 
				      const T* W, T* x, T* y,
				      unsigned int dim, T* s, const T* b) const
  {
    // calculates y = b + W*x (y == x)

    std::vector<T> temp;
    temp.resize(maxwidth);
    
#if 1
    if(typeid(T) == typeid(whiteice::math::blas_real<float>)){
      memcpy(temp.data(), b, yd*sizeof(T));
      
      cblas_sgemv(CblasRowMajor, CblasNoTrans, yd, xd,
		  1.0f, (float*)W, xd, (float*)x, 1, 
		  1.0f, (float*)temp.data(), 1);
      
      memcpy(y, temp.data(), yd*sizeof(T));
    }
    else if(typeid(T) == typeid(whiteice::math::blas_real<double>)){
      memcpy(temp.data(), b, yd*sizeof(T));
      
      cblas_dgemv(CblasRowMajor, CblasNoTrans, yd, xd,
		  1.0f, (double*)W, xd, (double*)x, 1, 
		  1.0f, (double*)temp.data(), 1);
      
      memcpy(y, temp.data(), yd*sizeof(T));
    }
    else
#endif
    {
#if 0
      // uses temporary space to handle correctly
      // the case when x and y vectors overlap (or are same)
      // T temp[yd]; [we have global temp to store results]
      
      for(unsigned int j=0;j<yd;j++){
	T sum = b[j];
	for(unsigned int i=0;i<xd;i++)
	  sum += W[i + j*xd]*x[i];
	
	temp[j] -= sum;
	
	std::cout << temp[j] << std::endl;
      }
#endif         
      
      for(unsigned int j=0;j<yd;j++){
	T sum = b[j];
	for(unsigned int i=0;i<xd;i++)
	  sum += W[i + j*xd]*x[i];
	
	temp[j] = sum;
      }
      
      memcpy(y, temp.data(), yd*sizeof(T));
    }

  }
  
  
  ////////////////////////////////////////////////////////////
  // can be used to decrease memory usage
  
#if 0
  // changes NN to compressed form of operation or
  // back to normal non-compressed form
  template <typename T>  
  bool nnetwork<T>::compress() throw(){
    if(compressed) return false;
    
    compressor = new MemoryCompressor();
    compressor->setMemory(data, sizeof(T)*size);
    
    if(compressor->compress()){
      free(data); data = 0;
      compressor->setMemory(data, 0);
      compressed = true;
      return true;
    }
    else{
      if(compressor->getTarget() != 0)
	free(compressor->getTarget());
      
      delete compressor;
      compressor = 0;
      return false;
    }
  }
  
  
  template <typename T>
  bool nnetwork<T>::decompress() throw(){
    if(compressed == false) return false;
    
    if(compressor->decompress()){
      data = (T*)( compressor->getMemory() );
      
      free(compressor->getTarget());
      
      delete compressor;
      
      compressor = 0;
      compressed = false;
      
      return true;
    }
    else
      return false;
  }
  
  
  template <typename T>
  bool nnetwork<T>::iscompressed() const throw(){
    return compressed;
  }
  
  
  // returns compression ratio: compressed/orig
  template <typename T>
  float nnetwork<T>::ratio() const throw(){
    if(compressed) return compressor->ratio();
    else return 1.0f;
  }
#endif
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class nnetwork< float >;
  template class nnetwork< double >;  
  template class nnetwork< math::blas_real<float> >;
  template class nnetwork< math::blas_real<double> >;
  
};
