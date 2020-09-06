// TODO:
//   - refactor gradient etc code 
// 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <exception>
#include <stdexcept>
#include <typeinfo>

// remove define if you want large initialization arguments
// (good for deep nnetworks(?))
// #define SMALL_RANDOMIZE_INIT 1 [DON'T ENABLE / DON'T WORK WITH DEEP NETWORKS]

#include "nnetwork.h"
#include "dataset.h"
#include "dinrhiw_blas.h"
#include "Log.h"

namespace whiteice
{
  
  
  template <typename T>
  nnetwork<T>::nnetwork()
  {
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

    state.resize(maxwidth);
    temp.resize(maxwidth);
    lgrad.resize(maxwidth);

    // there are arch.size()-1 layers in our network
    // which are all optimized as the default
    frozen.resize(arch.size()-1);
    for(unsigned int i=0;i<frozen.size();i++)
      frozen[i] = false;

    nonlinearity.resize(arch.size()-1);
    for(unsigned int i=0;i<nonlinearity.size();i++)
      nonlinearity[i] = rectifier;
    
    nonlinearity[nonlinearity.size()-1] = pureLinear;

    
    inputValues.resize(1);
    outputValues.resize(1);

    retain_probability = T(1.0);

    randomize();
  }
  
  
  template <typename T>
  nnetwork<T>::nnetwork(const nnetwork<T>& nn)
  {
    inputValues.resize(nn.inputValues.size());
    outputValues.resize(nn.outputValues.size());

    hasValidBPData = nn.hasValidBPData;
    maxwidth = nn.maxwidth;
    size = nn.size;

    arch   = nn.arch;
    bpdata = nn.bpdata;
    data   = nn.data;
    state  = nn.state;
    temp   = nn.temp;
    lgrad  = nn.lgrad;
    nonlinearity = nn.nonlinearity;
    frozen = nn.frozen;
    retain_probability = nn.retain_probability;
    dropout = nn.dropout;
  }
  
  
  template <typename T>
  nnetwork<T>::nnetwork(const std::vector<unsigned int>& nnarch,
			const typename nnetwork<T>::nonLinearity nl)
    
  {
    if(nnarch.size() < 2)
      throw std::invalid_argument("invalid network architecture");

    maxwidth = 0;
    
    for(unsigned int i=0;i<nnarch.size();i++){
      if(nnarch[i] <= 0)
	throw std::invalid_argument("invalid network architecture");
      if(nnarch[i] > maxwidth)
	maxwidth = nnarch[i];
    }

    // sets up architecture
    arch = nnarch;

    nonlinearity.resize(arch.size()-1);
    for(unsigned int i=0;i<nonlinearity.size();i++)
      nonlinearity[i] = nl;

    // HERE WE ALWAYS SET LAST LAYERS NONLINEARITY TO BE LINEAR FOR NOW..
    // nonlinearity[nonlinearity.size()-1] = pureLinear; 
    
    unsigned int memuse = 0;
    
    for(unsigned int i=0;i<arch.size();i++){
      if(i > 0) 
	memuse += (arch[i-1] + 1)*arch[i];
    }
    
    size = memuse;
    
    data.resize(size);
    
    state.resize(maxwidth);
    temp.resize(maxwidth);
    lgrad.resize(maxwidth);
     
    inputValues.resize(arch[0]);
    outputValues.resize(arch[arch.size()-1]);
    
    // there are arch.size()-1 layers in our network
    // which are all optimized as the default
    frozen.resize(arch.size()-1);
    for(unsigned int i=0;i<frozen.size();i++)
      frozen[i] = false;

    // intializes all weights/layers (randomly)
    randomize();
   
    hasValidBPData = false;

    retain_probability = T(1.0);
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

    nonlinearity = nn.nonlinearity;
    frozen = nn.frozen;
    retain_probability = nn.retain_probability;
    dropout = nn.dropout;
    
    data = nn.data;
    bpdata = nn.bpdata;

    state = nn.state;
    temp = nn.temp;
    lgrad = nn.lgrad;

    inputValues = nn.inputValues;
    outputValues = nn.outputValues;
    
    samples = nn.samples;

    return (*this);
  }

  template <typename T>
  void nnetwork<T>::printInfo() const
  {
    // prints nnetwork information (mostly for debugging purposes)

    printf("NETWORK LAYOUT (%d): \n", getLayers());

    for(unsigned int l=0;l<getLayers();l++){
      bool frozen = this->getFrozen(l);
      unsigned int nl = (unsigned int)this->getNonlinearity(l);
      unsigned int inputsize = this->getInputs(l);
      unsigned int width = this->getNeurons(l);

      if(frozen)
	printf("%d->%d (F%d) ", inputsize, width, nl);
      else
	printf("%d->%d( %d) ", inputsize, width, nl);
    }

    printf("\n");
    fflush(stdout);

    
    printf("NEURAL NETWORK LAYER WEIGHTS:\n");

    for(unsigned int l=0;l<getLayers();l++){
      math::matrix<T> W;
      math::vertex<T> b;
      
      this->getBias(b, l);
      this->getWeights(W, l);
    
      std::cout << "W(" << l << ") = " << W << std::endl;
      std::cout << "b(" << l << ") = " << b << std::endl;
    }
      
  }

  
  template <typename T>
  void nnetwork<T>::diagnosticsInfo() const
  {
    char buffer[128];    
    snprintf(buffer, 128, "nnetwork: DIAGNOSTIC/MAXVALUE (%d layers):",
	     getLayers());
    whiteice::logging.info(buffer);


    for(unsigned int l=0;l<getLayers();l++){
      math::matrix<T> W;
      math::vertex<T> b;
      auto maxvalueW = -abs(T(INFINITY));
      auto maxvalueb = -abs(T(INFINITY));
      auto minvalueW = abs(T(+INFINITY));
      auto minvalueb = abs(T(+INFINITY));
      
      this->getBias(b, l);
      this->getWeights(W, l);

      for(unsigned int i=0;i<b.size();i++){
	if(maxvalueb < abs(b[i]))
	  maxvalueb = abs(b[i]);
	if(minvalueb > abs(b[i]))
	  minvalueb = abs(b[i]);
      }

      for(unsigned int j=0;j<W.ysize();j++){
	for(unsigned int i=0;i<W.xsize();i++){
	  if(maxvalueW < abs(W(j, i)))
	    maxvalueW = abs(W(j, i));
	  if(minvalueW > abs(W(j, i)))
	    minvalueW = abs(W(j, i));
	}
      }


      double temp = 0.0;
      whiteice::math::convert(temp, minvalueW);

      double temp2 = 0.0;
      whiteice::math::convert(temp2, maxvalueW);

      double temp3 = 0.0;
      whiteice::math::convert(temp3, minvalueb);

      double temp4 = 0.0;
      whiteice::math::convert(temp4, maxvalueb);

      snprintf(buffer, 128, "nnetwork: LAYER %d/%d %d-%d MIN/MAX ABS-VALUE W=%f/%f b=%f/%f",
	       l+1, getLayers(), getInputs(l), getNeurons(l),
	       temp, temp2, temp3, temp4);
      whiteice::logging.info(buffer);
      
    }
  }

  
  ////////////////////////////////////////////////////////////

  // returns input and output dimensions of neural network
  template <typename T>
  unsigned int nnetwork<T>::input_size() const {
    if(arch.size() > 0) return arch[0];
    else return 0;
  }
  
  template <typename T>
  unsigned int nnetwork<T>::output_size() const {
    unsigned int index = arch.size()-1;
    
    if(arch.size() > 0) return arch[index];
    else return 0;
  }

  template <typename T>
  unsigned int nnetwork<T>::gradient_size() const 
  {
    return size; // number of parameters in neural network
  }


  template <typename T>
  void nnetwork<T>::getArchitecture(std::vector<unsigned int>& nn_arch) const
  {
    nn_arch = this->arch;
  }

  // invalidates all data and essentially creates a new network over previous one
  template <typename T>
  bool nnetwork<T>::setArchitecture(const std::vector<unsigned int>& nnarch,
				    const typename nnetwork<T>::nonLinearity nl)
  {
    if(nnarch.size() <= 1) return false;

    if(arch.size() == nnarch.size()){
      unsigned int same = true;

      for(unsigned int i=0;i<nnarch.size();i++){
	if(nnarch[i] <= 0) return false;
	if(arch[i] != nnarch[i]){
	  same = false;
	  break;
	}
      }

      for(unsigned int i=0;i<nonlinearity.size();i++){
	if(nonlinearity[i] != nl){
	  same = false;
	  break;
	}
      }
	  
      
      if(same == true)
	return true; // we already have the target NN architecture
    }

    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>)){
      if(nl == rectifier || nl == sigmoid){
	printf("nnetwork::setAchitecture() warning: chosen non-linearity don't work even with regularized complex-valued neural network.");
      }
    }
    
    ///////////////////////////////////////////////////////////////////////////
    // resets nnetwork<T> parameters
    
    maxwidth = 0;
    
    for(unsigned int i=0;i<nnarch.size();i++){
      if(nnarch[i] > maxwidth)
	maxwidth = nnarch[i];
    }

    // sets up architecture
    arch = nnarch;

    nonlinearity.resize(arch.size()-1);
    for(unsigned int i=0;i<nonlinearity.size();i++)
      nonlinearity[i] = nl;

    // HERE WE ALWAYS SET LAST LAYERS NONLINEARITY TO BE LINEAR FOR NOW..
    // nonlinearity[nonlinearity.size()-1] = pureLinear; 
    
    unsigned int memuse = 0;
    
    for(unsigned int i=0;i<arch.size();i++){
      if(i > 0) 
	memuse += (arch[i-1] + 1)*arch[i];
    }
    
    size = memuse;
    
    data.resize(size);
    
    state.resize(maxwidth);
    temp.resize(maxwidth);
    lgrad.resize(maxwidth);
     
    inputValues.resize(arch[0]);
    outputValues.resize(arch[arch.size()-1]);
    
    // there are arch.size()-1 layers in our network
    // which are all optimized as the default
    frozen.resize(arch.size()-1);
    for(unsigned int i=0;i<frozen.size();i++)
      frozen[i] = false;

    dropout.clear(); // removes dropout data after architecture change
    
    hasValidBPData = false;

    return true;
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

      T* bpptr = NULL;
      
      {
	unsigned int bpsize = 0;
	
	for(unsigned int i=0;i<arch.size();i++)
	  bpsize += arch[i];
	
	bpdata.resize(bpsize);
	memset((T*)&(bpdata[0]), 0, bpsize*sizeof(T)); // TODO remove after debug
      }

      bpptr = &(bpdata[0]);

      // saves input to backprogation data
      {
	memcpy((T*)bpptr, (const T*)&(state[0]), arch[aindex]*sizeof(T));
	
	bpptr += arch[aindex];
      }

      T* dptr = &(data[0]);
      
      while(aindex+1 < arch.size()){
	
	if(collectSamples){
	  math::vertex<T> x;
	  x.resize(arch[aindex]);
	  memcpy((T*)&(x[0]), (const T*)&(state[0]), arch[aindex]*sizeof(T));
	  samples[aindex].push_back(x);
	}
	
	// gemv(a[1], a[0], dptr, state, state); // s = W*s
	// gvadd(a[1], state, dptr + a[0]*a[1]); // s += b;

	// s = b + W*s
	gemv_gvadd(arch[aindex+1], arch[aindex], dptr, &(state[0]), &(state[0]),
		   arch[aindex+1], &(state[0]), dptr + arch[aindex]*arch[aindex+1]);

	
	// COPIES LOCAL FIELD v FROM state TO BACKPROGRAGATION DATA
	{
	  memcpy((T*)bpptr, (const T*)&(state[0]), arch[aindex+1]*sizeof(T));
	  
	  bpptr += arch[aindex+1];
	}

	// s = g(v)

	for(unsigned int i=0;i<arch[aindex+1];i++){
	  state[i] = nonlin(state[i], aindex, i);
	}

	dptr += (arch[aindex] + 1)*arch[aindex+1]; // matrix W and bias b
	
	aindex++; // next layer
      }
      
    }
    else{
      T* dptr = &(data[0]);
      
      while(aindex+1 < arch.size()){
	
	if(collectSamples){
	  math::vertex<T> x;
	  x.resize(arch[aindex]);
	  memcpy((T*)&(x[0]), (const T*)&(state[0]), arch[aindex]*sizeof(T));
	  samples[aindex].push_back(x);
	}
	
	// gemv(a[1], a[0], dptr, state, state); // s = W*s
	// gvadd(a[1], state, dptr + a[0]*a[1]); // s += b;
	
	// s = b + W*s
	gemv_gvadd(arch[aindex+1], arch[aindex], dptr, &(state[0]), &(state[0]),
		   arch[aindex+1], &(state[0]), dptr + arch[aindex]*arch[aindex+1]);
	
	// s = g(v)

	for(unsigned int i=0;i<arch[aindex+1];i++){
	  state[i] = nonlin(state[i], aindex, i);
	}

	dptr += (arch[aindex] + 1)*arch[aindex+1]; // matrix W and bias b
	aindex++; // next layer
      }
      
    }
    
    
    if(!outputValues.importData(&(state[0]))){
      std::cout << "Failed to import data to vertex from memory." << std::endl;
      return false;
    }
    
    // FIXME (remove) debugging checks bpdata structure
    if(bpdata.size() > 0)
    {
      auto maxvalue = abs(bpdata[0]);
      
      for(const auto& v : bpdata){
	if(abs(v) > maxvalue)
	  maxvalue = abs(v);
      }
      
      // std::cout << "MAX BPDATA: " << maxvalue << std::endl;
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
	  
	for(unsigned int i=0;i<arch[aindex+1];i++){
	  // state[i] = nonlin(state[i], aindex + 1, i);
	  state[i] = nonlin(state[i], aindex, i);
	}
	
	dptr += (arch[aindex] + 1)*arch[aindex+1]; // matrix W and bias b
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
  bool nnetwork<T>::randomize(const unsigned int type,
			      const bool smallvalues)
  {
    memset(data.data(), 0, sizeof(T)*data.size()); // TODO remove after debugging

    if(type == 0){
      // bulk random [-1,+1] initialization
      unsigned int start = 0;
      unsigned int end   = 0;

      const whiteice::math::blas_complex<double> ar(2.0f,0.0f), br(1.0f, 0.0f);
      const whiteice::math::blas_complex<double> ai(0.0f,2.0f), bi(0.0f, 1.0f);

      for(unsigned int i=1;i<arch.size();i++){
      	end   += (arch[i-1] + 1)*arch[i];

      	if(frozen[i-1] == false){
	  for(unsigned int j=start;j<end;j++){
	    // RNG is real valued, a and b are complex
	    // this means value is complex valued [-1,+1]+[-1,+1]i
	    const auto value = (T(ar)*rng.uniform() - T(br)) + (T(ai)*rng.uniform() - T(bi));
	    
	    whiteice::math::convert(data[j], value); // if we have real data we keep only real part
	  }
      	}
	
      	start += (arch[i-1] + 1)*arch[i];
      }

    }
    else if(type == 1)
    {
      unsigned int start = 0;
      unsigned int end   = 0;
      
      for(unsigned int i=1;i<arch.size();i++){
	end   += (arch[i-1] + 1)*arch[i];
	
	if(frozen[i-1] == false){ // do not touch frozen layers when using randomize()
	  
	  // this initialization is as described in the paper of Xavier Glorot
	  // "Understanding the difficulty of training deep neural networks"
	  
	  T var = math::sqrt(6.0f / (arch[i-1] + arch[i]));
	  // T scaling = T(2.2); // for asinh()

#ifdef SMALL_RANDOMIZE_INIT
	  T scaling = T(0.1f); // was chosen value
#else
	  T scaling = T(1.0f); // no scaling so use values as in paper
	  if(smallvalues)
	    scaling = T(0.1f);
#endif
	  
	  var *= scaling;

	  const whiteice::math::blas_complex<double> ar(2.0f,0.0f), br(1.0f, 0.0f);
	  const whiteice::math::blas_complex<double> ai(0.0f,2.0f), bi(0.0f, 1.0f);

	  // set weight values W
	  for(unsigned int j=start;j<(end-arch[i]);j++){
	    // var * ( [-1+1]+[-1,+1]i )
	    const auto value = ((ar*rng.uniform() - br) + (ai*rng.uniform() - bi))*var;
	    whiteice::math::convert(data[j], value);
	    
	    //T r = T(2.0f)*rng.uniform() - T(1.0f); // [-1,1]    
	    //data[j] = var*r;
	    // NOTE: asinh(x) requires aprox 3x bigger values before
	    // reaching saturation than tanh(x)
	  }

	  // sets bias terms to zero
	  for(unsigned int j=(end-arch[i]);j<end;j++){
	    data[j] = T(0.0f);
	  }
	  
	}
	
	start += (arch[i-1] + 1)*arch[i];
      }

      assert(start == data.size()); // we have processed the whole memory area correctly
    }
    else{ // type = 2
      unsigned int start = 0;
      unsigned int end   = 0;
      
      for(unsigned int i=1;i<arch.size();i++){
	end   += (arch[i-1] + 1)*arch[i];
	
	if(frozen[i-1] == false){ // do not touch frozen layers when using randomize()
	  // this initialization is as described in the paper of Xavier Glorot
	  // "Understanding the difficulty of training deep neural networks"
	  
	  // keep data variance aproximately 1 (assume inputs x1..xN have unit variance)
	  T var = math::sqrt(1.0f / arch[i-1]);
	  
#ifdef SMALL_RANDOMIZE_INIT
	  T scaling = T(0.1); // was chosen value
#else
	  T scaling = T(1.0);
	  if(smallvalues)
	    scaling = T(0.1);
#endif
	  
	  var *= scaling;

	  // set weight values W
	  for(unsigned int j=start;j<(end-arch[i]);j++){
	    T r = rng.normal();
	    data[j] = var*r;
	  }
	  
	  // sets bias terms to zero
	  for(unsigned int j=(end-arch[i]);j<end;j++){
	    data[j] = T(0.0f);
	  }
	  
	}
	
	start += (arch[i-1] + 1)*arch[i];
      }
      
      assert(start == data.size()); // we have processed the whole memory area correctly
    }
    
    
    
    return true;
  }
  

  // set parameters to fit the data from dataset (we set weights to match data values)
  // [experimental code]
  template <typename T>
  bool nnetwork<T>::presetWeightsFromData(const whiteice::dataset<T>& ds)
  {
    if(ds.getNumberOfClusters() < 2) return false;
    if(ds.dimension(0) != input_size()) return false;
    if(ds.size(0) <= 0) return false;

    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>)){
      printf("Warning/ERROR: nnetwork::presetWeightsFromData() don't work with complex data.\n");
      printf("make symmetric_pseudoinverse() work to get this function working.\n");
      return false;
    }

    this->randomize(); // sets biases to random values

    // set weights to match to data so that inner product is matched exactly to some data
    std::vector< whiteice::math::vertex<T> > inputdata;
    std::vector< whiteice::math::vertex<T> > outputdata;
    std::vector< whiteice::math::vertex<T> > realoutput;

    for(unsigned int i=0;i<ds.size(0);i++)
      inputdata.push_back(ds.access(0, i));

    for(unsigned int i=0;i<ds.size(1);i++)
      realoutput.push_back(ds.access(1, i));
    

    for(unsigned int l=0;l<getLayers();l++){
      math::matrix<T> W;
      math::vertex<T> b;
      
      getWeights(W, l);
      getBias(b, l);

      if(l != getLayers()-1){ // not the last layer
	for(unsigned int j=0;j<W.ysize();j++){
	  unsigned int k1 = rand() % inputdata.size();
	  unsigned int k2 = rand() % inputdata.size();

	  T norm1 = inputdata[k1].norm(); norm1 = norm1*norm1;
	  T norm2 = inputdata[k2].norm(); norm2 = norm2*norm2;
	  if(norm1 <= T(0.0f)) norm1 = T(1.0f);
	  if(norm2 <= T(0.0f)) norm2 = T(1.0f);

	  for(unsigned int i=0;i<W.xsize();i++){
	    W(j,i) = T(0.5f)*inputdata[k1][i]/norm1 + T(0.5f)*inputdata[k2][i]/norm2;
	  }
	}

	b.zero();
      }
      else{ // the last layer, calculate linear mapping y=A*x + b
	// W = Cyx*inv(Cxx)
	// b = E[y] - W*E[x]
	
	whiteice::math::vertex<T> mx, my;
	whiteice::math::matrix<T> Cxx, Cyx;

	mean_covariance_estimate(mx, Cxx, inputdata);
	mean_crosscorrelation_estimate(mx, my, Cyx, inputdata, realoutput);

	if(Cxx.symmetric_pseudoinverse() == false){
	  printf("ERROR: nnetwork<>::presetWeightsFromData(): symmetric pseudoinverse FAILED.\n");
	  fflush(stdout);
	  assert(0);
	}

	W = Cyx*Cxx;
	b = my - W*mx;
      }

      setWeights(W, l);
      setBias(b, l);

      outputdata.resize(inputdata.size());

      // processes data in parallel
#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<inputdata.size();i++){
	auto out = W*inputdata[i] + b;
	for(unsigned int n=0;n<out.size();n++)
	  out[n] = nonlin(out[n], l, n);

	outputdata[i] = out;
      }

      inputdata = outputdata;
      outputdata.clear();
    }

    return true;
  }
  

  // set parameters to fit the data from dataset but uses random weights except for the last layer
  // [experimental code]
  template <typename T>
  bool nnetwork<T>::presetWeightsFromDataRandom(const whiteice::dataset<T>& ds)
  {
    if(ds.getNumberOfClusters() < 2) return false;
    if(ds.dimension(0) != input_size()) return false;
    if(ds.size(0) <= 0) return false;

    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>)){
      printf("Warning/ERROR: nnetwork::presetWeightsFromDataRandom() don't work with complex data.\n");
      printf("make symmetric_pseudoinverse() work to get this function working.\n");
      return false;
    }
    
    // set weights to match to data so that inner product is matched exactly to some data
    std::vector< whiteice::math::vertex<T> > inputdata;
    std::vector< whiteice::math::vertex<T> > outputdata;
    std::vector< whiteice::math::vertex<T> > realoutput;
    
    for(unsigned int i=0;i<ds.size(0);i++)
      inputdata.push_back(ds.access(0, i));
    
    for(unsigned int i=0;i<ds.size(1);i++)
      realoutput.push_back(ds.access(1, i));
    
    for(unsigned int l=0;l<getLayers();l++){
      math::matrix<T> W;
      math::vertex<T> b;
      
      getWeights(W, l);
      getBias(b, l);
      
      if(l != getLayers()-1){ // not the last layer
	
	for(unsigned int j=0;j<W.ysize();j++){
	  T var = math::sqrt(1.0f / W.xsize());
	  for(unsigned int i=0;i<W.xsize();i++){
	    W(j,i) = var*rng.normal();
	  }
	}
	
	b.zero();
      }
      else{ // the last layer, calculate linear mapping y=A*x + b
	// W = Cyx*inv(Cxx)
	// b = E[y] - W*E[x]
	
	whiteice::math::vertex<T> mx, my;
	whiteice::math::matrix<T> Cxx, Cyx;
	
	mean_covariance_estimate(mx, Cxx, inputdata);
	mean_crosscorrelation_estimate(mx, my, Cyx, inputdata, realoutput);
	
	if(Cxx.symmetric_pseudoinverse() == false){
	  printf("ERROR: nnetwork<>::presetWeightsFromDataRandom(): symmetric pseudoinverse FAILED.\n");
	  fflush(stdout);
	  assert(0);
	}
	
	W = Cyx*Cxx;
	b = my - W*mx;
      }
      
      setWeights(W, l);
      setBias(b, l);
      
      outputdata.resize(inputdata.size());
      
      // processes data in parallel
#pragma omp parallel for schedule(auto)
      for(unsigned int i=0;i<inputdata.size();i++){
	auto out = W*inputdata[i] + b;
	for(unsigned int n=0;n<out.size();n++)
	  out[n] = nonlin(out[n], l, n);
	
	outputdata[i] = out;
      }
      
      inputdata = outputdata;
      outputdata.clear();
    }
    
    return true;
  }
  
  
  
  
  // calculates gradient of [ 1/2*(network(last_input|w) - last_output)^2 ]
  // => error*GRAD[function(x,w)]
  // uses values stored by previous computation
  // accesses to the neural network matrix and vector parameter data (memory block)
  template <typename T>
  bool nnetwork<T>::mse_gradient(const math::vertex<T>& error,
				 math::vertex<T>& grad) const
  {
    if(!hasValidBPData)
      return false;

    
    bool complex_data = false;
    
    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>))
    {
      // with complex data we need to take conjugate of gradient values
      complex_data = true;
    }
    
    
    // TODO remove later: for debugging 
    {
      memset(lgrad.data(), 0, lgrad.size()*sizeof(T));
      memset(temp.data(), 0 , temp.size()*sizeof(T));
    }
    
    T* lgrad = (T*)this->lgrad.data();
    T* temp  = (T*)this->temp.data();
  
    // initial (last layer gradient)
    // error[i] * NONLIN'(v) 
    
    int layer = arch.size() - 2;
    const unsigned int *a = &(arch[arch.size()-1]);

    assert(error.size() == *a);

    if(!error.exportData(lgrad, error.size(), 0)){
      assert(0); // TODO: remove after debugging
      return false;
    }

    const T* _bpdata = &(bpdata[0]);
    
    // goes to end of bpdata input lists
    for(unsigned int i=0;i<arch.size();i++)
      _bpdata += arch[i];

    const unsigned int rows = *a;
    // const unsigned int cols = *(a - 1);
    
    _bpdata -= rows;
    const T* bptr = _bpdata;

    
    // calculates local gradient

    for(unsigned int i=0;i<arch[arch.size()-1];i++){
      if(complex_data){
	lgrad[i].conj();
      }
      
      lgrad[i] *= Dnonlin(*bptr, layer , i);
      bptr++;
    }
  
    const T* _data = &(data[0]);
    
    // goes to end of data area
    for(unsigned int i=1;i<arch.size();i++){
      _data += ((arch[i-1])*(arch[i])) + arch[i];
    }
    
    grad.resize(size);
    unsigned int gindex = grad.size(); // was "unsigned gindex" ..

    
    
    while(layer > 0){
      // updates W and b in this layer
      
      // delta W = (lgrad * input^T)
      // delta b =  lgrad;
      
      const unsigned int rows = *a;
      const unsigned int cols = *(a - 1);

      _bpdata -= cols;
      _data -= rows*cols + rows;
      gindex -= rows*cols + rows;
      
      const T* dptr = _data;

      {
	// matrix W gradients
	for(unsigned int y=0;y<rows;y++){
	  const T* bptr = _bpdata;
	  for(unsigned int x=0;x<cols;x++,gindex++){
	    grad[gindex] = lgrad[y] * nonlin(*bptr, layer-1, x);
	    bptr++;
	  }
	}
	
	dptr += (rows*cols);
	
	// bias b gradients
	for(unsigned int y=0;y<rows;y++,gindex++){
	  grad[gindex] = lgrad[y];
	}

	gindex -= (rows*cols + rows);

	// zeroes gradient for this layer
	// (FIXME: optimize me and do not calculate the gradient at all!)
	if(frozen[layer]){
	  memset(&(grad[gindex]), 0, sizeof(T)*(rows*cols + rows));
	}
      }
      

      // calculates next lgrad
      
      // for hidden layers: local gradient is:
      // lgrad[n] = diag(..g'(v[i])..)*(W^t * lgrad[n+1])

      const T* bptr = _bpdata;
      
      if(typeid(T) == typeid(whiteice::math::blas_real<float>)){
	
	cblas_sgemv(CblasRowMajor, CblasTrans,
		    rows, cols,
		    1.0f, (float*)_data, cols, (float*)lgrad, 1,
		    0.0f, (float*)temp, 1);
	
	for(unsigned int x=0;x<cols;x++){
	  temp[x] *= Dnonlin(*bptr, layer - 1, x);
	  bptr++;
	}
	
      }
      else if(typeid(T) == typeid(whiteice::math::blas_real<double>)){
	
	cblas_dgemv(CblasRowMajor, CblasTrans,
		    rows, cols,
		    1.0f, (double*)_data, cols, (double*)lgrad, 1,
		    0.0f, (double*)temp, 1);
	
	for(unsigned int x=0;x<cols;x++){
	  temp[x] *= Dnonlin(*bptr, layer - 1, x);
	  bptr++;
	}
	
      }
      else
	{
	for(unsigned int x=0;x<cols;x++){
	  T sum = T(0.0f);
	  for(unsigned int y=0;y<rows;y++)
	    sum += lgrad[y]*_data[x + y*cols];
	  
	  sum *= Dnonlin(*bptr, layer - 1, x);
	  
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
      
      layer--;
      a--;
    }
    
    
    {
      // calculates the first layer's delta W and delta b 
      // 
      // delta W += (lgrad * input^T)
      // delta b += lgrad;
      
      const unsigned int rows = *a;
      const unsigned int cols = *(a - 1);
      
      _bpdata -= cols;
      _data -= rows*cols + rows;
      gindex -= rows*cols + rows;
      
      assert(_bpdata == &(bpdata[0]));
      assert(_data == &(data[0]));
      assert(gindex == 0);
      
      for(unsigned int y=0;y<rows;y++){
	const T* bptr = _bpdata;
	for(unsigned int x=0;x<cols;x++,gindex++){
	  grad[gindex] = lgrad[y] * (*bptr); // bp data is here the input x !!
	  bptr++;
	}
      }
      
      
      _data += rows*cols;
    
      for(unsigned int y=0;y<rows;y++, gindex++){
	grad[gindex] = lgrad[y];
	// _data[y] += rate*lgrad[y];
      }

      gindex -= rows*cols + rows;
      assert(gindex == 0); // DEBUGGING REMOVE LATER
      
      // zeroes gradient for this layer
      // (FIXME: optimize me and do not calculate the gradient!)
      if(frozen[layer]){
	memset(&(grad[gindex]), 0, sizeof(T)*(rows*cols + rows));
      }
      
    }

    // for complex neural networks we need to calculate conjugate value of
    // the whole gradient (for this to work we need to calculate conjugate value
    // of error term (f(z)-y) as the input for the gradient operation
    if(complex_data){
      grad.conj();
    }
    
    
    return true;
  }


  /* 
   * calculates jacobian/gradient of parameter weights w f(v|w)
   *
   * For math documentation read docs/neural_network_gradient.tm
   *
   */
  template <typename T>
  bool nnetwork<T>::jacobian(const math::vertex<T>& input,
			     math::matrix<T>& grad) const
  {
    if(input.size() != this->input_size()) return false;
    
    // local fields for each layer and input
    std::vector< whiteice::math::vertex<T> > v;

    auto x = input;

    // forward pass: calculates local fields
    int l = 0;

    whiteice::math::matrix<T> W;
    whiteice::math::vertex<T> b;
    
    for(l=0;l<(signed)getLayers();l++){

      getWeights(W, l);
      getBias(b, l);
      
      x = W*x + b;

      v.push_back(x); // stores local field

      for(unsigned int i=0;i<getNeurons(l);i++){
	x[i] = nonlin(x[i], l, i);
      }
    }

    /////////////////////////////////////////////////
    // backward pass: calculates gradients

    l--;

    grad.resize(output_size(), gradient_size());
    grad.zero(); // REMOVE ME: for debugging..

    whiteice::math::matrix<T> lgrad; // calculates local gradient
    lgrad.resize(output_size(), output_size());
    lgrad.zero();

    for(unsigned int i=0;i<output_size();i++){
      lgrad(i,i) = Dnonlin(v[l][i], l, i);
    }

    unsigned int index = gradient_size();

    for(;l>0;l--){
      
      getWeights(W, l);
      getBias(b, l);

      // calculates gradient
      {
	whiteice::math::vertex<T> u(getNeurons(l));
	
	index -= W.ysize()*W.xsize() + b.size();

	// weight matrix gradient
	for(unsigned int j=0;j<W.ysize();j++){
	  for(unsigned int i=0;i<W.xsize();i++){
	    u.resize(getNeurons(l));
	    u.zero();
	    u[j] = nonlin(v[l-1][i], l-1, i);

	    u = lgrad*u;

	    for(unsigned int k=0;k<grad.ysize();k++)
	      grad(k, index) = u[k];

	    index++;
	  }
	}

	// bias vector gradient
	for(unsigned int j=0;j<b.size();j++){
	  u.resize(getNeurons(l));
	  u.zero();
	  u[j] = T(1.0f);

	  u = lgrad*u;

	  for(unsigned int k=0;k<grad.ysize();k++)
	    grad(k, index) = u[k];

	  index++;
	}

	index -= W.ysize()*W.xsize() + b.size();
      }


      // updates gradient
      auto temp = lgrad * W;
      lgrad.resize(temp.ysize(), getNeurons(l-1));
      
      for(unsigned int j=0;j<lgrad.ysize();j++)
	for(unsigned int i=0;i<lgrad.xsize();i++){
	  // lgrad(j,i) = temp(j,i)*nonlin(v[l-1][i], l-1, i);
	  lgrad(j,i) = temp(j,i)*Dnonlin(v[l-1][i], l-1, i);
	}
      
    }


    // l = 0 layer (input layer)
    {
      whiteice::math::matrix<T> W;
      whiteice::math::vertex<T> b;

      getWeights(W, l);
      getBias(b, l);

      // calculates gradient
      {
	whiteice::math::vertex<T> u(getNeurons(l));
	
	index -= W.ysize()*W.xsize() + b.size();

	// weight matrix gradient
	for(unsigned int j=0;j<W.ysize();j++){
	  for(unsigned int i=0;i<W.xsize();i++){
	    u.resize(getNeurons(l));
	    u.zero();
	    u[j] = input[i];

	    u = lgrad*u;

	    for(unsigned int k=0;k<grad.ysize();k++)
	      grad(k, index) = u[k];

	    index++;
	  }
	}

	// bias vector gradient
	for(unsigned int i=0;i<b.size();i++){
	  u.resize(getNeurons(l));
	  u.zero();
	  u[i] = T(1.0);

	  u = lgrad*u;

	  for(unsigned int k=0;k<grad.ysize();k++)
	    grad(k, index) = u[k];

	  index++;
	}

	index -= W.ysize()*W.xsize() + b.size();
      }
      
    }

    
    
    assert(index == 0);

    return true;
  }
  
  
  template <typename T> // non-linearity used in neural network
  inline T nnetwork<T>::nonlin(const T& input, unsigned int layer, unsigned int neuron) const 
  {
    assert(layer < getLayers());
    assert(neuron < getNeurons(layer));

    if(dropout.size() > 0){
      if(dropout[layer][neuron])
	return T(0.0f);
    }

    if(nonlinearity[layer] == softmax){
      const T k = T(1.50f);

      T in = input;
      if(abs(in) > abs(T(+20.0f)))
	in = abs(T(+20.0f))*in/abs(in);

      const T value = T(1.0f) + whiteice::math::exp(k*in);
      
      return whiteice::math::log(T(1.0f) + whiteice::math::exp(k*in))/k;
    }
    else if(nonlinearity[layer] == sigmoid){
      // NOTE sigmoid non-linearity give bad values if used with complex varibles
      //      I guess it goes to 1/f(z), f(z)->0 so large numbers appear even if
      //      we regularize the problem!
      
      // non-linearity motivated by restricted boltzman machines..
      T in = input;

      if(abs(in) > abs(T(+20.0f)))
	in = abs(T(+20.0f))*in/abs(in);
      
      T output = T(1.0f) / (T(1.0f) + math::exp(-in));
      return output;
    }
    else if(nonlinearity[layer] == stochasticSigmoid){
      // non-linearity motivated by restricted boltzman machines..
      whiteice::math::blas_complex<double> in, out;
      whiteice::math::convert(in, input);

      if(abs(in) > abs(math::blas_complex<double>(+20.0f)))
	in = abs(math::blas_complex<double>(+20.0f))*in/abs(in);
      
      out = math::blas_complex<double>(1.0f) / (math::blas_complex<double>(1.0f) + math::exp(-in));

      // IS THIS REALLY CORRECT(?)
      const auto rand_real = abs(T(((double)rand())/((double)RAND_MAX)));
      const auto rand_imag = abs(T(((double)rand())/((double)RAND_MAX)));

      whiteice::math::blas_complex<double> value;

      if(abs(math::real(out)) > rand_real){ value.real(1.0f); }
      else{ value.real(0.0f); }
      
      if(abs(math::imag(out)) > rand_imag){ value.imag(1.0f); }
      else{ value.imag(0.0f); }

      T output_value;
      whiteice::math::convert(output_value, value);
      
      return output_value;
    }
    else if(nonlinearity[layer] == tanh){

#ifdef SMALL_RANDOMIZE_INIT
      const T a = T(1.0f);
      const T b = T(1.0f);
#else      
      const T a = T(1.7159f);
      const T b = T(2.0f/3.0f);
#endif
      
      T in = input;
      
      if(abs(in) > abs(T(+10.0f))) in = abs(T(+10.0))*in/abs(in);
      
      const T e2x = whiteice::math::exp(T(2.0f)*b*in);
      const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
      const T output = a*tanhbx;

      return output;
    }
    else if(nonlinearity[layer] == halfLinear){
      // tanh(x) + 0.5x: from a research paper statistically
      // better gradiets for deep neural networks
      {
#ifdef SMALL_RANDOMIZE_INIT	
	const T a = T(1.0f);
	const T b = T(1.0f);
#else
	const T a = T(1.7159f); // suggested by Haykin's neural network book (1999)
	const T b = T(2.0f/3.0f);
#endif

	if(abs(input) > abs(T(+10.0f))){
	  return a*input/abs(input) + T(0.5f)*a*b*input;
	}
	
	// for real valued data
	//if(input > T(10.0)) return a + T(0.5f)*a*b*input;
	//else if(input < T(-10.0)) return -a + T(0.5)*a*b*input;
	
	const T e2x = whiteice::math::exp(T(2.0f)*b*input);
	const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
	const T output = a*tanhbx;
	
	return (output + T(0.5f)*a*b*input);
      }
      
    }
    else if(nonlinearity[layer] == pureLinear){
      return input; // all layers/neurons are linear..
    }
    else if(nonlinearity[layer] == rectifier){
      math::blas_complex<double> out;
      out.real(input.real());
      out.imag(input.imag());
      
      if(input.real() < 0.0f){
	out.real(0.01f*out.real());
      }

      if(input.real() < 0.0f){
	out.imag(0.01f*out.imag());
      }

      return T(out);
    }
    else{
      assert(0);
    }

    return T(0.0);
  }
  
  
  template <typename T> // derivat of non-linearity used in neural network
  inline T nnetwork<T>::Dnonlin(const T& input, unsigned int layer, unsigned int neuron) const 
  {
    assert(layer < getLayers());
    assert(neuron < getNeurons(layer));

    if(dropout.size() > 0){ // drop out is activated
      if(dropout[layer][neuron])
	return T(0.0f); // this neuron is disabled 
    }

    if(nonlinearity[layer] == softmax){
      const T k = T(1.50f);

      T in = input;
      
      if(abs(in) > abs(T(+20.0f)))
	in = abs(T(+20.0f))*in/abs(in);

      const T divider = T(1.0f) + whiteice::math::exp(-k*in);
      
      return T(1.0f)/divider;
    }
	
    else if(nonlinearity[layer] == sigmoid){
      // non-linearity motivated by restricted boltzman machines..
      T in = input;

      if(abs(in) > abs(T(+20.0f))) in = abs(T(+20.0))*in/abs(in);

      T output = T(1.0f) + math::exp(-in);
      output = math::exp(-in) / (output*output);
      return output;
    }
    else if(nonlinearity[layer] == stochasticSigmoid){
      // FIXME: what is "correct" derivate here? I guess we should calculate E{g'(x)} or something..
      // in general stochastic layers should be frozen so that they are optimized
      // through other means than following the gradient..
      T in = input;

      if(abs(in) > abs(T(+20.0f))) in = abs(T(+20.0))*in/abs(in);

      // non-linearity motivated by restricted boltzman machines..
      T output = T(1.0f) + math::exp(-in);
      output = math::exp(-in) / (output*output);
      return output;
    }
    else if(nonlinearity[layer] == tanh){

#ifdef SMALL_RANDOMIZE_INIT
      const T a = T(1.0f);
      const T b = T(1.0f);
#else      
      const T a = T(1.7159f);
      const T b = T(2.0f/3.0f);
#endif
      
      T in = input;

      if(abs(in) > abs(T(+10.0f))) in = abs(T(+10.0))*in/abs(in);

      // for real valued data:
      //if(in > T(+10.0f)) in = T(+10.0);
      //else if(in < T(-10.0f)) in = T(-10.0f);
      
      const T e2x = whiteice::math::exp(T(2.0f)*b*in);
      const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));

      T output = a*b*(T(1.0f) - tanhbx*tanhbx);
      
      return output;
    }
    else if(nonlinearity[layer] == halfLinear){

      // tanh(x) + 0.5x: from a research paper statistically
      // better gradiets for deep neural networks
      {
#ifdef SMALL_RANDOMIZE_INIT
	const T a = T(1.0f);
	const T b = T(1.0f);
#else      
	const T a = T(1.7159f); // suggested by Haykin's neural network book (1999)
	const T b = T(2.0f/3.0f);
#endif
	if(abs(input) > abs(T(+10.0f))) return T(0.0f) + T(0.5f)*a*b;
	
	// for real valued data
	//if(input > T(10.0)) return T(0.0) + T(0.5)*a*b;
	//else if(input < T(-10.0)) return T(0.0) + T(0.5)*a*b;
	
	const T e2x = whiteice::math::exp(T(2.0f)*b*input);
	const T tanhbx = (e2x - T(1.0f)) / (e2x + T(1.0f));
	
	T output = a*b*(T(1.0f) - tanhbx*tanhbx);
	
	return (output + T(0.5f)*a*b);
      }      
    }
    else if(nonlinearity[layer] == pureLinear){
      return T(1.0f); // all layers/neurons are linear..
    }
    else if(nonlinearity[layer] == rectifier){
      math::blas_complex<double> out;
      out.real(1.0f);
      out.imag(1.0f);

      if(input.real() < 0.0f){
	out.real(0.01f);
      }

      if(input.imag() < 0.0f){
	out.imag(0.01f);
      }

      return T(out);
      
#if 0      
      if(input < T(0.0f)){
	// use ReLU (leaky rectifier)
	return T(0.01f);
      }
      else return T(1.0f);
#endif
    }
    else{
      assert(0);
    }

    return T(0.0);
  }

  
  template <typename T>
  inline T nnetwork<T>::inv_nonlin(const T& input, unsigned int layer, unsigned int neuron) const 
  {
    // inverse of non-linearity used
#if 0
    // sinh non-linearity..
    // (sinh()) non-linearity is maybe a bit better non-linearity..
    T output = math::sinh(input); 
#endif
    // THIS DO NOT WORK CURRENTLY
    
    T output = 0.0f;

    assert(0); // there is NO inverse function
    
    return output;
  }
  

  /*
   * calculates gradient of input value GRAD[f(v|w)] while keeping weights w constant
   */
  template <typename T>
  bool nnetwork<T>::gradient_value(const math::vertex<T>& input,
				   math::matrix<T>& grad) const
  {
    const unsigned int L = getLayers();
    
    grad.resize(input_size(), input_size());
    grad.identity();
    
    math::vertex<T> x = input;

    math::matrix<T> W;
    math::vertex<T> b;
    
    for(unsigned int l=0;l<L;l++){
      getWeights(W, l);
      getBias(b, l);
      
      grad = W*grad;
      
      x = W*x + b;

      for(unsigned int j=0;j<grad.ysize();j++){
	for(unsigned int i=0;i<grad.xsize();i++){
	  grad(j,i) *= Dnonlin(x[j], l, j);
	}
      }

      for(unsigned int i=0;i<x.size();i++){
	x[i] = nonlin(x[i], l, i);
      }
      
    }
    
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
#define FNN_FROZEN_CFGSTR           "FNN_FROZEN"
#define FNN_TIMESTAMP_CFGSTR        "FNN_TIMESTAMP"
#define FNN_RETAIN_CFGSTR           "FNN_RETAIN"

  //////////////////////////////////////////////////////////////////////

  template <typename T>
  bool nnetwork<T>::save(const std::string& filename) const
  {
    try{
      whiteice::dataset<T> conf;
      whiteice::math::vertex<T> data;

      // writes version information
      {
	if(conf.createCluster(FNN_VERSION_CFGSTR, 1) == false) return false;
	// version number = float
	data.resize(1);
	data[0] = T(3.000); // version 3.0
	if(conf.add(0, data) == false) return false;
      }

      // writes architecture information
      {
	if(conf.createCluster(FNN_ARCH_CFGSTR, arch.size()) == false) return false;
	data.resize(arch.size());
	for(unsigned int i=0;i<arch.size();i++)
	  data[i] = T((float)arch[i]);
	if(conf.add(1, data) == false) return false;
      }

      // weights: we just convert everything to a big vertex vector and write it
      {
	if(this->exportdata(data) == false) return false;
	
	if(conf.createCluster(FNN_VWEIGHTS_CFGSTR, data.size()) == false) return false;
	if(conf.add(2, data) == false) return false;
      }

      // used non-linearity
      {
	data.resize(this->nonlinearity.size());

	for(unsigned int l=0;l<nonlinearity.size();l++){
	  if(nonlinearity[l] == sigmoid){
	    data[l] = T(0.0);
	  }
	  else if(nonlinearity[l] == stochasticSigmoid){
	    data[l] = T(1.0);
	  }	  
	  else if(nonlinearity[l] == halfLinear){
	    data[l] = T(2.0);
	  }
	  else if(nonlinearity[l] == pureLinear){
	    data[l] = T(3.0);
	  }
	  else if(nonlinearity[l] == tanh){
	    data[l] = T(4.0);
	  }
	  else if(nonlinearity[l] == rectifier){
	    data[l] = T(5.0);
	  }
	  else if(nonlinearity[l] == softmax){
	    data[l] = T(6.0);
	  }
	  else return false; // error!
	}

	if(conf.createCluster(FNN_NONLINEARITY_CFGSTR, data.size()) == false) return false;
	if(conf.add(3, data) == false) return false;
      }

      // frozen status of each layer
      {
	data.resize(this->frozen.size());
	
	for(unsigned int l=0;l<frozen.size();l++){
	  if(frozen[l] == false){
	    data[l] = T(0.0);
	  }
	  else{
	    data[l] = T(1.0);
	  }
	}

	if(conf.createCluster(FNN_FROZEN_CFGSTR, data.size()) == false) return false;
	if(conf.add(4, data) == false) return false;
      }

      // saves retain probability and other parameters
      {
	data.resize(1);
	data[0] = this->retain_probability;

	if(conf.createCluster(FNN_RETAIN_CFGSTR, data.size()) == false) return false;
	if(conf.add(5, data) == false) return false;
      }

      // timestamp
      {
	char buffer[128];
	time_t now = time(0);
	snprintf(buffer, 128, "%s", ctime(&now));
	
	std::string timestamp = buffer;

	if(conf.createCluster(FNN_TIMESTAMP_CFGSTR, timestamp.length()) == false)
	  return false;
	if(conf.add(6, timestamp) == false) return false;
      }

      // don't save dropout or retain probability
      
      return conf.save(filename);
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception "
		<< "File: " << __FILE__ << " "
		<< "Line: " << __LINE__ << " "
		<< e.what() << std::endl;
      
      return false;
    }

    return false;
  } 
  
  
  ///////////////////////////////////////////////////////////////////////////
  

  // load neuralnetwork data from file
  template <typename T>
  bool nnetwork<T>::load(const std::string& filename)
  {
    try{
      whiteice::dataset<T> conf;
      whiteice::math::vertex<T> conf_data;

      // loaded parameters of neural network
      
      std::vector<unsigned int> conf_arch;
      whiteice::math::vertex<T> conf_weights;
      std::vector<nonLinearity> conf_nonlins;
      std::vector<bool> conf_frozen;
      T conf_retain = T(1.0);
      
      if(conf.load(filename) == false) return false;

      // checks version number
      {
	const unsigned int cluster = conf.getCluster(FNN_VERSION_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false;
	if(conf.dimension(cluster) != 1) return false;

	conf_data = conf.access(cluster, 0);
	
	if(conf_data[0] != T(3.000)) // only handles version 3.0 files
	  return false;
      }

      // checks number of clusters (7 in version 3.0 files)
      {
	if(conf.getNumberOfClusters() != 7) return false;
      }

      // gets architecture information
      {
	const int unsigned cluster = conf.getCluster(FNN_ARCH_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) < 2) return false; // bad architecture size

	conf_data = conf.access(cluster, 0);

	if(conf_data.size() < 2) return false; // bad architecture size;

	conf_arch.resize(conf_data.size());

	for(unsigned int i=0;i<conf_arch.size();i++){
	  int value = 0;
	  whiteice::math::convert(value, conf_data[i]);

	  if(value <= 0) return false; // bad data

	  conf_arch[i] = (unsigned int)value;
	}
      }

      // gets weights parameter vector
      {
	const unsigned int cluster = conf.getCluster(FNN_VWEIGHTS_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) < 2) return false; // bad weight size
	
	conf_weights = conf.access(cluster, 0);
      }

      // gets non-linearities
      {
	const unsigned int cluster = conf.getCluster(FNN_NONLINEARITY_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) != conf_arch.size()-1)
	  return false; // number of layers don't match architecture size

	conf_data = conf.access(cluster, 0);

	conf_nonlins.resize(conf_data.size());

	for(unsigned int l=0;l<conf_data.size();l++){
	  if(conf_data[l] == T(0.0)) conf_nonlins[l] = sigmoid;
	  else if(conf_data[l] == T(1.0)) conf_nonlins[l] = stochasticSigmoid;
	  else if(conf_data[l] == T(2.0)) conf_nonlins[l] = halfLinear;
	  else if(conf_data[l] == T(3.0)) conf_nonlins[l] = pureLinear;
	  else if(conf_data[l] == T(4.0)) conf_nonlins[l] = tanh;
	  else if(conf_data[l] == T(5.0)) conf_nonlins[l] = rectifier;
	  else if(conf_data[l] == T(6.0)) conf_nonlins[l] = softmax;
	  else return false; // unknown non-linearity
	}
      }

      // gets frozen status of each layer
      {
	const unsigned int cluster = conf.getCluster(FNN_FROZEN_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) != conf_arch.size()-1)
	  return false; // number of layers don't match architecture size
	
	conf_data = conf.access(cluster, 0);

	conf_frozen.resize(conf_data.size());

	for(unsigned int l=0;l<conf_frozen.size();l++){
	  if(conf_data[l] == T(0.0)) conf_frozen[l] = false;
	  else if(conf_data[l] == T(1.0)) conf_frozen[l] = true;
	  else return false; // unknown value
	}
      }

      // gets misc parameters (retain_probability)
      {
	const unsigned int cluster = conf.getCluster(FNN_RETAIN_CFGSTR);
	if(cluster >= conf.getNumberOfClusters()) return false;
	if(conf.size(cluster) != 1) return false; 
	if(conf.dimension(cluster) != 1)
	  return false; // only single parameter saved
	
	conf_data = conf.access(cluster, 0);

	if(conf_data.size() != 1) return false;

	conf_retain = conf_data[0];

	if(abs(conf_retain) < abs(T(0.0)) || abs(conf_retain) > abs(T(1.0)))
	  return false; // correct interval for data
      }

      // don't check timestamp metainformation

      // parameters where successfully loaded from the disk.
      // now calculates all the parameters and set them as neural network values
      {
	this->arch = conf_arch;

	unsigned int memuse = 0;
	maxwidth = arch[0];
	
	unsigned int i = 1;
	while(i < arch.size()){
	  memuse += (arch[i-1] + 1)*arch[i];
	  
	  if(arch[i] > maxwidth)
	    maxwidth = arch[i];
	  i++;
	}

	this->data.resize(memuse);

	state.resize(maxwidth);
	temp.resize(maxwidth);
	lgrad.resize(maxwidth);

	hasValidBPData = false;
	bpdata.resize(1);
	size = memuse;
	
	inputValues.resize(arch[0]);
	outputValues.resize(arch[arch.size() - 1]);

	this->frozen = conf_frozen;
	this->nonlinearity = conf_nonlins;
	this->retain_probability = conf_retain;
	
	if(this->importdata(conf_weights) == false) // this should never fail
	  return false;
	
	dropout.clear(); // dropout is disabled in saved networks
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
  bool nnetwork<T>::exportdata(math::vertex<T>& v) const {
    v.resize(size);
    
    // NN exports TO vertex, vertex imports FROM NN
    return v.importData(&(data[0]), size, 0);
  }
  
  template <typename T>
  bool nnetwork<T>::importdata(const math::vertex<T>& v) {
    if(v.size() != size)
      return false;
    
    // nn imports FROM vertex, vertex exports TO network
    if(v.exportData(&(data[0]), size, 0) == false)
      return false;

    const bool verbose = false;

    // "safebox" (keeps data always within sane levels)
    for(unsigned int i=0;i<data.size();i++){
      if(whiteice::math::isnan(data[i])){
	if(verbose)
	  std::cout << "nnetwork::importdata() warning. bad NaN data: " << data[i] << std::endl;
	data[i] = T(0.0f);
      }
      
      if(whiteice::math::isinf(data[i])){
	if(verbose)
	  std::cout << "nnetwork::importdata() warning. bad Inf data: " << data[i] << std::endl;
	data[i] = T(0.0f);
      }

      if(data[i].real() > 10000.0f){
	if(verbose)
	  std::cout << "nnetwork::importdata() warning. bad real data: " << data[i] << std::endl;
	data[i].real(+10000.0f);	
      }
      else if(data[i].real() < -10000.0f){
	if(verbose)
	  std::cout << "nnetwork::importdata() warning. bad real data: " << data[i] << std::endl;
	data[i].real(-10000.0f);
      }

      if(data[i].imag() > 10000.0f){
	if(verbose)
	  std::cout << "nnetwork::importdata() warning. bad imag data: " << data[i] << std::endl;
	data[i].imag(+10000.0f);
      }
      else if(data[i].imag() < -10000.0f){
	if(verbose)
	  std::cout << "nnetwork::importdata() warning. bad imag data: " << data[i] << std::endl;
	data[i].imag(-10000.0f);
      }
      
    }
    
    return true;
  }
  
  
  // number of dimensions used by import/export
  template <typename T>
  unsigned int nnetwork<T>::exportdatasize() const {
    return size;
  }

  ////////////////////////////////////////////////////////////////////////////
  
  template <typename T>
  unsigned int nnetwork<T>::getLayers() const {
    return (arch.size()-1); 
  }

  // number of neurons per layer
  template <typename T>
  unsigned int nnetwork<T>::getNeurons(unsigned int layer) const 
  {
    if(layer+1 >= arch.size()) return 0;
    return arch[layer+1];
  }

  // number of neurons per layer
  template <typename T>
  unsigned int nnetwork<T>::getInputs(unsigned int layer) const 
  {
    if(layer >= arch.size()-1) return 0;
    return arch[layer];
  }

  template <typename T>
  bool nnetwork<T>::getBias(math::vertex<T>& b, unsigned int layer) const 
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
  bool nnetwork<T>::setBias(const math::vertex<T>& b, unsigned int layer) 
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
  bool nnetwork<T>::getWeights(math::matrix<T>& w, unsigned int layer) const 
  {
    if(layer+1 >= arch.size()) return false;

    w.resize(arch[layer+1], arch[layer]);

    const T* dptr = &(data[0]);

    for(unsigned int i=0;i<layer;i++){
      dptr = dptr + arch[i]*arch[i+1]; // weight matrix w for layer i
      dptr = dptr + arch[i+1]; // bias b for layer i
    }

    for(unsigned int j=0;j<arch[layer+1];j++)
      for(unsigned int i=0;i<arch[layer];i++)
	w(j,i) = dptr[j*arch[layer] + i];

    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::setWeights(const math::matrix<T>& w, unsigned int layer) 
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
  bool nnetwork<T>::setNonlinearity(nnetwork<T>::nonLinearity nl)
  {
    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>)){
      if(nl == rectifier || nl == sigmoid){
	printf("nnetwork::setNonlinearity() warning: chosen non-linearity don't work even with regularized complex-valued neural network.");
      }
    }
    
    for(unsigned int l=0;l<nonlinearity.size();l++)
      nonlinearity[l] = nl;

    // last layer is always linear!! (for now)
    // nonlinearity[nonlinearity.size()-1] = pureLinear; 
    
    return true;
  }

  template <typename T>
  typename nnetwork<T>::nonLinearity nnetwork<T>::getNonlinearity(unsigned int layer) const 
  {
    if(layer >= nonlinearity.size()) return pureLinear; // silent failure..

    return nonlinearity[layer];
  }

  template <typename T>
  bool nnetwork<T>::setNonlinearity(unsigned int layer, nonLinearity nl)
  {
    if(layer >= nonlinearity.size()) return false;
    
    if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
       typeid(T) == typeid(whiteice::math::blas_complex<double>)){
      if(nl == rectifier || nl == sigmoid){
	printf("nnetwork::setNonlinearity() warning: chosen non-linearity don't work even with regularized complex-valued neural network.");
      }
    }

    nonlinearity[layer] = nl;
    
    return true;
  }
  
  template <typename T>
  void nnetwork<T>::getNonlinearity(std::vector<nonLinearity>& nls) const 
  {
    nls = nonlinearity;
  }

  template <typename T>
  bool nnetwork<T>::setNonlinearity(const std::vector<nonLinearity>& nls) 
  {
    if(nonlinearity.size() != nls.size()) return false;

    for(const auto& nl : nls){
      if(typeid(T) == typeid(whiteice::math::blas_complex<float>) ||
	 typeid(T) == typeid(whiteice::math::blas_complex<double>)){
	if(nl == rectifier || nl == sigmoid){
	  printf("nnetwork::setNonlinearity() warning: chosen non-linearity don't work even with regularized complex-valued neural network.");
	}
      }
    }

    nonlinearity = nls;
    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::setFrozen(unsigned int layer, bool f)
  {
    if(layer >= frozen.size())
      return false;

    frozen[layer] = f;
    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::setFrozen(const std::vector<bool>& f)
  {
    if(this->frozen.size() != f.size()) return false;

    this->frozen = f;
    
    return true;
  }

  
  template <typename T>
  bool nnetwork<T>::getFrozen(unsigned int layer) const
  {
    if(layer < frozen.size()) return frozen[layer];
    else return false; // silent error
  }

  
  template <typename T>
  void nnetwork<T>::getFrozen(std::vector<bool>& frozen) const
  {
    frozen = this->frozen;
  }

  // creates subnet starting from fromLayer:th layer to the output
  template <typename T>
  nnetwork<T>* nnetwork<T>::createSubnet(const unsigned int fromLayer)
  {
    if(fromLayer >= getLayers()) return nullptr;

    std::vector<unsigned int> a;

    for(unsigned int i=fromLayer;i<arch.size();i++)
      a.push_back(arch[i]);

    nnetwork<T>* nn = new nnetwork<T>(a);

    math::matrix<T> W;
    math::vertex<T> b;

    // sets parameters of the network
    for(unsigned int i=fromLayer;i<(arch.size()-1);i++){
      nn->frozen[i-fromLayer] = this->frozen[i];
      nn->nonlinearity[i-fromLayer] = this->nonlinearity[i];
      
      this->getWeights(W, i);
      nn->setWeights(W, i - fromLayer);
      
      this->getBias(b, i);
      nn->setBias(b, i - fromLayer);
    }

    return nn;
  }

  // creates subnet starting from fromLayer:th layer to the toLayer
  template <typename T>
  nnetwork<T>* nnetwork<T>::createSubnet(const unsigned int fromLayer,
					 const unsigned int toLayer)
  {
    if(fromLayer >= getLayers()) return nullptr;
    if(toLayer >= arch.size() || fromLayer >= toLayer) return nullptr;

    std::vector<unsigned int> a;

    for(unsigned int i=fromLayer;i<=(toLayer+1);i++)
      a.push_back(arch[i]);

    nnetwork<T>* nn = new nnetwork<T>(a);

    math::matrix<T> W;
    math::vertex<T> b;

    // sets parameters of the network
    for(unsigned int i=fromLayer;i<=toLayer;i++){
      nn->frozen[i-fromLayer] = this->frozen[i];
      nn->nonlinearity[i-fromLayer] = this->nonlinearity[i];
      
      this->getWeights(W, i);
      nn->setWeights(W, i - fromLayer);
      
      this->getBias(b, i);
      nn->setBias(b, i - fromLayer);
    }

    return nn;
  }

  // injects (if possible) subnet into net starting from fromLayer:th layer
  template <typename T>
  bool nnetwork<T>::injectSubnet(const unsigned int fromLayer, nnetwork<T>* nn)
  {
    if(nn == nullptr) return false;

    // check if architecture matches exactly
    if(this->arch.size() != nn->arch.size() + fromLayer)
      return false;

    for(unsigned int i=0;i<nn->arch.size();i++){
      if(arch[fromLayer+i] != nn->arch[i])
	return false;
    }

    for(unsigned int l=0;l<nn->getLayers();l++){
      if(nonlinearity[fromLayer+l] != nn->nonlinearity[l])
	return false;
    }

    math::matrix<T> W;
    math::vertex<T> b;
    
    for(unsigned int i=0;i<nn->getLayers();i++){
      nn->getBias(b, i);
      this->setBias(b, fromLayer + i);

      nn->getWeights(W, i);
      this->setWeights(W, fromLayer + i);
    }

    return true;
  }
  
  
  template <typename T>
  unsigned int nnetwork<T>::getSamplesCollected() const 
  {
    if(samples.size() > 0)
      return samples[0].size();
    else
      return 0;
  }
  
  template <typename T>
  bool nnetwork<T>::getSamples(std::vector< math::vertex<T> >& samples, unsigned int layer, unsigned int MAXSAMPLES) const 
  {
    if(layer >= this->samples.size()) return false;

    if(MAXSAMPLES == 0 || MAXSAMPLES >= this->samples[layer].size()){
      samples = this->samples[layer];
      return true;
    }

    for(unsigned int i=0;i<MAXSAMPLES;i++){
      const unsigned int index = rng.rand() % this->samples[layer].size();
      samples.push_back(this->samples[layer][index]);
    }
    
    return true;
  }
  
  template <typename T>
  void nnetwork<T>::clearSamples() 
  {
    for(auto& s : samples)
      s.clear();
  }

  template <typename T>
  bool nnetwork<T>::setDropOut(T probability) 
  {
    if(abs(probability) <= abs(T(0.0)) || abs(probability) > abs(T(1.0)))
      return false; // we cannot set all neurons to be dropout neurons

    retain_probability = probability;
    dropout.resize(getLayers());
    
    for(unsigned int l=0;l<dropout.size();l++){
      dropout[l].resize(getNeurons(l));
      if(l != (dropout.size()-1)){
	unsigned int numdropped = 0;

	for(unsigned int i=0;i<dropout[l].size();i++){
	  if(abs(rng.uniform()) > abs(retain_probability)){
	    dropout[l][i] = true;
	    numdropped++;
	  }
	  else{
	    dropout[l][i] = false;
	  }
	}

	// if we are about to drop all nodes we always
	// randomly keep at least one node
	if(numdropped == dropout[l].size()){
	  const unsigned int index = rng.rand() % dropout[l].size();
	  dropout[l][index] = false;
	}
      }
      else{ // we always keep all last layer nodes
	for(unsigned int i=0;i<dropout[l].size();i++){
	  dropout[l][i] = false;
	}
      }
    }

    return true;
  }

  template <typename T>
  bool nnetwork<T>::removeDropOut(T probability) 
  {    
    // scales weights according to retain_probability
    // (except the first layer where we always keep all inputs)

    for(unsigned int l=1;l<getLayers();l++){
      math::matrix<T> w;
      getWeights(w, l);
      w = probability*w;
      setWeights(w, l);
    }

    dropout.clear();

    retain_probability = T(1.0);

    return true;
  }

  // remove all drop-out
  template <typename T>
  void nnetwork<T>::clearDropOut() 
  {
    retain_probability = T(1.0);
    dropout.clear();
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
    
    memcpy((T*)y, (const T*)&(temp[0]), yd*sizeof(T));
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
      memcpy((T*)temp.data(), (T*)b, yd*sizeof(T));
      
      cblas_sgemv(CblasRowMajor, CblasNoTrans, yd, xd,
		  1.0f, (float*)W, xd, (float*)x, 1, 
		  1.0f, (float*)temp.data(), 1);
      
      memcpy((T*)y, (const T*)temp.data(), yd*sizeof(T));
    }
    else if(typeid(T) == typeid(whiteice::math::blas_real<double>)){
      memcpy((T*)temp.data(), (const T*)b, yd*sizeof(T));
      
      cblas_dgemv(CblasRowMajor, CblasNoTrans, yd, xd,
		  1.0f, (double*)W, xd, (double*)x, 1, 
		  1.0f, (double*)temp.data(), 1);
      
      memcpy((T*)y, (const T*)temp.data(), yd*sizeof(T));
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
      
      memcpy((T*)y, (const T*)temp.data(), yd*sizeof(T));
    }

  }
  
  
  ////////////////////////////////////////////////////////////
  // can be used to decrease memory usage
  
#if 0
  // changes NN to compressed form of operation or
  // back to normal non-compressed form
  template <typename T>  
  bool nnetwork<T>::compress() {
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
  bool nnetwork<T>::decompress() {
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
  bool nnetwork<T>::iscompressed() const {
    return compressed;
  }
  
  
  // returns compression ratio: compressed/orig
  template <typename T>
  float nnetwork<T>::ratio() const {
    if(compressed) return compressor->ratio();
    else return 1.0f;
  }
  
#endif
  
  
  //////////////////////////////////////////////////////////////////////
  
  //template class nnetwork< float >;
  //template class nnetwork< double >;
  
  template class nnetwork< math::blas_real<float> >;
  template class nnetwork< math::blas_real<double> >;

  template class nnetwork< math::blas_complex<float> >;
  template class nnetwork< math::blas_complex<double> >;
  
};
