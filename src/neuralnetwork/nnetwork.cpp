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
#include "atlas.h"


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
    
    for(unsigned int i=0;i<data.size();i++)
      data[i] = T( 1.0f*(((float)rand())/((float)RAND_MAX)) - 0.5f );

    state.resize(maxwidth);
    temp.resize(maxwidth);
    
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

    arch = nn.arch;
    bpdata = nn.bpdata;
    data = nn.data;
    state = nn.state;
    temp = nn.temp;
    
  }
  
  
  template <typename T>
  nnetwork<T>::nnetwork(const std::vector<unsigned int>& nnarch) 
    throw(std::invalid_argument)
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
    
    unsigned int memuse = 0;
    
    for(unsigned int i=0;i<arch.size();i++){
      if(i > 0) 
	memuse += (arch[i-1] + 1)*arch[i];
    }
    
    size = memuse;

    data.resize(size);
    
    // intializes all layers (randomly)
    
    for(unsigned int i=0;i<memuse;i++)
      data[i] = T( 1.0*(((float)rand())/((float)RAND_MAX)) - 0.5f );

    state.resize(maxwidth);
    temp.resize(maxwidth);
     
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

    data = nn.data;
    bpdata = nn.bpdata;

    state = nn.state;
    temp = nn.temp;
    
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
  bool nnetwork<T>::calculate(bool gradInfo)
  {
    const T af = T(1.7159f);
    const T bf = T(0.6666f);
    
    if(!inputValues.exportData(&(state[0]))) return false;
    
    
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

	// gemv(a[1], a[0], dptr, state, state); // s = W*s
	// gvadd(a[1], state, dptr + a[0]*a[1]); // s += b;

	// s = b + W*s
	gemv_gvadd(arch[aindex+1], arch[aindex], dptr, &(state[0]), &(state[0]),
		   arch[aindex+1], &(state[0]), dptr + arch[aindex]*arch[aindex+1]); // s += b;
	
	// s = g(v)
	
	if(aindex+2 < arch.size()){ // not the last layer
	  // f(x) = b * ( (1 - Exp[-ax]) / (1 + Exp[-ax]) )
	  //      = b * ( 2 / (1 + Exp[-ax]) - 1)
	  
	  for(unsigned int i=0;i<arch[aindex+1];i++){
	    T expbx = math::exp(-bf*state[i]);
	    // state[i] = af * ( (T(1.0f) - expbx) / (T(1.0f) + expbx) );
	    state[i] = af * ( T(2.0f) / (T(1.0f) + expbx) - T(1.0f) ); // numerically more stable (no NaNs)
	  }
	}
	
	
	dptr += (arch[aindex] + 1)*arch[aindex+1];

	aindex++; // next layer
      }
            
    }
    else{
      T* dptr = &(data[0]);
      
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
	    T expbx = math::exp(-bf*state[i]);
	    // state[i] = af * ( (T(1.0f) - expbx) / (T(1.0f) + expbx) );
	    state[i] = af * ( T(2.0f) / (T(1.0f) + expbx) - T(1.0f) ); // numerically more stable (no NaNs)
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
  
  
  template <typename T> // number of layers
  unsigned int nnetwork<T>::length() const {
    return arch.size();
  }
  
  
  template <typename T>
  bool nnetwork<T>::randomize(){
    
    for(unsigned int i=0;i<data.size();i++)
      data[i] = T( 1.0f*(((float)rand())/((float)RAND_MAX)) - 0.5f );
    
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
    // updates weights with backpropagation
    // this is cut'n'paste + small changes from backprop()
    
    const T af = T(1.7159f);
    const T bf = T(0.6666f);
        
    if(!hasValidBPData)
      return false;
    
    T* lgrad = (T*)malloc(maxwidth * sizeof(T));
    T* temp  = (T*)malloc(maxwidth * sizeof(T));
  
    if(lgrad == 0 || temp == 0){
      if(lgrad) free(lgrad);
      if(temp) free(temp);
      return false;
    }
  
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
    
    if(!error.exportData(lgrad)){
      free(lgrad);
      free(temp);
      return false;
    }
    
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
    unsigned gindex = grad.size();
    
    
    while(counter > 1){
      // updates W and b in this layer
      
      // W += lrate * (lgrad * input^T)
      // b += lrate * lgrad;
      
      const unsigned int rows = *a;
      const unsigned int cols = *(a - 1);
      
      _bpdata -= *(a - 1);
      _data -= (*a) * (*(a - 1)) + *a;
      gindex -= (*a) * (*(a - 1)) + *a;
      const T* dptr = _data;
      
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
      
      // calculates next lgrad
      
      // for hidden layers: local gradient is:
      // grad[n] = diag(..g'(v[i])..)*(W^t * grad[n+1])

      // FIXME: THERE IS A BUG IN BPTR handling???
      const T* bptr = _bpdata;
    
      for(unsigned int x=0;x<cols;x++){
	T sum = T(0.0f);
	for(unsigned int y=0;y<rows;y++)
	  sum += lgrad[y]*_data[x + y*cols];
	
	// fast calculation of g'(v) for logistic/sigmoidal non-lins
	// is g'(v) = g(v)*(1-g(v)) , if a = 1, for a != 1:
	// is g'(v) = a*g(v)*(1-g(v)) = a*y*(1-y)
	// in practice:
	//
	// f'(x) = (0.5*a*b) * ( 1 + f(x)/a ) * ( 1 - f(x)/a ) 
	
	T fxa = (*bptr)/af;
	sum *= (T(0.50f)*af*bf) * ((T(1.0f) + fxa)*(T(1.0f) - fxa));
	
	temp[x] = sum;
	bptr++; // BUG HERE???
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
      // updates the first layer's W and b 
      // 
      // W += rate * (lgrad * input^T)
      // b += rate * lgrad;
      
      const unsigned int rows = *a;
      const unsigned int cols = *(a - 1);
      
      _bpdata -= *(a - 1);
      _data -= (*a) * (*(a - 1)) + *a;
      gindex -= (*a) * (*(a - 1)) + *a;
      
      assert(_bpdata == &(bpdata[0]));
      assert(_data == &(data[0]));
      assert(gindex == 0);
      
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
    
    
    free(lgrad);
    free(temp);
    
    
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
  

  // load & saves neuralnetwork data from file
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
	maxwidth = 0;
	
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

  
  //////////////////////////////////////////////////////////////////////  
  




#undef FNN_LAYER_AFUN_PARAM_CFGSTR
#undef FNN_VERSION_CFGSTR
#undef FNN_VWEIGHTS_CFGSTR
#undef FNN_LAYER_W_CFGSTR
#undef FNN_LAYER_B_CFGSTR
#undef FNN_MOMENS_CFGSTR
#undef FNN_LRATES_CFGSTR
#undef FNN_AFUNS_CFGSTR
#undef FNN_ARCH_CFGSTR
  
  
  //////////////////////////////////////////////////////////////////////
  
  
  // exports and imports neural network parameters to/from vertex
  template <typename T>
  bool nnetwork<T>::exportdata(math::vertex<T>& v) const throw(){
    if(v.size() < size)
      v.resize(size);
    
    // NN exports TO vertex, vertex imports FROM NN
    return v.importData(&(data[0]), size, 0);
  }
  
  template <typename T>
  bool nnetwork<T>::importdata(const math::vertex<T>& v) throw(){
    if(v.size() < size)
      return false;
    
    // nn imports FROM vertex, vertex exports TO network
    return v.exportData(&(data[0]), size, 0);
  }
  
  
  // number of dimensions used by import/export
  template <typename T>
  unsigned int nnetwork<T>::exportdatasize() const throw(){
    return size;
  }
  
  
  
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
				      T* W, T* x, T* y,
				      unsigned int dim, T* s, T* b)
  {
    // uses temporary space to handle correctly
    // the case when x and y vectors overlap (or are same)
    // T temp[yd]; [we have global temp to store results]
    
    for(unsigned int j=0;j<yd;j++){
      T sum = b[j];
      for(unsigned int i=0;i<xd;i++)
	sum += W[i + j*xd]*x[i];
      
      temp[j] = sum;
    }
    
    memcpy(y, &(temp[0]), yd*sizeof(T));
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
  template class nnetwork< math::atlas_real<float> >;
  template class nnetwork< math::atlas_real<double> >;
  
};
