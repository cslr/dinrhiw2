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
    data = NULL;
    arch = NULL;
    bpdata = NULL;
    maxwidth = 1;
    archLength = 2;
    flags = 0;
    size = 2;
    
    arch = (unsigned int*)malloc((archLength+1)*sizeof(unsigned int));
    if(arch == 0)
      throw std::bad_alloc();
    
    arch[0] = 1;
    arch[1] = 1;
    arch[2] = 0;
    
    data = (T*)malloc(size*sizeof(T));
    if(data == 0){
      free(arch);
      throw std::bad_alloc();
    }
    
    for(unsigned int i=0;i<size;i++)
      data[i] = T( 1.0f*(((float)rand())/((float)RAND_MAX)) - 0.5f );
    
    state = (T*)malloc(maxwidth*sizeof(T));
    temp  = (T*)malloc(maxwidth*sizeof(T));
    if(state == 0 || temp == 0){
      if(state) free(state);
      if(temp) free(temp);
      if(data) free(data);
      if(arch) free(arch);
      throw std::bad_alloc();
    }
    
    
    rate = T(0.01f);
    
    
    inputValues.resize(1);
    outputValues.resize(1);
    
    compressed = false;
    compressor = 0;
    hasValidBPData = false;
  }
  
  
  template <typename T>
  nnetwork<T>::nnetwork(const nnetwork<T>& nn)
  {
    if(nn.compressed) // FIX THIS (decompress, copy, compress both)
      throw std::invalid_argument("Cannot copy compressed network.");

    data = NULL;
    arch = NULL;
    bpdata = NULL;
    
    inputValues.resize(nn.inputValues.size());
    outputValues.resize(nn.outputValues.size());
    
    arch = (unsigned int*)malloc((nn.archLength+1)*sizeof(unsigned int));
    if(arch == 0){
      printf("arch allocation failed\n");
      throw std::bad_alloc();
    }
    
    memcpy(arch, nn.arch, (nn.archLength+1)*sizeof(unsigned int));
    maxwidth = nn.maxwidth;
    
    if(nn.flags){
      unsigned int counter = 0;
      while(nn.flags[counter] != 0) counter++;
      flags = (unsigned int*)malloc((counter + 1) * sizeof(unsigned int));
      
      if(flags == 0){
	free(arch);
	printf("flags allocation failed\n");
	throw std::bad_alloc();
      }
      
      memcpy(flags, nn.flags, (counter + 1) * sizeof(unsigned int));
    }
    else flags = 0;
    
    
    if(nn.bpdata){
      // allocates bpdata for each layer
      // input x (prev layers y) for each layer
    
      unsigned int bpsize = 0;
      unsigned int* tmp = nn.arch;
      while(tmp[1] != 0){
	bpsize += tmp[0];
	tmp++;
      }
      
      bpdata = (T*)malloc(bpsize*sizeof(T));
      if(bpdata == 0){
	printf("bpdata allocation failed\n");
	if(arch) free(arch);
	if(flags) free(flags);
	throw std::bad_alloc();
      }
      
      memcpy(bpdata, nn.bpdata, bpsize*sizeof(T));
    }
    else bpdata = 0;
    
    
    hasValidBPData = nn.hasValidBPData;
    archLength = nn.archLength;
    size = nn.size;
    rate = nn.rate;
    
    data = (T*)malloc(size*sizeof(T));
    if(data == 0){
      if(arch) free(arch);
      if(flags) free(flags);
      if(bpdata) free(bpdata);

      printf("data allocation failed\n");
      
      throw std::bad_alloc();
    }
    
    memcpy(data, nn.data, size*sizeof(T));
    
    state = (T*)malloc(nn.maxwidth*sizeof(T));
    temp  = (T*)malloc(nn.maxwidth*sizeof(T));

    maxwidth = nn.maxwidth;
    
    if(state == 0 || temp == 0){
      if(state) free(state);
      if(temp) free(temp);
      if(data) free(data);
      if(arch) free(arch);
      if(flags) free(flags);
      if(bpdata) free(bpdata);

      printf("state or temp allocation failed\n");
      
      throw std::bad_alloc();
    }
    
    
    compressed = false;
    compressor = 0;
  }
  
  
  template <typename T> // L*W rectangle network
  nnetwork<T>::nnetwork(const unsigned int layers, 
			const unsigned int width){
    if(width < 1)
      throw std::invalid_argument("invalid network architecture");
    
    if(layers <= 0)
      throw std::invalid_argument("invalid network architecture");

    data = NULL;
    arch = NULL;
    bpdata = NULL;
    
    inputValues.resize(width);
    outputValues.resize(width);
    
    // sets up architecture
    
    arch = (unsigned int*)malloc((layers+2)*sizeof(unsigned int));
    if(arch == 0)
      throw std::bad_alloc();
    
    maxwidth = width;
    unsigned int memuse = 0;
    
    for(unsigned int i=0;i<(layers+1);i++){
      arch[i] = width;
      
      if(i > 0) 
	memuse += (width + 1)*width;
    }
    
    arch[layers+1] = 0;
    archLength = layers + 1;
    flags = 0; // no flags
    size = memuse;
    rate = T(0.01f);
    
    data = (T*)malloc(memuse*sizeof(T));
    if(data == 0){
      free(arch);
      throw std::bad_alloc();
    }
    
    // intializes all layers
    
    for(unsigned int i=0;i<memuse;i++)
      data[i] = T( 1.0f*(((float)rand())/((float)RAND_MAX)) - 0.5f );
    
    state = (T*)malloc(maxwidth*sizeof(T));
    temp = (T*)malloc(maxwidth*sizeof(T));
    if(state == 0 || temp == 0){
      if(state) free(state);
      if(temp) free(temp);
      if(data) free(data);
      if(arch) free(arch);
      throw std::bad_alloc();
    }
    
    inputValues.resize(arch[0]);
    outputValues.resize(arch[archLength - 1]);
    
    
    flags = 0;
    bpdata = 0;
    
    hasValidBPData = false;
    compressed = false;
    compressor = 0;
  }
  
  
  template <typename T>
  nnetwork<T>::nnetwork(const std::vector<unsigned int>& nnstruc,
			bool compressed_network) 
    throw(std::invalid_argument)
  {
    if(nnstruc.size() < 2)
      throw std::invalid_argument("invalid network architecture");
    
    for(unsigned int i=0;i<nnstruc.size();i++)
      if(nnstruc[i] <= 0)
	throw std::invalid_argument("invalid network architecture");

    data = NULL;
    bpdata = NULL;
    arch = NULL;
   
    
    // sets up architecture
    
    arch = (unsigned int*)malloc((nnstruc.size()+1)*sizeof(unsigned int));
    if(arch == 0)
      throw std::bad_alloc();
    
    maxwidth = 0;
    unsigned int memuse = 0;
    
    for(unsigned int i=0;i<nnstruc.size();i++){
      arch[i] = nnstruc[i];
      
      if(i > 0) 
	memuse += (arch[i-1] + 1)*arch[i];
      
      if(nnstruc[i] > maxwidth)
	maxwidth = nnstruc[i];
    }
    
    arch[nnstruc.size()] = 0;
    archLength = nnstruc.size();
    flags = 0; // no flags
    size = memuse;
    
    data = (T*)malloc(memuse*sizeof(T));
    if(data == 0){
      free(arch);
      throw std::bad_alloc();
    }
    
    // intializes all layers (randomly)
    
    for(unsigned int i=0;i<memuse;i++)
      data[i] = T( 1.0*(((float)rand())/((float)RAND_MAX)) - 0.5f );
    
    state = (T*)malloc(maxwidth*sizeof(T));
    temp  = (T*)malloc(maxwidth*sizeof(T));
    if(state == 0 || temp == 0){
      if(state) free(state);
      if(temp) free(temp);
      if(data) free(data);
      if(arch) free(arch);
      throw std::bad_alloc();
    }
    
    
    rate = T(0.01f);
    
     
    inputValues.resize(arch[0]);
    outputValues.resize(arch[archLength - 1]);
   
    bpdata = 0;
    hasValidBPData = false;
    compressed = false;
    compressor = 0;
  }
  
  
  template <typename T>
  nnetwork<T>::~nnetwork()
  {
    if(compressor) free(compressor);
    if(bpdata) free(bpdata);
    if(flags) free(flags);
    if(arch) free(arch);
    if(data) free(data);
    if(state) free(state);
    if(temp) free(temp);
  }
  
  ////////////////////////////////////////////////////////////

  // returns input and output dimensions of neural network
  template <typename T>
  unsigned int nnetwork<T>::input_size() const throw(){
    return arch[0];
  }
  
  template <typename T>
  unsigned int nnetwork<T>::output_size() const throw(){
    return arch[archLength-1];
  }


  template <typename T>
  void nnetwork<T>::getArchitecture(std::vector<unsigned int>& nn_arch) const
  {
    nn_arch.resize(archLength);
    for(unsigned int i=0;i<archLength;i++){
      nn_arch[i] = this->arch[i];
    }
  }


  template <typename T>
  nnetwork<T>& nnetwork<T>::operator=(const nnetwork<T>& nn)
  {
    if(nn.compressed) // FIX THIS (decompress, copy, compress both)
      throw std::invalid_argument("Cannot copy compressed network.");

    printf("NN = operator\n");

    unsigned int* new_arch = 0;
    unsigned int new_maxwidth = 0;
    unsigned int new_archLength = 0;
    
    unsigned int* new_flags = 0;
    unsigned int new_size = 0;
    bool new_hasValidBPData = false;
    
    T* new_state = 0;
    T* new_data = 0;
    T* new_bpdata = 0;
    T* new_temp = 0;
    
    T new_rate = T(0.0);
    
    new_arch = (unsigned int*)malloc((nn.archLength+1)*sizeof(unsigned int));
    if(new_arch == 0){
      printf("arch allocation failed\n");
      throw std::bad_alloc();
    }
    
    memcpy(new_arch, nn.arch, (nn.archLength+1)*sizeof(unsigned int));
    new_maxwidth = nn.maxwidth;
    
    if(nn.flags){
      unsigned int counter = 0;
      while(nn.flags[counter] != 0) counter++;
      new_flags = (unsigned int*)malloc((counter + 1) * sizeof(unsigned int));
      
      if(new_flags == 0){
	free(new_arch);
	printf("flags allocation failed\n");
	throw std::bad_alloc();
      }
      
      memcpy(new_flags, nn.flags, (counter + 1) * sizeof(unsigned int));
    }
    else new_flags = 0;
    
    
    if(nn.bpdata){
      // allocates bpdata for each layer
      // input x (prev layers y) for each layer
    
      unsigned int bpsize = 0;
      unsigned int* tmp = nn.arch;
      while(tmp[1] != 0){
	bpsize += tmp[0];
	tmp++;
      }
      
      new_bpdata = (T*)malloc(bpsize*sizeof(T));
      if(new_bpdata == 0){
	printf("bpdata allocation failed\n");
	if(new_arch) free(new_arch);
	if(new_flags) free(new_flags);
	throw std::bad_alloc();
      }
      
      memcpy(new_bpdata, nn.bpdata, bpsize*sizeof(T));
    }
    else new_bpdata = 0;
    
    
    new_hasValidBPData = nn.hasValidBPData;
    new_archLength = nn.archLength;
    new_size = nn.size;
    new_rate = nn.rate;
    
    new_data = (T*)malloc(size*sizeof(T));
    if(new_data == 0){
      if(new_arch) free(new_arch);
      if(new_flags) free(new_flags);
      if(new_bpdata) free(new_bpdata);

      printf("data allocation failed\n");
      
      throw std::bad_alloc();
    }
    
    memcpy(new_data, nn.data, size*sizeof(T));
    
    new_state = (T*)malloc(nn.maxwidth*sizeof(T));
    new_temp  = (T*)malloc(nn.maxwidth*sizeof(T));

    new_maxwidth = nn.maxwidth;
    
    if(new_state == 0 || new_temp == 0){
      if(new_state) free(new_state);
      if(new_temp) free(new_temp);
      if(new_data) free(new_data);
      if(new_arch) free(new_arch);
      if(new_flags) free(new_flags);
      if(new_bpdata) free(new_bpdata);

      printf("state or temp allocation failed\n");
      
      throw std::bad_alloc();
    }
    
    
    ////////////////////////////////////////////////

    if(state) free(state);
    if(temp) free(temp);
    if(data) free(data);
    if(arch) free(arch);
    if(flags) free(flags);
    if(bpdata) free(bpdata);

    arch = new_arch;
    maxwidth = new_maxwidth;
    archLength = new_archLength;
    
    flags = new_flags;
    size = new_size;
    
    state = new_state;
    data = new_data;
    bpdata = new_bpdata;
    temp = new_temp;
    
    rate = new_rate;
    hasValidBPData = new_hasValidBPData;
    
    ////////////////////////////////////////////////

    compressed = false;
    compressor = 0;
    
    inputValues.resize(nn.inputValues.size());
    outputValues.resize(nn.outputValues.size());
    

    
    return (*this);
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
    
    if(compressed) return false;
    if(!inputValues.exportData(state)) return false;

    
    unsigned int* a = arch;
    
    
    if(gradInfo){ // saves bpdata
      
      if(bpdata == 0){
	unsigned int bpsize = 0;
	unsigned int* tmp = arch;
	while(tmp[1] != 0){
	  bpsize += tmp[0];
	  tmp++;
	}
	
	bpdata = (T*)malloc(bpsize*sizeof(T));
	if(bpdata == 0){
	  std::cout << "Memory allocation for backpropagation data failed." 
		    << std::endl;
	  return false;
	}
      }
      
      
      T* bpptr = bpdata;
      T* dptr = data;
      
      
      while(a[1] != 0){
	// copies layer input x to bpdata
	memcpy(bpptr, state, a[0]*sizeof(T));
	bpptr += a[0];

	// gemv(a[1], a[0], dptr, state, state); // s = W*s
	// gvadd(a[1], state, dptr + a[0]*a[1]); // s += b;

	// s = b + W*s
	gemv_gvadd(a[1], a[0], dptr, state, state,
		   a[1], state, dptr + a[0]*a[1]); // s += b;
	
	// s = g(v)
	
	if(a[2] != 0){ // not the last layer
	  // f(x) = b * ( (1 - Exp[-ax]) / (1 + Exp[-ax]) )
	  //      = b * ( 2 / (1 + Exp[-ax]) - 1)
	  
	  for(unsigned int i=0;i<a[1];i++){
	    T expbx = math::exp(-bf*state[i]);
	    // state[i] = af * ( (T(1.0f) - expbx) / (T(1.0f) + expbx) );
	    state[i] = af * ( T(2.0f) / (T(1.0f) + expbx) - T(1.0f) ); // numerically more stable (no NaNs)
	  }
	}
	
	
	dptr += (a[0] + 1)*a[1];
	a++; // next layer
      }
            
    }
    else{
      T* dptr = data;
      
      while(a[1] != 0){
	// gemv(a[1], a[0], dptr, state, state); // s = W*s
	// gvadd(a[1], state, dptr + a[0]*a[1]); // s += b;

	// s = b + W*s
	gemv_gvadd(a[1], a[0], dptr, state, state,
		   a[1], state, dptr + a[0]*a[1]);
	
	// s = g(v)
	
	if(a[2] != 0){ // not the last layer
	  // f(x)  = a * (1 - Exp[-bx]) / (1 + Exp[-bx])
	  // f'(x) = (0.5*a*b) * ( 1 + f(x)/a ) * ( 1 - f(x)/a )
	  
	  for(unsigned int i=0;i<a[1];i++){
	    T expbx = math::exp(-bf*state[i]);
	    // state[i] = af * ( (T(1.0f) - expbx) / (T(1.0f) + expbx) );
	    state[i] = af * ( T(2.0f) / (T(1.0f) + expbx) - T(1.0f) ); // numerically more stable (no NaNs)
	  }
	}
      
	dptr += (a[0] + 1)*a[1];
	a++; // next layer
      }
      
    }
    
    
    if(!outputValues.importData(state)){
      std::cout << "Failed to import data to vertex from memory." << std::endl;
      return false;
    }
    
    
    hasValidBPData = gradInfo;
    
    return true;
  }
  
  
  template <typename T> // number of layers
  unsigned int nnetwork<T>::length() const {
    return archLength;
  }
  
  
  template <typename T>
  bool nnetwork<T>::randomize(){
    if(compressed) return false;
    
    for(unsigned int i=0;i<size;i++)
      data[i] = T( 1.0f*(((float)rand())/((float)RAND_MAX)) - 0.5f );
    
    return true;
  }
  
  
  // sets learning rate globally
  template <typename T>
  bool nnetwork<T>::setlearningrate(T rate){
    this->rate = rate;
    return true;
  }
  
  
  // changes weights (smaller error): w_new -= lrate*w
  template <typename T>
  bool nnetwork<T>::backprop(const math::vertex<T>& error)
  {
    // updates weights with backpropagation
    
    if(compressed) return false;
    
    if(!hasValidBPData){
      std::cout << "Error: don't have valid backpropagation data." 
		<< std::endl;
      return false;
    }
    
    const T af = T(1.7159f);
    const T bf = T(0.6666f);
    
    // note: mallocs are slow and shouldn't been done everytime
    // backpropagation is calculated
    T* lgrad = (T*)malloc(maxwidth * sizeof(T));
    T* temp  = (T*)malloc(maxwidth * sizeof(T));
  
    if(lgrad == 0 || temp == 0){
      std::cout << "Memory allocation failure." << std::endl;
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
  
    
    unsigned int counter = archLength - 1;
    unsigned int *a = &(arch[archLength - 1]);
    
    if(!error.exportData(lgrad)){
      std::cout << "FAIL 3" << std::endl;
      free(lgrad);
      free(temp);
      return false;
    }
    
    T* _bpdata = bpdata;
    
    // goes to end of bpdata input lists
    for(unsigned int i=0;i<(archLength - 1);i++)
      _bpdata += arch[i];
  
    T* _data = data;
    
    // goes to end of data area
    _data += size;
    /*
      for(unsigned int i=1;i<archLength;i++)
      _data += ((arch[i-1])*(arch[i])) + arch[i];
    */
    
    
    while(counter > 1){
      // updates W and b in this layer
      
      // W += lrate * (lgrad * input^T)
      // b += lrate * lgrad;
      
      const unsigned int rows = *a;
      const unsigned int cols = *(a - 1);
      
      _bpdata -= *(a - 1);
      _data -= (*a) * (*(a - 1)) + *a;
      T* dptr = _data;
      
      // updates matrix W
      for(unsigned int y=0;y<rows;y++){
	T* bptr = _bpdata;
	for(unsigned int x=0;x<cols;x++){
	  dptr[x + y*cols] += rate * lgrad[y] * (*bptr);
	  bptr++;
	}
      }
      
      dptr += rows*cols;
      
      // updates biases b
      for(unsigned int y=0;y<rows;y++)
	dptr[y] += rate * lgrad[y];
      
      // calculates next lgrad
      
      // for hidden layers: local gradient is:
      // grad[n] = diag(..g'(v[i])..)*(W^t * grad[n+1])
      //
    
      for(unsigned int x=0;x<cols;x++){
	T* bptr = _bpdata;
	
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
      
      assert(_bpdata == bpdata);
      assert(_data == data);
      
      for(unsigned int y=0;y<rows;y++){
	T* bptr = _bpdata;
	for(unsigned int x=0;x<cols;x++){
	  _data[x + y*cols] += rate * lgrad[y] * (*bptr);
	  bptr++;
	}
      }
      
      
      _data += rows*cols;
    
      for(unsigned int y=0;y<rows;y++)
	_data[y] += rate*lgrad[y];
      
      _data += rows;
      
    }
    
    free(lgrad);
    free(temp);
    
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
    
    
    if(compressed) return false;
    
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
  
    
    unsigned int counter = archLength - 1;
    unsigned int *a = &(arch[archLength - 1]);
    
    if(!error.exportData(lgrad)){
      free(lgrad);
      free(temp);
      return false;
    }
    
    T* _bpdata = bpdata;
    
    // goes to end of bpdata input lists
    for(unsigned int i=0;i<(archLength - 1);i++)
      _bpdata += arch[i];
  
    T* _data = data;
    
    // goes to end of data area
    for(unsigned int i=1;i<archLength;i++){
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
      T* dptr = _data;
      
      for(unsigned int y=0;y<rows;y++){
	T* bptr = _bpdata;
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
      T* bptr = _bpdata;
    
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
      
      assert(_bpdata == bpdata);
      assert(_data == data);
      assert(gindex == 0);
      
      for(unsigned int y=0;y<rows;y++){
	T* bptr = _bpdata;
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
  

  // load & saves neuralnetwork data from file
  template <typename T>
  bool nnetwork<T>::load(const std::string& filename) throw(){
    try{
      if(compressed) return false;
      
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
      
      if(versionid != 100)
	return false;
      
      
      unsigned int* new_arch;
      T* new_data;
      unsigned int memuse;
      unsigned int new_maxwidth;
      unsigned int new_archLength;
      
      
      // gets architecture
      {
	if(!configuration.get(FNN_ARCH_CFGSTR,ints))
	  return false;
	  
	if(ints.size() < 2)
	  return false;
	
	for(unsigned int i=0;i<ints.size();i++)
	  if(ints[i] <= 0) return false;
	
	new_archLength = ints.size();
	
	new_arch = (unsigned int*)malloc((ints.size()+1)*sizeof(unsigned int));
	if(new_arch == 0)
	  return false;
	
	for(unsigned int i=0;i<new_archLength;i++)
	  new_arch[i] = ints[i];
	
	new_arch[new_archLength] = 0;
	
	memuse = 0;
	new_maxwidth = new_arch[0];
	
	unsigned int i = 1;
	while(new_arch[i] != 0){
	  memuse += (new_arch[i-1] + 1)*new_arch[i];
	  
	  if(new_arch[i] > new_maxwidth)
	    new_maxwidth = new_arch[i];
	  
	  i++;
	}
	
	new_data = (T*)malloc(memuse*sizeof(T));
	
	if(new_data == 0){
	  free(new_arch);
	  return false;
	}
	
	
	// sets up the loaded data (architecture)
	data = new_data;
	maxwidth = new_maxwidth;
	arch = new_arch;
	archLength = new_archLength;
	hasValidBPData = false;
	size = memuse;
	
	inputValues.resize(arch[0]);
	outputValues.resize(arch[archLength - 1]);
	
	ints.clear();
      }
      
      
      // gets layer weights & biases
      {
	char str[100];
	unsigned int index=0;

	for(unsigned int i=1;i<archLength;i++){
	  sprintf(str, FNN_LAYER_W_CFGSTR, (i - 1));
	  
	  if(!configuration.get(str, floats))
	    return false;
	  
	  if(floats.size() != new_arch[i-1]*new_arch[i])
	    return false;

	  for(unsigned int j=0;j<floats.size();j++, index++)
	    data[index] = T(floats[j]);
	  
	  floats.clear();
	  
	  
	  sprintf(str, FNN_LAYER_B_CFGSTR, (i - 1));
	  
	  if(!configuration.get(str, floats))
	    return false;
	  
	  if(floats.size() != new_arch[i])
	    return false;
	  
	  for(unsigned int j=0;j<floats.size();j++, index++)
	    data[index] = T(floats[j]);
	  
	  floats.clear();
	}
      }
      
      
      // no activation function information
      
      
      // gets layer learning rate
      {
	if(!configuration.get(FNN_LRATES_CFGSTR, floats))
	  return false;
	
	if(floats.size() > 0){
	  rate = T(0.0f);
	  
	  for(unsigned int i=0;i<floats.size();i++)
	    rate += T(floats[i]);
	  
	  rate /= T(floats.size());
	}
	
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
  bool nnetwork<T>::save(const std::string& filename) const throw(){
    try{
      if(compressed) return false;
      
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
	for(unsigned int i=0;i<archLength;i++)
	  ints.push_back(arch[i]);

	if(!configuration.set(FNN_ARCH_CFGSTR, ints))
	  return false;
	
	ints.clear();
      }
      
      
      // writes layer specific information
      
      
      // layer weights
      {
	char str[100];
	unsigned int index = 0;
	
	for(unsigned int i=1;i<archLength;i++){
	  
	  const unsigned int N = arch[i-1]*arch[i];
	  
	  for(unsigned int j=0;j<N;j++,index++){
	    float f;
	    if(!math::convert(f, data[index]))
	      return false;
	    
	    floats.push_back(f);
	  }
	  
	  sprintf(str, FNN_LAYER_W_CFGSTR, (i - 1));
	  
	  if(!configuration.set(str, floats))
	    return false;
	  
	  floats.clear();
	
	  // layer biases
	  const unsigned int M = arch[i];
	  
	  for(unsigned j=0;j<M;j++,index++){
	    float f;
	    if(!math::convert(f, data[index]))
	      return false;
	    
	    floats.push_back(f);
	  }
	  
	  sprintf(str, FNN_LAYER_B_CFGSTR, (i - 1));
	  
	  if(!configuration.set(str, floats))
	    return false;
	  
	  floats.clear();
	  
	}
      }
      
      
      // no activation functions info saved
      
      
      // layer learning rate
      {
	float f;
	if(!math::convert(f, rate))
	  return false;
	
	floats.push_back(f);
	
	if(!configuration.set(FNN_LRATES_CFGSTR, floats))
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


  //////////////////////////////////////////////////////////////////////
  
  template <typename T>
  bool nnetwork<T>::save2(const std::string& filename) const throw(){
    try{
      if(compressed) return false;
      
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
	for(unsigned int i=0;i<archLength;i++)
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
  bool nnetwork<T>::load2(const std::string& filename) throw(){
    try{
      if(compressed) return false;
      
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
      
      
      unsigned int* new_arch;
      T* new_data;
      unsigned int memuse;
      unsigned int new_maxwidth;
      unsigned int new_archLength;
      
      
      // gets architecture
      {
	if(!configuration.get(FNN_ARCH_CFGSTR,ints))
	  return false;
	  
	if(ints.size() < 2)
	  return false;
	
	for(unsigned int i=0;i<ints.size();i++)
	  if(ints[i] <= 0) return false;
	
	new_archLength = ints.size();
	
	new_arch = (unsigned int*)malloc((ints.size()+1)*sizeof(unsigned int));
	if(new_arch == 0)
	  return false;
	
	for(unsigned int i=0;i<new_archLength;i++)
	  new_arch[i] = ints[i];
	
	new_arch[new_archLength] = 0;
	
	memuse = 0;
	new_maxwidth = new_arch[0];
	
	unsigned int i = 1;
	while(new_arch[i] != 0){
	  memuse += (new_arch[i-1] + 1)*new_arch[i];
	  
	  if(new_arch[i] > new_maxwidth)
	    new_maxwidth = new_arch[i];
	  
	  i++;
	}
	
	new_data = (T*)malloc(memuse*sizeof(T));
	
	if(new_data == 0){
	  free(new_arch);
	  return false;
	}

	T* new_state = (T*)malloc(new_maxwidth*sizeof(T));
	T* new_temp  = (T*)malloc(new_maxwidth*sizeof(T));
	if(new_state == 0 || new_temp == 0){
	  if(new_state) free(new_state);
	  if(new_temp) free(new_temp);
	  free(new_data);
	  free(new_arch);
	  return false;
	} 
	
	if(data) free(data);
	if(arch) free(arch);
	if(state) free(state);
	if(temp) free(temp);

	// sets up the loaded data
	data = new_data;
	maxwidth = new_maxwidth;
	arch = new_arch;
	archLength = new_archLength;
	state = new_state;
	temp = new_temp;
	hasValidBPData = false;
	size = memuse;
	
	inputValues.resize(arch[0]);
	outputValues.resize(arch[archLength - 1]);
	
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
    if(compressed) return false;
    
    if(v.size() < size)
      v.resize(size);
    
    // NN exports TO vertex, vertex imports FROM NN
    return v.importData(data, size, 0);
  }
  
  template <typename T>
  bool nnetwork<T>::importdata(const math::vertex<T>& v) throw(){
    if(compressed) return false;
    
    if(v.size() < size)
      return false;
    
    // nn imports FROM vertex, vertex exports TO network
    return v.exportData(data, size, 0);
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
    
    memcpy(y, temp, yd*sizeof(T));
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
    
    memcpy(y, temp, yd*sizeof(T));
  }
  
  
  ////////////////////////////////////////////////////////////
  // can be used to decrease memory usage
  
  
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
  
  
  //////////////////////////////////////////////////////////////////////
  
  template class nnetwork< float >;
  template class nnetwork< double >;  
  template class nnetwork< math::atlas_real<float> >;
  template class nnetwork< math::atlas_real<double> >;
  
};
