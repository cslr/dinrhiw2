/*
 * SOM2D implementation
 *
 * - when implementation is stable / not compiled with electric fence
 *   enable posix_memalign() and disable malloc()s -> faster
 *   (efence doesn't work with posix_memalign())
 *
 */

#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>
#include <exception>
#include <math.h>

#include "SOM2D.h"
#include "cblas.h"
#include "vertex.h"
#include "data_source.h"
#include "linear_ETA.h"
#include "conffile.h"

//#include "dlib.h"


using namespace whiteice;
using namespace whiteice::math;



namespace whiteice
{

  SOM2D::SOM2D(unsigned int w, unsigned int h, unsigned int dim) :
    som_width(w) , som_height(h) , som_dimension(dim),
    som_widthf(w), som_heightf(h), som_dimensionf(dim)
  {
    somtable = 0;
    
    // posix_memalign(&somtable, sizeof(void*)*2,
    // som_dimension*som_height*som_width*sizeof(float));
    
    somtable = (float*)malloc(som_dimension*som_height*som_width*sizeof(float));
    
    if(somtable == 0)
      throw std::bad_alloc();
    
    show_visualization = false;
    show_eta = true;
    graphics_on = false;
    
    umatrix = 0;
    
    randomize();
  }
  
  
  SOM2D::~SOM2D()
  {
    if(somtable) free(somtable);

#if 0    
    close_visualization();
#endif
  }
  
  
  // learns given data
  bool SOM2D::learn(data_source< vertex<float> >& source, bool full) throw()
  {        
    if(source.size() <= 0) return false;
    if(source[0].size() != som_dimension) return false;
    
    ETA<double>* eta = 0;
    
    if(show_eta){
      try{ eta = new linear_ETA<double>(); }
      catch(std::exception& e){ return false; }
    }
    
#if 0
    if(show_visualization)
      open_visualization();
#endif
    
    const unsigned int MAXSTEPS = 1000;
    const unsigned int CNGSTEPS = 500*som_height*som_width;
    const unsigned int ETA_DELTA = MAXSTEPS / 10;
    
    learning_rate0 = 0.1f;
    hvariance0 = sqrtf(som_height*som_width);
    hvariance_t1 = ((float)MAXSTEPS)/(logf(hvariance0));
    learning_rate_t2 = ((float)MAXSTEPS);
    
    float hvariance = hvariance0;
    float learning_rate = learning_rate0;
    
    if(eta) eta->start(0.0, (double)(MAXSTEPS + CNGSTEPS));
    
    // SELF-ORGANIZING PHASE
    
    for(unsigned int i=0, eta_counter=0;i<MAXSTEPS;i++, eta_counter++){
      
      if(eta){
	eta->update((double)i);

	if(eta_counter >= ETA_DELTA){
	  eta_counter = 0;
	  report_eta(i, MAXSTEPS+CNGSTEPS, eta);
	}
      }
      
      
#if 0      
      draw_visualization();
#endif
      
      // FINDS WINNER FOR RANDOMLY CHOSEN DATA
      
      unsigned int dindex = rand() % source.size();            
      unsigned int winner = find_winner(source[dindex].data);
      
      // UPDATES SOM LATTICE
      
      // winner coordinates as floats
      float wx = winner % som_width , wy = winner / som_width;
      
      // winner index (back) to somtable memory index
      winner *= som_dimension;
      
      float x = 0, y = 0; // floating point coordinates of current index
      float h;
      
      
      for(unsigned int index=0;index<som_height*som_width*som_dimension;index += som_dimension){
	
	// calculates h() function for a given point
	
	h = wraparound_sqdistance(x - wx, y - wy) / (-2.0f * hvariance);
	h = learning_rate * expf(h);
	
	// w -= h*w <=> w = (1-h) * w
	cblas_sscal(som_dimension, (1 - h), &(somtable[index]), 1);
	
	// w += h*x
	cblas_saxpy(som_dimension,  h, &(somtable[winner]),  1, &(somtable[index]), 1);
	
	
	// updates coordinates
	x += 1.0f;
	
	if(x >= som_widthf){
	  x = 0.0f;
	  y += 1.0f;
	}
      }
      
      
      // UPDATES LEARNING PARAMETERS
      
      learning_rate = learning_rate0 * expf( i / (-learning_rate_t2));
      hvariance = hvariance0 * expf(i/(-hvariance_t1)) * expf(i/(-hvariance_t1));
    }
    
    
    // CONVERGENCE PHASE
    if(full)
      for(unsigned int i=0, eta_counter=0;i<CNGSTEPS;i++, eta_counter++){
	
	if(eta){
	  eta->update((double)(i+MAXSTEPS));
	  
	  if(eta_counter >= ETA_DELTA){
	    eta_counter = 0;
	    report_eta(i+MAXSTEPS, MAXSTEPS+CNGSTEPS, eta);
	  }
	}

#if 0	
	draw_visualization();
#endif
      
	// FINDS WINNER FOR RANDOMLY CHOSEN DATA
	
	unsigned int dindex = rand() % source.size();            
	unsigned int winner = find_winner(source[dindex].data);
	
	// UPDATES SOM LATTICE
	
	// winner coordinates as floats
	float wx = winner % som_width , wy = winner / som_width;
	
	// winner index (back) to somtable memory index
	winner *= som_dimension;
	
	float x = 0, y = 0; // floating point coordinates of current index
	float h;
	
	
	for(unsigned int index=0;index<som_height*som_width*som_dimension;index += som_dimension){
	  
	  // calculates h() function for a given point
	  
	  h = wraparound_sqdistance(x - wx, y - wy) / (-2.0f * hvariance);
	  h = 0.01f * expf(h);
	  
	  // w -= h*w <=> w = (1 - h) * w
	  cblas_sscal(som_dimension, (1 - h), &(somtable[index]), 1);
	  
	  // w += h*x
	  cblas_saxpy(som_dimension,  h, &(somtable[winner]),  1, &(somtable[index]), 1);
	  
	  
	  // updates coordinates
	  x += 1.0f;
	  
	  if(x >= som_widthf){
	    x = 0.0f;
	    y += 1.0f;
	  }
	}
	
	
	// UPDATES LEARNING PARAMETERS
	
	hvariance = 0.5 + ((CNGSTEPS - i)/((float)CNGSTEPS))*5;
      }
    
    
    if(eta){
      report_eta(CNGSTEPS, MAXSTEPS+CNGSTEPS, eta);
      
      delete eta;
    }
    
    
    return true;
  }
    
  
  
  // randomizes som vertex values
  bool SOM2D::randomize() throw()
  {
    // calculates random values to between [-1,1]
    
    const unsigned int N = som_dimension*som_width*som_height;
    
    for(unsigned int i=0;i<N;i++)
      somtable[i] = rand()/((float)RAND_MAX);
    
    // normalizes lengt of som vectors
    float len;
    
    for(unsigned int i=0;i<N;i += som_dimension){
      len = cblas_snrm2(som_dimension, &(somtable[i]), 1);
      if(len != 0.0f) len = 1.0f / len;
      cblas_sscal(som_dimension, len, &(somtable[i]), 1);
    }
    
    return true;
  }
  
  
  
  float SOM2D::somdistance(const vertex<float>& v1,
			   const vertex<float>& v2) const throw()
  {
    if(v1.size() != som_dimension || v2.size() != som_dimension)
      return -1.0f; // (error)
       
    
    // don't make two separate activate() calls because
    // that would cause data to be read through twice
    // (with lots of data this causes more cache misses)
    
    const unsigned int N=som_dimension*som_height*som_width;
  
    unsigned int winner[2];
    float tmp, result[2];
    
    result[0] = cblas_sdot(som_dimension, v1.data, 1, somtable, 1);
    result[1] = cblas_sdot(som_dimension, v2.data, 1, somtable, 1);
    winner[0] = 0; winner[1] = 0;
    
    for(unsigned int i=som_dimension;i<N;i += som_dimension){
      
      tmp = cblas_sdot(som_dimension, v1.data, 1, &(somtable[i]), 1);
      if(result[0] < tmp){
	result[0] = tmp;
	winner[0] = i/som_dimension;
      }
      
      tmp = cblas_sdot(som_dimension, v2.data, 1, &(somtable[i]), 1);
      if(result[1] < tmp){
	result[1] = tmp;
	winner[1] = i/som_dimension;
      }
    }
    
    // finally converts indexes to coordinates
    
    float dy = (float)( (winner[0] / som_height) - (winner[1] / som_height) );
    float dx = (float)( (winner[0] % som_height) - (winner[1] % som_height) );
    
    return sqrtf(wraparound_sqdistance(dx, dy));
  }
  
  
  // returns winner vertex raw index for a given vertex
  unsigned int SOM2D::activate(const vertex<float>& v) const throw()
  {
    unsigned int winner = find_winner(v.data);
    
    return winner;
  }
    
  
  // reads som vertex given lattice coordinate
  vertex<float> SOM2D::operator()(unsigned int i, unsigned int j) const throw()
  {
    vertex<float> r(som_dimension);
    memcpy(r.data, &(somtable[(i + j*som_width)*som_dimension]), som_dimension*sizeof(float));
    
    return r;
  }
  
  
  // reads som vertex given direct raw index coordinate to a table
  vertex<float> SOM2D::operator()(unsigned int index) const throw()
  {
    vertex<float> v(som_dimension);
    
    memcpy(v.data, &(somtable[index*som_dimension]), som_dimension*sizeof(float));
    
    return v;
  }
  

  
  ////////////////////////////////////////////////////////////////////////////////
  
  // constant field names in SOM configuration files
  const std::string SOM_VERSION_CFGSTR  = "SOM_CONFIG_VERSION";
  const std::string SOM_SIZES_CFGSTR    = "SOM_SIZES";
  const std::string SOM_PARAMS_CFGSTR   = "SOM_PARAMS";
  const std::string SOM_ETA_CFGSTR      = "SOM_USE_ETA";
  const std::string SOM_ROWPROTO_CFGSTR = "SOM_ROW%d";  
  
  
  
  // loads SOM data from file , failure puts
  // SOM in unknown state!
  bool SOM2D::load(const std::string& filename) throw()
  {
    whiteice::conffile configuration;
    std::vector<int> ints;
    std::vector<float> floats;
    std::vector<std::string> strings;    
    
    if(!configuration.load(filename))
      return false;
    
    ints.clear();
    configuration.get(SOM_VERSION_CFGSTR, ints);
    
    if(ints.size() != 1) return false;
    
    // recognizes version 0.101 (= 101)
    if(ints[0] != 101)
      return false;
    
    ints.clear();
    if(!configuration.get(SOM_SIZES_CFGSTR, ints)) return false;
    if(ints.size() != 3) return false;    
    this->som_width = ints[0];
    this->som_height = ints[1];
    this->som_dimension = ints[2];
    
    {
      float* tmp = (float*)realloc(somtable, sizeof(float)*som_width*som_height*som_dimension);
				   
				   
      if(tmp == 0) return false;
      else somtable = tmp;
    }
    
    floats.clear();
    if(!configuration.get(SOM_PARAMS_CFGSTR, floats)) return false;
    
    ints.clear();
    if(!configuration.get(SOM_ETA_CFGSTR, ints)) return false;
    if(ints.size() != 1) return false;
  
    this->show_eta = (bool)(ints[0]);
    
    char *buf = 0;
    
    try{
      // now starts loading actual som data
            
      buf = new char[50];
      
      for(unsigned int i=0;i<som_width*som_height;i++){
	sprintf(buf,SOM_ROWPROTO_CFGSTR.c_str(), i);
	floats.clear();
	
	if(!configuration.get(buf, floats)) return false;
	if(floats.size() != som_dimension) return false;
	
	for(unsigned int j=0;j<som_dimension;j++)
	  somtable[i*som_dimension + j] = floats[j];
      }
      
      delete[] buf;
    }
    catch(std::exception& e){
      // probably bad_alloc...
      if(buf) delete[] buf;
      return false;
    }  
    
  return true;
  }
  
  
  // saves SOM data to file
  bool SOM2D::save(const std::string& filename) const throw()
  {  
    whiteice::conffile configuration;
    std::vector<int> ints;
    std::vector<float> floats;
    std::vector<std::string> strings;
    
    ints.clear();
    ints.push_back(101); // 1000 = 1.000 etc. 101 = 0.101
    if(!configuration.set(SOM_VERSION_CFGSTR, ints)) return false;
    
    ints.clear();
    ints.push_back(som_width);
    ints.push_back(som_height);
    ints.push_back(som_dimension);
    if(!configuration.set(SOM_SIZES_CFGSTR, ints)) return false;
    
    floats.clear();
    floats.push_back(0.0f); // dummy value
    if(!configuration.set(SOM_PARAMS_CFGSTR, floats)) return false;
    
    ints.clear();
    ints.push_back((int)show_eta);
    if(!configuration.set(SOM_ETA_CFGSTR, ints)) return false;

    char *buf = 0;
    
    try{
      // now starts saving actual som data
      // number of som rows etc. can be calculated from
      // SOM_SIZES
      
      buf = new char[50];
      
      floats.resize(som_dimension);
      
      for(unsigned int i=0;i<som_width*som_height;i++){      
	sprintf(buf,SOM_ROWPROTO_CFGSTR.c_str(), i);		
	
	// accuracy loss maybe problem here !!
	// (should add double and long double support
	//  to ConfFile + generic printable interface support)
	for(unsigned int j=0;j<som_dimension;j++)
	  floats[j] = (float)somtable[i*som_dimension + j];
	
	if(!configuration.set(buf, floats)){
	  delete[] buf;
	  return false;
	}
      }
      
      delete[] buf;
    }
    catch(std::exception& e){
      // probably bad_alloc...
      if(buf) delete[] buf;
      return false;
    }
    
    // saves configuration file and returns results
    return configuration.save(filename);
  }
  

  bool SOM2D::show(bool on) throw()
  {
#if 0
    if(graphics_on && on == false){
      close_visualization();
    }
    if(graphics_on == false && on == true){
      open_visualization();
      draw_visualization();
    }
#endif
    
    return true;
  }

  ////////////////////////////////////////////////////////////////////////////////
  
  
  unsigned int SOM2D::find_winner(const float* vmemory) const throw()
  {
    // calculates inner products and finds the biggest one
    const unsigned int N = som_dimension*som_width*som_height;
    
    unsigned int winner = 0;
    float tmp, result = 0;
    
    result = cblas_sdot(som_dimension, vmemory, 1, somtable, 1);    
    
    for(unsigned int i=som_dimension;i<N;i+= som_dimension){
      tmp = cblas_sdot(som_dimension, vmemory, 1, &(somtable[i]), 1);
      
      if(result < tmp){
	result = tmp;
	winner = i/som_dimension;
      }
    }
    
    return winner;
  }
  
  
  // calculates squared wrap-a-round distance between two coordinates
  float SOM2D::wraparound_sqdistance(float dx, float dy) const throw()
  {
    // wrap'a'round distance
    
    if(dy > 0.0f){
      if(som_heightf - dy < dy)
	dy = som_heightf - dy;
    }
    else{
      if(som_heightf + dy < -dy)
	dy += som_heightf;
    }
    
    if(dx > 0.0f){
      if(som_widthf - dx < dx)
	dx = som_widthf - dx;
    }
    else{
      if(som_widthf + dx < -dx)
	dx += som_widthf;
    }
    
    return (dx*dx + dy*dy);
  }
  
  
  ////////////////////////////////////////////////////////////////////////////////
  
#if 0
  bool SOM2D::open_visualization() throw()
  {
    using namespace dlib_global;
    
    if(graphics_on) return true;

    umatrix = (float*)malloc(som_width*som_height*sizeof(float));
    if(umatrix == 0) return false;
    
    dlib.extensions("verbose off");
    dlib.setName("SOM");
    if(!dlib.open(640, 480)) return false;
    
    dlib.clear();
    dlib.update();    
    
    graphics_on = true;
    
    return true;
  } 
  
  
  bool SOM2D::close_visualization() throw()
  {
    using namespace dlib_global;
    
    if(!graphics_on) return true;
    
    if(dlib.close() == false)
      return false;

    free(umatrix);
    umatrix = 0;        
    
    graphics_on = false;
    
    return true;
  }
  
  
  bool SOM2D::draw_visualization() throw()
  {
    using namespace dlib_global;
    
    if(!graphics_on) return false;
    
    float* vmemory = (float*)malloc(sizeof(float)*som_dimension);
    if(vmemory == 0) return false;
    
    // calculates U-matrix visualization
    // uses square 4 neighbourhood
    
    float umatrix_max = 0.0f;
    
    // calculates simple non-border cases 
    for(unsigned int y=1,index=som_width*som_dimension;y<(som_height - 1);y++){
      index += som_dimension;
      for(unsigned int x=1;x<(som_width - 1);x++){
	
	cblas_scopy(som_dimension, &(somtable[index]), 1, vmemory, 1);
	cblas_sscal(som_dimension, -1.0f, vmemory, 1);
	cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_dimension]), 1, vmemory, 1);
	cblas_saxpy(som_dimension, 0.25f, &(somtable[index - som_dimension]), 1, vmemory, 1);
	cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_width*som_dimension]), 1, vmemory, 1);
	cblas_saxpy(som_dimension, 0.25f, &(somtable[index - som_width*som_dimension]), 1, vmemory, 1);
	
	umatrix[index/som_dimension] = cblas_snrm2(som_dimension, vmemory, 1);
	if(umatrix[index/som_dimension] > umatrix_max)
	  umatrix_max = umatrix[index/som_dimension];
	
	index += som_dimension;
      }
      index += som_dimension;
    }
    
    
    // calculates border cases.
    
    // y=0 line (no corners)
    for(unsigned int index=som_dimension;index<(som_width - 1)*som_dimension;index += som_dimension){

      cblas_scopy(som_dimension, &(somtable[index]), 1, vmemory, 1);
      cblas_sscal(som_dimension, -1.0f, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index - som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_width*som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + (som_height - 1)*som_width*som_dimension]), 1, vmemory, 1);
      
      umatrix[index/som_dimension] = cblas_snrm2(som_dimension, vmemory, 1);
      if(umatrix[index/som_dimension] > umatrix_max)
	umatrix_max = umatrix[index/som_dimension];
    }
    
    // y=(height-1) line (no corners)
    for(unsigned int index=som_dimension;index<(som_width - 1)*som_dimension;index += som_dimension){
      
      cblas_scopy(som_dimension, &(somtable[index + (som_height-1)*som_width*som_dimension]), 1, vmemory, 1);
      cblas_sscal(som_dimension, -1.0f, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f,
		  &(somtable[index + som_dimension + som_width*(som_height - 1)*som_dimension]), 1,
		  vmemory, 1);
      
      cblas_saxpy(som_dimension, 0.25f,
		  &(somtable[index - som_dimension + som_width*(som_height - 1)*som_dimension]), 1,
		  vmemory, 1);
      
      cblas_saxpy(som_dimension, 0.25f,
		  &(somtable[index - som_width*som_dimension + som_width*(som_height-1)*som_dimension]), 1,
		  vmemory, 1);
      
      cblas_saxpy(som_dimension, 0.25f,
		  &(somtable[index]), 1,
		  vmemory, 1);
      
      umatrix[index/som_dimension + (som_height-1)*som_width] = cblas_snrm2(som_dimension, vmemory, 1);      
      if(umatrix[index/som_dimension + (som_height-1)*som_width] > umatrix_max)
	umatrix_max = umatrix[index/som_dimension + (som_height-1)*som_width];
    }
    
    // x = 0 line (no corners)
    for(unsigned int index=som_width*som_dimension;
	index<som_width*(som_height-1)*som_dimension;
	index += som_width*som_dimension){
      
      cblas_scopy(som_dimension, &(somtable[index]), 1, vmemory, 1);
      cblas_sscal(som_dimension, -1.0f, vmemory, 1);
      
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + (som_width - 1)*som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_width*som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index - som_width*som_dimension]), 1, vmemory, 1);
      
      umatrix[index/som_dimension] = cblas_snrm2(som_dimension, vmemory, 1);
      if(umatrix[index/som_dimension] > umatrix_max)
	umatrix_max = umatrix[index/som_dimension];
    }
    
    // x = (width-1) line (no corners)
    for(unsigned int index=(2*som_width - 1)*som_dimension;
	index<som_width*(som_height-1)*som_dimension;
	index += som_width*som_dimension){

      cblas_scopy(som_dimension, &(somtable[index]), 1, vmemory, 1);
      cblas_sscal(som_dimension, -1.0f, vmemory, 1);
      
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index - som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + (1 - som_width)*som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index + som_width*som_dimension]), 1, vmemory, 1);
      cblas_saxpy(som_dimension, 0.25f, &(somtable[index - som_width*som_dimension]), 1, vmemory, 1);
      
      umatrix[index/som_dimension] = cblas_snrm2(som_dimension, vmemory, 1);
      if(umatrix[index/som_dimension] > umatrix_max)
	umatrix_max = umatrix[index/som_dimension];
    }
    
    // calculates (0,0)
    cblas_scopy(som_dimension, &(somtable[0]), 1, vmemory, 1);
    cblas_sscal(som_dimension, -1.0f, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f, &(somtable[(som_width - 1)*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f, &(somtable[som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f, &(somtable[som_width*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f, &(somtable[(som_height - 1)*som_width*som_dimension]), 1, vmemory, 1);
      
    umatrix[0] = cblas_snrm2(som_dimension, vmemory, 1);    
    if(umatrix[0] > umatrix_max)
      umatrix_max = umatrix[0];
    
    // calculates (0, width-1)
    cblas_scopy(som_dimension, &(somtable[(som_width - 1)*som_dimension]), 1, vmemory, 1);
    cblas_sscal(som_dimension, -1.0f, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f, &(somtable[0]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f, &(somtable[(som_width - 2)*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f, &(somtable[(2*som_width - 1)*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_width*som_height - 1)*som_dimension]), 1,
		vmemory, 1);
      
    umatrix[som_width - 1] = cblas_snrm2(som_dimension, vmemory, 1);
    if(umatrix[som_width - 1] > umatrix_max)
      umatrix_max = umatrix[som_width - 1];
    
    // calculates (height-1, 0);

    cblas_scopy(som_dimension, &(somtable[(som_height - 1)*som_width*som_dimension]), 1, vmemory, 1);
    cblas_sscal(som_dimension, -1.0f, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_width*(som_height - 1) + 1)*som_dimension]), 1, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_width*som_height - 1)*som_dimension]), 1, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_height - 2)*som_width*som_dimension]), 1, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[0]), 1, vmemory, 1);
      
    umatrix[(som_height-1)*som_width] = cblas_snrm2(som_dimension, vmemory, 1);
    if(umatrix[(som_height - 1)*som_width] > umatrix_max)
      umatrix_max = umatrix[(som_height - 1)*som_width];
    
    // calculates (height-1, width-1)

    cblas_scopy(som_dimension, &(somtable[(som_width*som_height - 1)*som_dimension]), 1, vmemory, 1);
    cblas_sscal(som_dimension, -1.0f, vmemory, 1);
    
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_width*som_height - 2)*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_height - 1)*som_width*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[(som_width - 1)*som_dimension]), 1, vmemory, 1);
    cblas_saxpy(som_dimension, 0.25f,
		&(somtable[((som_height - 1)*som_width - 1)*som_dimension]), 1, vmemory, 1);
    
    umatrix[som_height*som_width - 1] = cblas_snrm2(som_dimension, vmemory, 1);
    if(umatrix[som_height*som_width - 1] > umatrix_max)
      umatrix_max = umatrix[som_height*som_width - 1];
    
    free(vmemory);
    
    if(umatrix_max == 0.0f)
      umatrix_max = 1.0f;
    
    // draws the umatrix visualization
    // umatrix values are between 0..1
    
    dlib.clear();
    
    const unsigned int W = dlib.width();
    const unsigned int ph = dlib.height()/som_height;
    const unsigned int pw = dlib.width()/som_width;
    unsigned int color = (((unsigned int)((umatrix[0]/umatrix_max)*0xFF))*(0x010101));
    
    for(unsigned int j=0,index = 0;j<som_height;j++){
      for(unsigned int i=0;i<som_width;i++,index++){
	
	for(unsigned int y=0;y<ph;y++){
	  for(unsigned int x=0;x<pw;x++){
	    dlib[i*pw + j*ph*W + x + y*W] = color;
	  }
	}
	
	
	color = (((unsigned int)((umatrix[index]/umatrix_max)*0xFF))*(0x010101));
      }
    }
    
    
    dlib.update();
    
    return true;
  }
#endif

  
  // size of the som lattice
  unsigned int SOM2D::width() const throw(){ return som_width; }
  unsigned int SOM2D::height() const throw(){ return som_height; }
  unsigned int SOM2D::dimension() const throw(){ return som_dimension; }
  unsigned int SOM2D::size() const throw(){ return (som_width*som_height); }
  
  

  
  ////////////////////////////////////////////////////////////////////////////////

  
  void SOM2D::report_eta(const unsigned int ITER,
			 const unsigned int MAXITER,
			 ETA<double>* eta)
  {
    std::cout << "SOM ITER: " << ITER << " / " << MAXITER  << " ";
      
    double secs = eta->estimate();
    unsigned int mins  = 0;
    unsigned int hours = 0;
    unsigned int days = 0;
    unsigned int years = 0;
    
    years = (unsigned int)(secs/(365.242199*24.0*3600.0));
    secs -= years*(365.242199*24.0*3600.0);
    
    days  = (unsigned int)(secs/(24.0*3600.0));
    secs -= days*24.0*3600.0;
    
    hours = (unsigned int)(secs/3600.0);
    secs -= hours*3600.0;
    
    mins  = (unsigned int)(secs/60.0);
    secs -= mins*60.0;
    
    std::cout << "ETA: "; 
    
    if(years > 0){
      if(years == 1)
	std::cout << "1 year ";
      else if(years > 1)
	std::cout << years << " years ";
    }
    
    if(days > 0 || years > 0){
      if(days == 1)
	std::cout << "1 day ";
      else if(days > 1)
	std::cout << days << " days ";
    }
    
    if(hours > 0 || days > 0 || years > 0){
      if(hours == 1)
	std::cout << "1 hour ";
      else if(hours > 1)
	std::cout << hours << " hours ";
    }
    
    if(mins > 0 || hours > 0 || days > 0 || years > 0){
      if(mins == 1)
	std::cout << "1 min ";
      else if(mins > 0)
	std::cout << mins << " mins ";
    }
    
    if(mins > 0 || hours > 0 || days >> 0 || years > 0){
      secs = (unsigned int)secs;
    }
    
    
    if(secs > 1)
      std::cout << secs << " secs " << std::endl;
    else
      std::cout << secs << " sec " << std::endl;
  }
    
};
  


