
#ifndef SOM_cpp
#define SOM_cpp

#include "SOM.h"

#include <iostream>
#include <vector>
#include <stdexcept>
#include <exception>

#include <stdio.h>
#include <stdlib.h>

#include "linear_ETA.h"
#include "conffile.h"
#include "SOMstatic.h"
#include "dlib.h"


namespace whiteice
{
  
  template <typename T>
  SOM<T>::SOM(unsigned int w, unsigned int h, unsigned int d) throw(std::logic_error) :
    width(w), height(h), dimension(d) 
  {
    if(w <= 0 || h <= 0 || d <= 0)
      throw std::logic_error("width,height and/or dimension is zero");
    
    // checks w and h are powers of two
    {
      unsigned int tmp;
      
      wbits = 0;
      tmp = w;
      while((tmp & 1) == 0){ tmp >>= 1; wbits++; }
      tmp >>= 1;
      if(tmp != 0) throw std::logic_error("width is not power of two");
      
      tmp = h;
      while((tmp & 1) == 0) tmp >>= 1;
      tmp >>= 1;
      if(tmp != 0) throw std::logic_error("height is not power of two");
    }
    
    som.resize(width*height);
    
    for(unsigned int i=0;i<width*height;i++){
      som[i].resize(dimension);
      
      for(unsigned int j=0;j<dimension;j++){
	som[i][j] = T((((float)rand()) / ((float)RAND_MAX)) - 0.5f);
      }
      
      normalize_length(som[i]);
    }
    
    
    initial_learning_rate = T(0.90f);
    initial_variance_distance = 
      T((width < height) ? height : width); // max(width, height)
    
    target_variance_distance = T(0.5f);
    
    
    show_eta = true;
    show_visualization = true;
    graphics_on = false;
  }
  
  
  template <typename T>
  SOM<T>::~SOM()
  {
    close_visualization();
  }

  
  // run SOM with given data
  template <typename T>
  bool SOM<T>::learn(const std::vector< std::vector<T> >& data) throw()
  {
    if(data.size() <= 0) return false;
    if(data[0].size() != dimension) return false;
    
    ETA<unsigned int>* eta;
    
    try{ eta = new linear_ETA<unsigned int>(); }
    catch(std::exception& e){ return false; }
    
    learning_rate = initial_learning_rate;
    variance_distance = initial_variance_distance;
    
    // should be 500*n*n times
    const unsigned int maxstep = 10 * data.size();
    
    if(show_visualization)
      open_visualization();
    
    if(show_eta)
      eta->start(0, maxstep);    
    
    for(unsigned int i=0;i<maxstep;i++)
    {
      eta->update(i);
      
      if(show_eta){      
	if(i % 100 == 0)
	  std::cout << "SOM ITER: " << i << " / " << maxstep  << " "
		    << "ETA: " << eta->estimate() << " secs" << std::endl;
      }
      
      if(show_visualization)
	draw_visualization();
      
      
      unsigned int index = rand() % data.size();
      unsigned int winner = find_closest(data[index]);        
      
      move_towards(winner, data[index]);
      
      // updates 'learning' parameters
      learning_rate = initial_learning_rate * (1.0f - ((float)i)/(float)maxstep);
      variance_distance = initial_variance_distance - 
	(((float)i)/(float)maxstep) * (initial_variance_distance - target_variance_distance);
    }
    
    delete eta;
    
    return true;
  }
  
  
  // returns distance in feature space
  template <typename T>
  T SOM<T>::som_distance(const std::vector<T>& v,
			 const std::vector<T>& w) const throw()
  {
    return T(sqrt(som_sqr_distance(find_closest(v), find_closest(w))));
  }
  
  
  // returns index to vector representing given vector
  template <typename T>
  unsigned int SOM<T>::representing_vector(const std::vector<T>& v) const throw()
  {
    return find_closest(v);
  }
  
  
  // returns som vector for given vector index
  template <typename T>
  std::vector<T>& SOM<T>::operator[](unsigned int index) const throw(std::out_of_range)
  {
    if(index >= som.size())
      throw std::logic_error("vector index too big");
    
    return som[index];
  }
  
  
  
  // randomizes som values
  template <typename T>
  bool SOM<T>::randomize() const throw()
  {
    for(unsigned int i=0;i<width*height;i++){
      for(unsigned int j=0;j<dimension;j++){
	som[i][j] = T((((float)rand()) / ((float)RAND_MAX)) - 0.5f);
      }
      
      normalize_length(som[i]);
    }
    
    return true;
  }
  
  
  
  // loads SOM data from file , failure puts
  // SOM in unknown state!
  template <typename T>
  bool SOM<T>::load(const std::string& filename) throw()
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
    
    // recognizes version 0.1 (= 100)
    if(ints[0] != 100)
      return false;
    
    ints.clear();
    if(!configuration.get(SOM_SIZES_CFGSTR, ints)) return false;
    if(ints.size() != 4) return false;
    this->wbits = ints[0];
    this->width = ints[1];
    this->height = ints[2];
    this->dimension = ints[3];
    
    floats.clear();
    if(!configuration.get(SOM_PARAMS_CFGSTR, floats)) return false;
    if(floats.size() != 5) return false;
    
    this->initial_learning_rate = floats[0];
    this->initial_variance_distance = floats[1];
    this->target_variance_distance = floats[2];
    this->learning_rate = floats[3];
    this->variance_distance = floats[4];
    
    
    ints.clear();
    if(!configuration.get(SOM_ETA_CFGSTR, ints)) return false;
    if(ints.size() != 1) return false;
  
    this->show_eta = (bool)(ints[0]);
    
    som.resize(width*height);
    char *buf = 0;
    
    try{
      // now starts loading actual som data
            
      buf = new char[50];
      
      for(unsigned int i=0;i<som.size();i++){
	sprintf(buf,SOM_ROWPROTO_CFGSTR.c_str(), i);
	floats.clear();
	som[i].resize(dimension);
	
	if(!configuration.get(buf, floats)) return false;
	if(floats.size() != dimension) return false;
	
	for(unsigned int j=0;j<dimension;j++)
	  som[i][j] = T(floats[j]);
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
  template <typename T>
  bool SOM<T>::save(const std::string& filename) const throw()
  {  
    whiteice::conffile configuration;
    std::vector<int> ints;
    std::vector<float> floats;
    std::vector<std::string> strings;
    
    ints.clear();
    ints.push_back(100); // 1000 = 1.000 etc. 100 = 0.1
    if(!configuration.set(SOM_VERSION_CFGSTR, ints)) return false;
    
    ints.clear();
    ints.push_back(wbits); // redundant
    ints.push_back(width);
    ints.push_back(height);
    ints.push_back(dimension);
    if(!configuration.set(SOM_SIZES_CFGSTR, ints)) return false;
    
    floats.clear();
    floats.push_back(initial_learning_rate);
    floats.push_back(initial_variance_distance);
    floats.push_back(target_variance_distance);
    floats.push_back(learning_rate);
    floats.push_back(variance_distance);
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
      
      for(unsigned int i=0;i<som.size();i++){      
	sprintf(buf,SOM_ROWPROTO_CFGSTR.c_str(), i);
	floats.clear();
	floats.resize(som[i].size());
	
	// accuracy loss maybe problem here !!
	// (should add double and long double support
	//  to ConfFile + generic printable interface support)
	for(unsigned int j=0;j<som[i].size();j++)
	  floats[j] = (float)som[i][j];
	
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
  
  /************************************************************/
  
  
  // finds closests som representation
  // vector index for given vector
  template <typename T>
  unsigned int SOM<T>::find_closest(const std::vector<T>& data)
    const throw()
  {
    unsigned int winner = 0;
    T smallest_distance;
    
    if(width > height)
      smallest_distance = T(width*width);
    else
      smallest_distance = T(height*height);
    
    
    for(unsigned int index=0;index<som.size();index++){
      T distance = vector_sqr_distance(data, som[index]);
      
      if(distance < smallest_distance){
	distance = smallest_distance;
	winner = index;
      }
    }
    
    return winner;
  }
  
  
  // moves all nodes in a neighbouthood of winner towards
  // data vector
  template <typename T>
  bool SOM<T>::move_towards(unsigned int winner,
			    const std::vector<T>& data) throw()
  {
    for(unsigned int index=0;index<som.size();index++){
      T activation = learning_rate * exp(- som_sqr_distance(winner, index)/variance_distance);
      
      for(unsigned int r=0;r<dimension;r++)
	som[index][r] += activation *
	  (data[r] - som[index][r]);
    mv SO}
    
    return true;
  }
  
  
  // calculates |x - y|^2 , inputs *must* have equal length
  template <typename T>
  T SOM<T>::vector_sqr_distance(const std::vector<T>& x,
				const std::vector<T>& y) const throw()
  {
    T len = T(0.0f);
    
    for(unsigned int i=0;i<x.size();i++)
      len += T( (x[i] - y[i]) * (x[i] - y[i]) );
    
    return len;
  }
  
  
  
  // calculates squared distance between vectors in som lattice
  template <typename T>
  T SOM<T>::som_sqr_distance(unsigned int index1,
			     unsigned int index2) const throw()
  {
    signed int x1 = index1 & (width - 1);
    signed int x2 = index2 & (width - 1);
    signed int y1 = index1 >> wbits;
    signed int y2 = index2 >> wbits;
    
    signed int dx = x1 - x2;
    signed int a = x2 - x1;
    
    if(dx < 0) dx += width;
    else if(a < 0) a += width;    
    if(a < dx) dx = a;
    
    signed int dy = y1 - y2;
    a = y2 - y1;
    
    if(dy < 0) dy += height;
    else if(a < 0) a += height;
    if(a < dy) dy = a;
    
    return T(dx*dx + dy*dy);
  }
  

  
  // normalizes length of given vector
  template <typename T>
  void SOM<T>::normalize_length(std::vector<T>& v) const throw()
  {
    T len = T(0.0f);
    
    for(unsigned int i=0;i<v.size();i++)
      len += v[i] * v[i];
    
    
    len = (float)sqrt(len);
    
    if(len)
      for(unsigned int i=0;i<v.size();i++)
	v[i] /= len;
  }
  
  
  
  template <typename T>
  bool SOM<T>::open_visualization() throw()
  {
    using namespace dlib_global;
    
    if(graphics_on) return true;
    
    dlib.extensions("verbose off");
    dlib.setName("SOM");
    if(!dlib.open(640, 480)) return false;
    
    dlib.clear();
    dlib.update();
    
    umatrix.resize(som.size());
    
    graphics_on = true;
    
    return true;
  } 
  
  
  template <typename T>
  bool SOM<T>::close_visualization() throw()
  {
    using namespace dlib_global;
    
    if(!graphics_on) return true;
    
    if(dlib.close() == false)
      return false;
    
    umatrix.resize(0);
    
    graphics_on = false;
    
    return true;
  }
  
  
  template <typename T>
  bool SOM<T>::draw_visualization() throw()
  {
    using namespace dlib_global;
    
    if(!graphics_on) return false;
    
    // calculates U-matrix visualization
    // uses square 4 neighbourhood
    
    // calculates simple non-border cases 
    for(unsigned int y=1,index=width;y<(height - 1);y++){
      index++;
      for(unsigned int x=1;x<(width - 1);x++){
	
	umatrix[index] =  sqrt(vector_sqr_distance(som[index],som[index + 1]));
	umatrix[index] += sqrt(vector_sqr_distance(som[index],som[index - 1]));
	umatrix[index] += sqrt(vector_sqr_distance(som[index],som[index - width]));
	umatrix[index] += sqrt(vector_sqr_distance(som[index],som[index + width]));
	umatrix[index] /= 4*2;
	
	index++;
      }
      index++;
    }
    
    
    // calculates border cases.
    
    // y=0 line (no corners)
    for(unsigned int index=1;index<(width - 1);index++){
      
      umatrix[index] =  sqrt(vector_sqr_distance(som[index],som[index + 1]));
      umatrix[index] += sqrt(vector_sqr_distance(som[index],som[index - 1]));
      umatrix[index] += sqrt(vector_sqr_distance(som[index],som[width*height + index - width]));
      umatrix[index] += sqrt(vector_sqr_distance(som[index],som[index + width]));
      umatrix[index] /= 4*2;
    }
    
    // y=(height-1) line (no corners)
    for(unsigned int index=1;index<(width - 1);index++){
      
      umatrix[index + (height - 1)*width] =  sqrt(vector_sqr_distance(som[index],som[index + 1 + width*(height - 1)]));
      umatrix[index + (height - 1)*width] += sqrt(vector_sqr_distance(som[index],som[index - 1 + width*(height - 1)]));
      umatrix[index + (height - 1)*width] += sqrt(vector_sqr_distance(som[index],som[index - width + width*(height - 1)]));
      umatrix[index + (height - 1)*width] += sqrt(vector_sqr_distance(som[index],som[index + width]));
      umatrix[index + (height - 1)*width] /= 4*2;
    }
    
    // x = 0 line (no corners)
    for(unsigned int index=width;index<width*(height-1);index += width){
      
      umatrix[index] =  sqrt(vector_sqr_distance(som[index], som[index + 1]));
      umatrix[index] += sqrt(vector_sqr_distance(som[index], som[index + (width - 1)]));
      umatrix[index] += sqrt(vector_sqr_distance(som[index], som[index + width]));
      umatrix[index] += sqrt(vector_sqr_distance(som[index], som[index - width]));
      umatrix[index] /= 4*2;
      
    }
    
    // x = (width-1) line (no corners)
    for(unsigned int index=(2*width - 1);index<width*(height-1);index += width){
      
      umatrix[index] =  sqrt(vector_sqr_distance(som[index], som[index - 1]));
      umatrix[index] += sqrt(vector_sqr_distance(som[index], som[index + 1 - width]));
      umatrix[index] += sqrt(vector_sqr_distance(som[index], som[index + width]));
      umatrix[index] += sqrt(vector_sqr_distance(som[index], som[index - width]));
      umatrix[index] /= 4*2;
      
    }
    
    // calculates (0,0)
    umatrix[0] =  sqrt(vector_sqr_distance(som[0], som[width*height - 1]));
    umatrix[0] += sqrt(vector_sqr_distance(som[0], som[1]));
    umatrix[0] += sqrt(vector_sqr_distance(som[0], som[1+width]));
    umatrix[0] += sqrt(vector_sqr_distance(som[0], som[(height-1)*width]));
    umatrix[0] /= 4*2;
    
    // calculates (0, width-1)
    umatrix[width-1] =  sqrt(vector_sqr_distance(som[width-1], som[0]));
    umatrix[width-1] += sqrt(vector_sqr_distance(som[width-1], som[width-2]));
    umatrix[width-1] += sqrt(vector_sqr_distance(som[width-1], som[2*width-1]));
    umatrix[width-1] += sqrt(vector_sqr_distance(som[width-1], som[width*height-1]));
    umatrix[width-1] /= 4*2;
    
    // calculates (height-1, 0);
    
    umatrix[(height-1)*width] =  sqrt(vector_sqr_distance(som[(height-1)*width], som[(height-1)*width + 1]));
    umatrix[(height-1)*width] += sqrt(vector_sqr_distance(som[(height-1)*width], som[height*width - 1]));
    umatrix[(height-1)*width] += sqrt(vector_sqr_distance(som[(height-1)*width], som[(height-2)*width]));
    umatrix[(height-1)*width] += sqrt(vector_sqr_distance(som[(height-1)*width], som[0]));
    umatrix[(height-1)*width] /= 4*2;
    
    // calculates (height-1, width-1)
    
    umatrix[height*width-1] =  sqrt(vector_sqr_distance(som[(height-1)*width], som[height*width - 2]));
    umatrix[height*width-1] += sqrt(vector_sqr_distance(som[(height-1)*width], som[(height-1)*width]));
    umatrix[height*width-1] += sqrt(vector_sqr_distance(som[(height-1)*width], som[width-1]));
    umatrix[height*width-1] += sqrt(vector_sqr_distance(som[(height-1)*width], som[(height-1)*width - 1]));
    umatrix[height*width-1] /= 4*2;
    
    
    // draws the umatrix visualization
    // umatrix values are between 0..1
    
    dlib.clear();
    
    const unsigned int W = dlib.width();
    const unsigned int ph = dlib.height()/height;
    const unsigned int pw = dlib.width()/width;
    unsigned int color = (((unsigned int)(umatrix[0]*0xFF))*(0x010101));
    
    for(unsigned int j=0,index = 0;j<height;j++){
      for(unsigned int i=0;i<width;i++,index++){
	
	for(unsigned int y=0;y<ph;y++){
	  for(unsigned int x=0;x<pw;x++){
	    dlib[i*pw + j*ph*W + x + y*W] = color;
	  }
	}
	
	
	color = (((unsigned int)(umatrix[index]*0xFF))*(0x010101));
      }
    }
    
    
    dlib.update();
    
    return true;
  }
  
}
  
#endif 
  
  


