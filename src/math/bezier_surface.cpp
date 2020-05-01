
#include <vector>
#include <cmath>
#include "number.h"
#include "bezier_surface.h"


#ifndef bezier_surface_cpp
#define bezier_surface_cpp

namespace whiteice
{
  namespace math
  {

    /**************************************************/
    
    template <typename T, typename S>
    bezier_surface<T,S>::bezier_surface() { }
    
    
    template <typename T, typename S>
    bezier_surface<T,S>::~bezier_surface() { }
    
    /***************************************************/
    
    template <typename T, typename S>
    unsigned int
    bezier_surface<T,S>::operator()(const std::vector< std::vector<T> >& data)
      
    {
      if(!data.size()) return 0;
      if(!data[0].size()) return 0;
      
      const unsigned int N = data.size(); // shorthand
      const unsigned int M = data[0].size();
      
      // solve/find out: good constant value?/good non-uniform values?
      const unsigned int frames = 1000;
      
      bc[0].resize(data.size());
      bc[1].resize(data.size());
      surface.resize(frames);
      
      /* calculates blending function coefficients */
      
      bc[0][0] = (S)(1); // C(n,0)
      bc[1][0] = (S)(1); // C(m,0)
      
      for(unsigned int k=1;k<N;k++)
	bc[0][k] = k*(N-k-1)*bc[1][k-1]; // C(n,k)
      
      for(unsigned int k=1;k<M;k++)
	bc[1][k] = k*(M-k-1)*bc[1][k-1]; // C(m,k)
      
      /* calculates blending function values
       * which are independent from inner loop
       */
      
      std::vector<S> bv;
      bv.resize(frames);
      
      // calculates v blending functions
      for(unsigned int j=0;j<frames;j++){
	double v = (double)j/(double)frames;
	
	bv[j] = bc[1][j]*
	  pow(v, (double)j)*
	  pow((1-v),(double)(M-j));
      }
      
      
      /* calculates surface */
      
      for(unsigned int j=0;j<frames;j++){
	
	for(unsigned int i=0;i<frames;i++){
	  double u = (double)i/(double)frames;
	  
	  T puv = (S)0;
	  
	  
	  for(unsigned int l=0;l<M;l++)
	    for(unsigned int k=0;k<N;k++){ // note: pow()s are slow
	      S bu = bc[k] * pow(u, (double)k) * pow((1-u),(double)(N-k));
	      
	      puv += data[l][k] * bu * bv[l];
	    }
	  
	  surface[j].push_back(puv);
	}
      }
      
      return surface.size();
    }
    
    
    /***************************************************/
    
    template <typename T, typename S>
    typename bezier_surface<T,S>::iterator
    bezier_surface<T,S>::begin() const 
    {
      return surface.begin();
    }
    
    template <typename T, typename S>
    typename bezier_surface<T,S>::iterator
    bezier_surface<T,S>::end() const 
      
    {
      return surface.end();
    }
  
  
    /***************************************************/
    
    template <typename T, typename S>
    std::vector<T>& bezier_surface<T,S>::operator[](unsigned int index)
      
    {
      return surface[index];
    }
  
    template <typename T, typename S>
    const std::vector<T>& bezier_surface<T,S>::operator[](unsigned int index)
      const 
    {
      return surface[index];
    }
  
    template <typename T, typename S>
    unsigned int bezier_surface<T,S>::size() const 
    {
      return surface.size();
    }
    
    
    template <typename T, typename S>
    void bezier_surface<T,S>::clear() 
    {
      surface.clear();
    }
    
  }
}
  
#endif



