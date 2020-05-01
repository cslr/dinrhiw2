

#include <vector>
#include <cmath>
#include "number.h"
#include "bezier.h"

#ifndef bezier_cpp
#define bezier_cpp

namespace whiteice
{
  namespace math
  {
    
    /**************************************************/
    
    template <typename T>
    bezier<T>::bezier() { }
    
    
    template <typename T>
    bezier<T>::~bezier() { }
    
    /***************************************************/
    
    template <typename T>
    unsigned int bezier<T>::operator()(const std::vector< vertex<T> >& data) 
    {
      if(data.size() <= 0) return 0;

      // solve/find out: good constant value?/good non-uniform values?
      const unsigned int frames = 1000;
      const unsigned int N = data.size(); // shorthand
      const unsigned int D = data[0].size(); // dimension of input vectors

      bc.resize(N);
      
      /* calculates blending function coefficients */
      
      bc[0] = T(1.0f); // C(n,0)
      for(unsigned int k=1;k<N;k++)
	bc[k] = k*(N-k-1)*bc[k-1]; // C(n,k)
      
      for(unsigned int i=0;i<frames;i++){
	double u = (double)i/(double)frames;
	
	vertex<T> pu(D);
	pu = T(0.0f);
	
	for(unsigned int k=0;k<N;k++) // pow()s are slow
	  pu += data[k] * bc[k] *
	    pow(u, (double)k) * 
	    pow((1-u),(double)(N-k));
	
	path.push_back(pu);
      }
      
      return path.size();
    }
    
    
    /***************************************************/
    
    template <typename T>
    typename bezier<T>::iterator bezier<T>::begin() const 
    {
      return path.begin();
    }
    
    template <typename T>
    typename bezier<T>::iterator bezier<T>::end() const 
    {
      return path.end();
    }
    
    
    /***************************************************/
    
    template <typename T>    
    vertex<T>& bezier<T>::operator[](unsigned int index)
      
    {
      return path[index];
    }
    
    template <typename T>
    const vertex<T>& bezier<T>::operator[](unsigned int index) const
      
    {
      return path[index];
    }
    
    template <typename T>
    unsigned int bezier<T>::size() const 
    {
      return path.size();
    }
    
    template <typename T>
    void bezier<T>::clear() 
    {
      path.clear();
    }
    
    
  }
}
  

#endif


