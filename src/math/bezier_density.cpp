

#include <vector>
#include <cmath>
#include "number.h"
#include "bezier_density.h"

#ifndef bezier_density_cpp
#define bezier_density_cpp

namespace whiteice
{
  namespace math
  {
  
    
    /**************************************************/
    
    template <typename T, typename S>
    bezier_density<T,S>::bezier_density() throw(){ }
    
    
    template <typename T, typename S>
    bezier_density<T,S>::~bezier_density() throw(){ }
    
    /***************************************************/
    
    template <typename T, typename S>
    unsigned int bezier_density<T,S>::operator()(const std::vector< std::vector< std::vector<T> > >& data) throw()
    {
      if(!data.size()) return 0;
      if(!data[0].size()) return 0;
      if(!data[0][0].size()) return 0;
    
      // assumes input data is correct
      const unsigned int N = data.size(); // shorthand
      const unsigned int M = data[0].size();
      const unsigned int L = data[0][0].size(); 
      
      // solve/find out: good constant value?/good non-uniform values?
      const unsigned int frames = 1000;
      
      for(unsigned int i=0;i<3;i++)
	bc[i].resize(data.size());
      
      density.resize(frames);
      
      for(unsigned int i=0;i<frames;i++)
	density[i].resize(frames);
      
      
      /* calculates blending function coefficients */
      
      for(unsigned int i=0;i<3;i++)
	bc[i][0] = (S)(1); // C(n,0)
      
      for(unsigned int k=1;k<N;k++)
	bc[0][k] = k*(N-k-1)*bc[0][k-1]; // C(n,k)
      
      for(unsigned int k=1;k<M;k++)
	bc[1][k] = k*(M-k-1)*bc[1][k-1]; // C(m,k)
      
      for(unsigned int k=1;k<L;k++)
	bc[2][k] = k*(L-k-1)*bc[2][k-1]; // C(l,k)
      
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
      
      std::vector<S> bw;
      bw.resize(frames);
      
      // calculates w blending functions
      for(unsigned int k=0;k<frames;k++){
	
	double w = (double)k/(double)frames;
	
	bw[k] = bc[2][k]*
	  pow(w, (double)k)*
	  pow((1-w),(double)(L-k));
      }
      
      
      /* calculates surface */
      
      for(unsigned int k=0;k<frames;k++){
	for(unsigned int j=0;j<frames;j++){
	  for(unsigned int i=0;i<frames;i++){
	    double u = (double)i/(double)frames;
	    
	    T puvw = (S)0;
	    
	    for(unsigned int p=0;p<L;p++)
	      for(unsigned int n=0;n<M;n++)
		for(unsigned int m=0;m<N;m++){ // pow()s are slow
		  S bu = bc[k] * pow(u, (double)m) * pow((1-u),(double)(N-m));
		  
		  
		  // DON'T KNOW WHAT IS CORRECT VALUE FOR: l
		  // puvw += data[p][l][k] * bu * bv[n] * bw[p];
		  puvw += data[p][i][k] * bu * bv[n] * bw[p];
		}
	    
	    density[k][j].push_back(puvw);
	  }
	}
      }
      
      return density.size();
    }
    
    
    /***************************************************/
    
    template <typename T, typename S>
    typename bezier_density<T,S>::iterator bezier_density<T,S>::begin()
      const throw()
    {
      return density.begin();
    }
  
    template <typename T, typename S>
    typename bezier_density<T,S>::iterator bezier_density<T,S>::end()
      const throw()
    {
      return density.end();
    }

    
    /***************************************************/
    
    template <typename T, typename S>
    std::vector< std::vector<T> >&
    bezier_density<T,S>::operator[](unsigned int index)
      throw(std::out_of_range)
    {
      return density[index];
    }
  
    
    template <typename T, typename S>
    const std::vector< std::vector<T> >&
    bezier_density<T,S>::operator[](unsigned int index)
      const throw(std::out_of_range)
    {
      return density[index];
    }
    
    template <typename T, typename S>
    unsigned int bezier_density<T,S>::size() const throw()
    {
      return density.size();
    }
    
    
    template <typename T, typename S>
    void bezier_density<T,S>::clear() throw()
    {
      density.clear();
    }
    
    
  }
}

  
#endif

