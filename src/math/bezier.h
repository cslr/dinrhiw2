/*
 * bezier 1-d interpolation
 * nop@iki.fi
 */

#include <vector>

#ifndef bezier_h
#define bezier_h

#include "vertex.h"


namespace whiteice
{
  namespace math
  {
  
    template <typename T>
      class bezier
      {
      public:
	
	// ctor & dtor
	bezier() ;
	~bezier() ;
	
	typedef typename std::vector< vertex<T> >::const_iterator iterator;
	
	// calculates bezier curves
	unsigned int operator()(const std::vector< vertex<T> >& data) ;
	
	iterator begin() const ;
	iterator end() const ;
	
	vertex<T>& operator[](unsigned int index) ;
	const vertex<T>& operator[](unsigned int index) const ;
	unsigned int size() const ;
	
	void clear() ;
	
      private:
	
	std::vector< vertex<T> > path;
	std::vector<T> bc; // blending coefficients
      };
  }
}

  
#include "bezier.cpp"

    


#endif
