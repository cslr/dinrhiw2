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
	bezier() throw();
	~bezier() throw();
	
	typedef typename std::vector< vertex<T> >::const_iterator iterator;
	
	// calculates bezier curves
	unsigned int operator()(const std::vector< vertex<T> >& data) throw();
	
	iterator begin() const throw();
	iterator end() const throw();
	
	vertex<T>& operator[](unsigned int index) throw(std::out_of_range);
	const vertex<T>& operator[](unsigned int index) const throw(std::out_of_range);
	unsigned int size() const throw();
	
	void clear() throw();
	
      private:
	
	std::vector< vertex<T> > path;
	std::vector<T> bc; // blending coefficients
      };
  }
}

  
#include "bezier.cpp"

    


#endif
