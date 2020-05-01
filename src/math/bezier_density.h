/*
 * 3d bezier density volume
 * nop@iki.fi
 */

#include <vector>

#ifndef bezier_density_h
#define bezier_density_h

namespace whiteice
{
  namespace math
  {
    template <typename T, typename S>
      class bezier_density
      {
      public:
	
	// ctor & dtor
	bezier_density() ;
	~bezier_density() ;
	
	typedef typename std::vector< std::vector< std::vector<T> > >::const_iterator iterator;
	
	// calculates bezier densities
	unsigned int operator()(const std::vector< std::vector< std::vector<T> > >& data) ;
	
	iterator begin() const ;
	iterator end() const ;
	
	std::vector< std::vector<T> >& operator[](unsigned int index)
	  ;
	
	const std::vector< std::vector<T> >& operator[](unsigned int index) const
	  ;
	
	unsigned int size() const ;
	
	void clear() ;
	
      private:
	
	std::vector< std::vector< std::vector<T> > > density;
	std::vector<S> bc[3]; // blending coefficients      
      };
    
  }
}


#endif

