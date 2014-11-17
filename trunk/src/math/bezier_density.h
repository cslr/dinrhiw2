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
	bezier_density() throw();
	~bezier_density() throw();
	
	typedef typename std::vector< std::vector< std::vector<T> > >::const_iterator iterator;
	
	// calculates bezier densities
	unsigned int operator()(const std::vector< std::vector< std::vector<T> > >& data) throw();
	
	iterator begin() const throw();
	iterator end() const throw();
	
	std::vector< std::vector<T> >& operator[](unsigned int index)
	  throw(std::out_of_range);
	
	const std::vector< std::vector<T> >& operator[](unsigned int index) const
	  throw(std::out_of_range);
	
	unsigned int size() const throw();
	
	void clear() throw();
	
      private:
	
	std::vector< std::vector< std::vector<T> > > density;
	std::vector<S> bc[3]; // blending coefficients      
      };
    
  }
}


#endif

