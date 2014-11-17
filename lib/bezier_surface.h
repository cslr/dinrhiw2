/*
 * 2d bezier surface
 * nop@iki.fi
 */

#include <vector>

#ifndef bezier_surface_h
#define bezier_surface_h

namespace whiteice
{
  namespace math
  {

    template <typename T, typename S>
      class bezier_surface
    {
    public:
      
      // ctor & dtor
      bezier_surface() throw();
      ~bezier_surface() throw();
      
      typedef typename std::vector< std::vector<T> >::const_iterator iterator;
      
      // calculates bezier surfaces
      unsigned int operator()(const std::vector< std::vector<T> >& data) throw();
      
      iterator begin() const throw();
      iterator end() const throw();
      
      std::vector<T>& operator[](unsigned int index)
	throw(std::out_of_range);

      const std::vector<T>& operator[](unsigned int index) const
	throw(std::out_of_range);

      unsigned int size() const throw();
      
      void clear() throw();
      
    private:
      
      std::vector< std::vector<T> > surface;
      std::vector<S> bc[2]; // blending coefficients
      
      
    };
    
  }
}

#include "bezier_surface.cpp"


#endif
