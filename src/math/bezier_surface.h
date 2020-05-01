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
      bezier_surface() ;
      ~bezier_surface() ;
      
      typedef typename std::vector< std::vector<T> >::const_iterator iterator;
      
      // calculates bezier surfaces
      unsigned int operator()(const std::vector< std::vector<T> >& data) ;
      
      iterator begin() const ;
      iterator end() const ;
      
      std::vector<T>& operator[](unsigned int index)
	;

      const std::vector<T>& operator[](unsigned int index) const
	;

      unsigned int size() const ;
      
      void clear() ;
      
    private:
      
      std::vector< std::vector<T> > surface;
      std::vector<S> bc[2]; // blending coefficients
      
      
    };
    
  }
}

#include "bezier_surface.cpp"


#endif
