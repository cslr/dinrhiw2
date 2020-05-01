/*
 * hermite / fergusson 3rd order interpolation
 * nop@iki.fi
 *
 * todo: constant speed / controlling speed of interpolation
 *  - don't save data to own buffer
 */


#ifndef hermite_h
#define hermite_h

#include <vector>
#include <stdexcept>
#include "number.h"


namespace whiteice
{
  namespace math
  {

    /*
     * T - type of interpolation
     * S - scalar type
     */
    template <typename T, typename S>
      class hermite
      {	
      public:
	
	enum curve_types
	{
	  undefined_curve,
	  hermite_curve, cardinal_spline_curve,
	  catmull_rom_spline_curve, kb_spline_curve
	};
	
	typedef typename std::vector<T>::const_iterator iterator;
	
	
	hermite() ;
	~hermite() ;  
	
	bool set_cardinal_parameter(const S& a) ;
	
	
	int calculate(const std::vector<T>& data,
		      bool clear_path = true,
		      curve_types = catmull_rom_spline_curve)
	  ;
	
	int operator()(const std::vector<T>& data,
		       bool clear_path = true,
		       curve_types = catmull_rom_spline_curve)
	  ;
	
	// calculate() using given number of frames MAX_FRAMES
	int calculate(const std::vector<T>& data,
		      int MAX_FRAMES,
		      bool clear_path = true,
		      curve_types = catmull_rom_spline_curve)
	  ;
	
	// operator()  using given number of frames MAX_FRAMES
	int operator()(const std::vector<T>& data,
		       int MAX_FRAMES,
		       bool clear_path = true,
		       curve_types = catmull_rom_spline_curve)
	  ;
	
	
	iterator begin() const ; // iterator      
	iterator end() const ;
	
	T& operator[](unsigned int index) ;  
	const T& operator[](unsigned int index) const ;  
	unsigned int size() const ;
	
      private:
	
	void cardinal_spline(const std::vector<T>& data,
			     bool clear_path = true) ;
	
	// cardinal spline with given number of frames
	void cardinal_spline(const std::vector<T>& data,
			     int MAX_FRAMES,
			     bool clear_path = true) ;
	
	
	// not implemented
	void kb_spline(bool clear_path = true) ;
	
	
	void hermite_line(const T& p0, const T& delta0,
			  const T& p1, const T& delta1,
			  bool clear_path=true,
			  bool plot_last_one=true) ;
	
	// hermite line with given number of frames
	void hermite_line(const T& p0, const T& delta0,
			  const T& p1, const T& delta1,
			  int MAX_FRAMES,
			  bool clear_path,
			  bool plot_last_one) ;
	
	std::vector<T> path;
	S derivate_parameter; // cardinal splines  
	
	curve_types curvetype;
      };
    
  }
}

#include "hermite.cpp"


#endif



