
#ifndef hermite_cpp
#define hermite_cpp

#include <vector>
#include <stdexcept>
#include <cassert>
#include <typeinfo>

#include "hermite.h"
#include "blade_math.h"



namespace whiteice
{
  namespace math
  {

    /* hermite ctor */
    template <typename T, typename S>
    hermite<T,S>::hermite() throw()
    {
      curvetype = undefined_curve;
      derivate_parameter = (S)(0.5);
    }
    
    
    
    /* empty dtor */
    template <typename T, typename S>
    hermite<T,S>::~hermite() throw(){ }
    
    
    
    template <typename T, typename S>
    bool hermite<T,S>::set_cardinal_parameter(const S& a) throw()
    {
      derivate_parameter = a;
      curvetype = cardinal_spline_curve;    
      return true;
    }
    
    
    template <typename T, typename S>
    int hermite<T,S>::calculate(const std::vector<T>& data,
				bool clear_path,
				curve_types ct)
      throw(std::domain_error)
    {
      if(data.size() == 0) return 0;    
      curvetype = ct;
      
      if(curvetype == catmull_rom_spline_curve)
	derivate_parameter = (S)0.5;
      
      
      if(curvetype == catmull_rom_spline_curve ||
	 curvetype == cardinal_spline_curve){
	
	cardinal_spline(data, clear_path);
      }
      else return 0;
      
      return path.size();
    }
    
    
    template <typename T, typename S>
    int hermite<T,S>::operator()(const std::vector<T>& data,
				 bool clear_path,
				 curve_types ct)
      throw(std::domain_error)
    {
      return calculate(data, clear_path, ct);
    }
    
    
    
    template <typename T, typename S>
    int hermite<T,S>::calculate(const std::vector<T>& data,
				int MAX_FRAMES,
				bool clear_path,
				curve_types ct)
      throw(std::domain_error)
    {
      if(data.size() == 0) return 0;    
      curvetype = ct;
      
      if(curvetype == catmull_rom_spline_curve)
	derivate_parameter = (S)0.5;
      
      
      if(curvetype == catmull_rom_spline_curve ||
	 curvetype == cardinal_spline_curve){
	
	cardinal_spline(data, MAX_FRAMES, clear_path);
      }
      else return 0;
      
      return path.size();
    }
    
    
    template <typename T, typename S>
    int hermite<T,S>::operator()(const std::vector<T>& data,
				 int MAX_FRAMES,
				 bool clear_path,
				 curve_types ct)
      throw(std::domain_error)
    {
      return calculate(data, MAX_FRAMES, clear_path, ct);
    }
    
    
    
    template <typename T, typename S>
    typename hermite<T,S>::iterator hermite<T,S>::begin() const throw(){
      return path.begin();
    }
    
    template <typename T, typename S>
    typename hermite<T,S>::iterator hermite<T,S>::end() const throw(){
      return path.end();
    }
    
    
    template <typename T, typename S>
    T& hermite<T,S>::operator[](unsigned int index)
      throw(std::out_of_range)
    {
      if(index >= path.size())
	throw std::out_of_range("hermite::operator[] - index out of range");
      return path[index];
    }
    
    
    template <typename T, typename S>
    const T& hermite<T,S>::operator[](unsigned int index) const
      throw(std::out_of_range)
    {
      if(index >= path.size())
	throw std::out_of_range("hermite::operator[] - index out of range");
      return path[index];
    }
    
    
    template <typename T, typename S>
    unsigned int hermite<T,S>::size() const throw()
    {
      return path.size();
    }
    
    
    
    template <typename T, typename S>
    void hermite<T,S>::cardinal_spline(const std::vector<T>& data,
				       bool clear_path)
      throw(std::domain_error)
    {
      cardinal_spline(data, data.size()*10, clear_path);
    }
  
    
    /*
     * generates cardinal spline
     * clear_path - creates new path/don't add to existing one
     */
    template <typename T, typename S>
    void hermite<T,S>::cardinal_spline(const std::vector<T>& data,
				       int MAX_FRAMES,
				       bool clear_path)
      throw(std::domain_error)
    {
      if((unsigned)MAX_FRAMES < data.size()*2)
	throw std::domain_error("hermite::cardinal_spline(): not enough frames.");
      
      if(clear_path) path.clear();
      
      int framesPerLine = (int)(((float)MAX_FRAMES) / ((float)data.size()));
      
      
      T delta0,delta1;
      
      for(unsigned int i=0;i<data.size();i++)
      {
	// calculates deltas and passes
	// them to hermite rutin which
	// interpolates between points
	
	if(i > 0 && i < data.size()-1){
	  
	  delta0 = derivate_parameter * 
	    (data[i+1] - data[i-1]);	
	}
	else if(i >= data.size()-1){
	  
	    delta0 = derivate_parameter * 
	      (data[0] - data[i-1]);
	}
	else{
	  
	  delta0 = derivate_parameter * 
	    (data[i+1] - data[data.size()-1]);
	}
	
	if(i < data.size()-2){
	  delta1 = derivate_parameter *
	    (data[i+2] - data[i]);
	}
	else{
	  delta1 = derivate_parameter *
	    (data[i+2 - data.size()] - data[i]);
	}
	
	if(i < data.size()-1){
	  
	  hermite_line(data[i], delta0,
		       data[i+1], delta1,
		       framesPerLine,
		       false, false);
	}
	else{
	  hermite_line(data[i], delta0,
		       data[i+1-data.size()], delta1,
		       framesPerLine,
		       false, false);
	}
      }
      
    }
    
    
    /*
     * clear_path - clears path if true
     */  
    template <typename T, typename S>
    void hermite<T,S>::kb_spline(bool clear_path)
      throw(std::domain_error)
    {
      using namespace std;
      
      assert(0); // not implemented (yet)
    }
    
    
    
    /*
     * creates hermite interpolation
     * from p0 to p1
     * gradient changes from delta0 to delta1
     * if plot_last_one is false
     *   last - p1 is not added to path
     *   (circular etc.)
     * clear_path - if true creates new path
     */
    template <typename T, typename S>
    void hermite<T,S>::hermite_line(const T& p0, const T& delta0,
				    const T& p1, const T& delta1,
				    bool clear_path,
				    bool plot_last_one)
      throw(std::domain_error)
    {
      
      if(clear_path) path.clear();
      
      T p, pn;
      S len = (S)whiteice::math::sqrt( (p1-p0)*(p1-p0) );
      
      
      /* length per frame, note: change to form f[i] = p[i].frame.
       * frames = f[i+1] - f[i]
       */
      
      S lpf = 0.25;
      
      int TOTAL_FRAMES = (int)(len/lpf);
      if(TOTAL_FRAMES < 2) TOTAL_FRAMES = 2;
      
      int MAX_FRAMES = TOTAL_FRAMES;
      
      if(plot_last_one == false) MAX_FRAMES--;      
      
      p = p0;
      path.push_back(p);  
      
      for(int t=1;t<=MAX_FRAMES;t++){
	
	p = pn;    
      
	S s = (S)t / (S)TOTAL_FRAMES;
	S h1 =  2*s*s*s - 3*s*s + 1;
	S h2 = -2*s*s*s + 3*s*s;
	S h3 =    s*s*s - 2*s*s + s;
	S h4 =    s*s*s -   s*s;
	
	pn = h1*p0 + h2*p1 + h3*delta0 + h4*delta1;
	
	path.push_back(pn);
      }
      
    }
    
    
    /*
     * create hermite interpolation from p0 to p1
     * this time with using given amount of frames MAX_FRAMES
     */
    template <typename T, typename S>
    void hermite<T,S>::hermite_line(const T& p0, const T& delta0,
				    const T& p1, const T& delta1,
				    int MAX_FRAMES,
				    bool clear_path,
				    bool plot_last_one)
      throw(std::domain_error)
    {
      if(MAX_FRAMES <  2)
	throw std::domain_error("hermite::hermite_line() number of frames <= 0");
      
      if(clear_path) path.clear();
      
      T p, pn;
      // S len = (S)blade_math::sqrt( (p1-p0)*(p1-p0) );
      
      int TOTAL_FRAMES = MAX_FRAMES;
      
      if(plot_last_one == false) MAX_FRAMES--;      
      
      p = p0;
      path.push_back(p);  
      
      for(int t=1;t<=MAX_FRAMES;t++){
	p = pn;
	
	S s = (S)t / (S)TOTAL_FRAMES;
	S h1 =  2*s*s*s - 3*s*s + 1;
	S h2 = -2*s*s*s + 3*s*s;
	S h3 =    s*s*s - 2*s*s + s;
	S h4 =    s*s*s -   s*s;
	
	pn = h1*p0 + h2*p1 + h3*delta0 + h4*delta1;
	
	path.push_back(pn);
      }
      
      
    }
    
    
  }  
}


#endif



