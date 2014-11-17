

#ifndef linear_ETA_cpp
#define linear_ETA_cpp

#include "linear_ETA.h"
#include <sys/time.h>
#include <time.h>

namespace whiteice
{

  template <typename T>
  linear_ETA<T>::linear_ETA()
  {
    time_origo = 0.0;
    time_origo = get_time();
  }
  
  
  template <typename T>
  linear_ETA<T>::linear_ETA(const linear_ETA<T>& eta)
  {
    // this is just direct copy
    
    this->begin_value = eta.begin_value;
    this->end_value = eta.end_value;
    
    this->current_eta = eta.current_eta;
    this->time_start = eta.time_start;
    this->time_origo = eta.time_origo;
  }
  
  
  template <typename T>
  linear_ETA<T>::~linear_ETA(){ }
  
  
  template <typename T>
  bool linear_ETA<T>::start(T begin, T end) throw()
  {
    time_start = get_time();
    begin_value = begin;
    end_value = end;
    current_eta = 0.0;
    
    if(begin_value == end_value){
      end_value++;
      return false;
    }
    
    return true;
  }
  
  
  template <typename T>
  bool linear_ETA<T>::update(T current) throw()
  {
    double current_time = get_time();
    
    if(current - begin_value > T(0.0))
      current_eta = T(current_time - time_start) * 
	(T(end_value - current) / T(current - begin_value));
    
    return true;
  }
  
  
  // ETA in seconds when the end value will be reached
  template <typename T>
  T linear_ETA<T>::estimate() const throw()
  {
    return current_eta;
  }
  
  
  /**************************************************/
  
  template <typename T>
  double linear_ETA<T>::get_time() const throw()
  {
    struct timeval  tv;
    struct timezone tz;
    
    if(gettimeofday(&tv, &tz) == -1) return -1.0;
    
    double t = ((double)tv.tv_sec + (double)tv.tv_usec/1000000.0);
    t -= time_origo; // keeps values small (-> smaller floating point calculation errors)
    
    return t;
  }

}


#endif
