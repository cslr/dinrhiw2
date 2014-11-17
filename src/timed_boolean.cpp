/*
 * timed_boolean variables timing:
 *  assumption about semantics of setitimer():
 *  old timer will be removed when / 
 *  if new timeout will be set.
 *
 * implementation: timed_booleans register to
 * static "global" itimer handling system which
 * uses timer and priority queue to change state
 * of timed_boolean, where next timeout is time of
 * the shortest remaining wait time.
 */

#include <stdexcept>
#include <time.h>
#include <sys/time.h>
// #include <sys/times.h> not needed(?)
#include "timed_boolean.h"
#include "augmented_data.h"

#include "config.h"



/********************************************************************************/

#ifndef HAVE_GETTIMEOFDAY
#ifdef WINNT

#include <windows.h>

extern "C" {
int gettimeofday(struct timeval* tv, void* tz){
  FILETIME ft;
  
  if(tv == 0) return -1;
  GetSystemTimeAsFileTime(&ft);
  
  // GetSystemTimeAsFileTime()
  // returns 64 bit ticks counter (100 ns) since
  // 00:00:00 UTC, January 1, 1601
  // gettimeofday() returns microsec ticks since 
  // 00:00:00 UTC, January 1, 1970 (Unix Epoch)
  // 
  // difference in milliseconds is 11644473600000L (from web)
  
  unsigned long long ticks = 0; // gcc specific
  ((unsigned int*)&ticks)[0] = ft.dwLowDateTime;
  ((unsigned int*)&ticks)[1] = ft.dwHighDateTime;
  
  // moves origo to Unix Epoch
  ticks -= 116444736000000000ULL; // 100ns tick length
  
  ticks /= 10ULL; // convers to microsecs
  
  tv->tv_sec  = (time_t)(ticks / 1000000ULL); // in secs
  tv->tv_usec = (unsigned int)(ticks % 1000000ULL);
  
  return 0; // OK = 0 , -1 = FAILURE
}
};

#else
#error "No gettimeofday() implementation for this platform."
#endif

#endif



namespace whiteice
{

  timed_boolean::timed_boolean(double wtime, bool initial)
  {
    variable = initial;
    inverted = false;
    double t0;

    if(!get_time(t0))
      throw std::runtime_error("timed_boolean cannot get current time");
    
    if(wtime > 0){
      t1 = t0 + wtime;
    }
    else{
      inverted = true;
    }    
  }
  
  timed_boolean::timed_boolean(const timed_boolean& tb)
  {
    variable = tb.variable;
    inverted = tb.inverted;
    t1 = tb.t1;
  }
  
  timed_boolean::~timed_boolean(){ }
  
  
  timed_boolean& timed_boolean::operator=(const bool bValue)
  {
    variable = bValue;
    return (*this);
  }
  
  
  timed_boolean& timed_boolean::operator=(const timed_boolean& tb)
  {
    variable = tb.variable;
    inverted = tb.inverted;
    t1 = tb.t1;
    return (*this);
  }
  
  
  bool timed_boolean::operator==(const bool bValue) const throw()
  {
    update();
    return (bValue == variable);
  }
  
  bool timed_boolean::operator!=(const bool bValue) const throw()
  {
    update();
    return (bValue != variable);
  }
  
  
  bool timed_boolean::operator==(const timed_boolean& b) const throw()
  {
    this->update();
    b.update();
    return(this->variable == b.variable);
  }
  
  bool timed_boolean::operator!=(const timed_boolean& b) const throw()
  {
    this->update();
    b.update();
    return(this->variable == b.variable);
  }
  
  
  bool timed_boolean::operator!() const throw()
  {
    update();
    return (!variable);
  }
  
  
  void timed_boolean::update() const throw()
  {
    if(inverted) return;
    
    if(time_left() <= 0){
      variable = !variable;
      inverted = true;
    }
  }
  
  
  double timed_boolean::time_left() const throw()
  {
    double t;
    
    if(!get_time(t))
      return -1.0;
    
    double result = t - t1;
    
    if(result <= 0)
      return 0.0;
    else
      return result;
  }
  
  
  /*
   * returns real world time
   */
  bool timed_boolean::get_time(double& t) const throw()
  {  
    struct timeval t1;
    gettimeofday(&t1,0);
    t = ( (double)t1.tv_sec + ((double)t1.tv_usec)*0.000001 );
    return true;
  }
  
  /********************************************************************************/
  
  std::ostream& operator<<(std::ostream& ios,
			   const whiteice::timed_boolean& t)
  {
    if(t == true) ios << true;
    else ios << false;
    
    return ios;
  }
}





