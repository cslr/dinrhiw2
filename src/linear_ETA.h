/*
 * *linear* estimated time of arrival class
 */
#ifndef linear_ETA_h
#define linear_ETA_h

#include "ETA.h"

namespace whiteice
{

  template <typename T>
    class linear_ETA : public ETA<T>
    {
    public:
      linear_ETA();
      linear_ETA(const linear_ETA<T>& eta);
      virtual ~linear_ETA();
      
      bool start(T begin, T end) throw();
      bool update(T current) throw();
      
      // ETA in seconds when the end value will be reached
      T estimate() const throw();
      
    private:
      
      T begin_value, end_value;
      
      
      // returns time in seconds since
      // some historical origo
      double get_time() const throw();
      
      T current_eta;
      double time_start;
      double time_origo;
      
    };
}


#include "linear_ETA.cpp"


#endif
