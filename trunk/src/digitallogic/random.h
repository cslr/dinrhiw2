/*
 * random input changes its
 * value randomly and evenly
 * distributedly every [t0..t1] seconds
 */

#ifndef whiteice__random__component_h
#define whiteice__random__component_h

#include "component.h"
#include <pthread.h>


namespace whiteice
{
  namespace digital_logic
  {
    
    class random : public component
    {
    public:
      random(float t0, float t1);
      virtual ~random();
      
    protected:
      // recalculates output from inputs
      bool update() throw();
      
      
    private:
      void calculate_wakeup();
      
      float t0, t1;
      float wakeup_time;
      
      pthread_t wakeup_thread;
      
    public:
      void wakeuploop();
      
    private:
      bool wakeup_running;
      
      // returns current time in
      // seconds since some (unknown)
      // point in history
      float get_time() const throw();
    };
    
  };
};

#endif
