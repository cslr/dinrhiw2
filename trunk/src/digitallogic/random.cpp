
#include "random.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>


extern "C" { static void* __randomwakeup_thread_init(void *component_ptr); };


namespace whiteice
{
  namespace digital_logic
  {
    
    random::random(float t0, float t1)
    {
      this->t0 = t0;
      this->t1 = t1;
      
      // calculates next wake up time
      
      calculate_wakeup();      
      
      // starts signaling thread
      wakeup_running = true;
      pthread_create( &wakeup_thread, 0, 
		      __randomwakeup_thread_init,
		      (void*)this);
    }
    
    
    random::~random()
    {
      wakeup_running = false;
      pthread_cancel(wakeup_thread); // kill
    }
    
    
    // recalculates output from inputs
    bool random::update() throw()
    {
      state = (bool)(rand() & 1);
      return true;
    }
    
    
    
    void random::calculate_wakeup()
    {
      wakeup_time = get_time();
      
      wakeup_time += (t1 - t0) *
	(((float)rand())/((float)RAND_MAX));
    }
    
    
    void random::wakeuploop()
    {
      struct timespec req;
      struct timespec rem;
      
      while(wakeup_running){
	// sleeps half the remaining time or >= 1 ms
	
	while(1){
	  float t = get_time();
	  
	  if(t > wakeup_time)
	    break;
	  
	  float sleeptime = (wakeup_time - t)/2.0f;
	  
	  if(sleeptime <= 0.001f)
	    sleeptime = 0.001f;
	  
	  req.tv_sec  = (time_t)floorf(sleeptime);
	  req.tv_nsec = (long)(1000000000.0f *
			       (sleeptime - floorf(sleeptime)));
	  
	  nanosleep(&req, &rem);
	}
	
	
	signal();
	
	calculate_wakeup();
      }
    }
    
    
    
    float random::get_time() const throw()
    {
      return ((float)clock() / ((float)CLOCKS_PER_SEC));
    }
    
    
  };
};


extern "C" {
  void* __randomwakeup_thread_init(void *component_ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(component_ptr)
      ((whiteice::digital_logic::random*)component_ptr)->wakeuploop();
    
    pthread_exit(0);
  }
};
