
#include <iostream>
#include <vector>
#include <pthread.h>
#include "component.h"


extern "C" { static void* __component_thread_init(void *component_ptr); };


namespace whiteice
{
  namespace digital_logic
  {
    
    component::component()
    {
      pthread_mutex_init(&state_lock, 0);
      pthread_cond_init (&state_change, 0);
      
      ok = true;
      state = false;
      counter = 1;
      
      pthread_create(&component_thread, 0, 
		     __component_thread_init,
		     (void*)this);
    }
    
    
    component::~component()
    {
      // cancels component thread
      ok = false;
      pthread_cond_signal( &state_change );
      pthread_cancel( component_thread );
      
      
      pthread_mutex_destroy( &state_lock );
      pthread_cond_destroy ( &state_change );
    }
    
    
    bool component::register_input(component* c)
    {
      pthread_mutex_lock( &state_lock );
      try{
	in.push_back(c);
	
	pthread_mutex_unlock( &state_lock );
	return true;
      }
      catch(std::exception& e){
	pthread_mutex_unlock( &state_lock );
	return false;
      }
    }
    
    
    bool component::unregister_input(component* c)
    {
      pthread_mutex_lock( &state_lock );
      try{
	
	if(in.size() <= 0){
	  pthread_mutex_unlock( &state_lock );
	  return false;
	}
	  
	bool removeMade = false;
	std::vector<component*>::iterator i;
	i = in.end();
	
	do{
	  i--;
	  
	  if((*i) == c){
	    in.erase(i);
	    removeMade = true;
	    break;
	  }
	}
	while(i != in.begin());
	
	pthread_mutex_unlock( &state_lock );
	return removeMade;
      }
      catch(std::exception& e){
	pthread_mutex_unlock( &state_lock );
	return false;
      }      
    }
    
    
    bool component::register_output(component* c)
    {
      pthread_mutex_lock( &state_lock );
      try{
	// all outputs are same so each component
	// can/should register only once
	
	std::vector<component*>::iterator i;
	i = out.begin();
	
	while(i != out.end()){
	  if((*i) == c){
	    pthread_mutex_unlock( &state_lock );
	    return false;
	  }
	  
	  i++;
	}
	
	
	out.push_back(c);
	
	pthread_mutex_unlock( &state_lock );
	return true;
      }
      catch(std::exception& e){
	pthread_mutex_unlock( &state_lock );
	return false;
      }
    }
    
    
    bool component::unregister_output(component* c)
    {
      pthread_mutex_lock( &state_lock );
      try{
	
	std::vector<component*>::iterator i;
	i = out.begin();
	
	while(i != out.end()){
	  if((*i) == c){
	    out.erase(i);
	    
	    pthread_mutex_unlock( &state_lock );
	    return true;
	  }
	  
	  i++;
	}
	
	pthread_mutex_unlock( &state_lock );
	return false;
      }
      catch(std::exception& e){
	pthread_mutex_unlock( &state_lock );
	return false;
      }      
    }
    
    
    bool component::get() const throw()
    {
      bool s;
      pthread_mutex_lock( &state_lock );
      s = state;
      pthread_mutex_unlock( &state_lock );
      
      return s;
    }
    
    // signals component that
    // one of it's output signals has changed
    void component::signal() throw()
    {
      pthread_cond_signal( &state_change );
    }
    
    
    
    unsigned int component::activation() const throw()
    {
      return counter;
    }
    
    
    const std::vector<component*>& component::inc() const throw(){
      return in;
    }
    
    
    void component::threadloop()
    {
      
      while(ok){
	pthread_mutex_lock( &state_lock );
	
	if(in.size() != 0)
	  ok = ok && update();
	else
	  state = false;
	
	
	// wakes up every output component
	std::vector<component*>::iterator i = out.begin();
	  
	while(i != out.end()){
	  (*i)->signal();
	  i++;
	}
	
	counter++;
	
	pthread_cond_wait( &state_change, &state_lock );
	pthread_mutex_unlock( &state_lock );
      }
    }
    
    
  };
};


extern "C" {
  void* __component_thread_init(void *component_ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(component_ptr)
      ((whiteice::digital_logic::component*)component_ptr)->threadloop();
    
    pthread_exit(0);
  }
};

    
