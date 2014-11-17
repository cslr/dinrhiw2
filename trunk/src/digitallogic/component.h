/*
 * component framework
 *
 * every component has single thread which waits that
 * someone wakes it up, it then updates it's outputs
 * and starts to wait state change again.
 * (when it's one of waker's outputs and changes)
 * 
 */

#ifndef whiteice__digital_logic__component_h
#define whiteice__digital_logic__component_h

#include <vector>
#include <pthread.h>


namespace whiteice
{
  namespace digital_logic
  {
    
    class component
    {
    public:
      component();
      virtual ~component();
      
      // (un)registers input to component
      // order or registerations may matter
      // and input component may not use
      // all the inputs registered when
      // calculating its state changes
      bool register_input(component* c);
      bool unregister_input(component* c);
      
      // (un)registers components which get
      // signaled when the state of this 
      // component changes
      // each component can (un)register only once
      bool register_output(component* c);
      bool unregister_output(component* c);
      
      // returns output
      bool get() const throw();
      
      // signals component that
      // one of it's input components has changed
      void signal() throw();
      
      
      // returns activation counter which
      // is incremented everytime component
      // is woken up
      unsigned int activation() const throw();
      
    protected:
      
      // for update(): a method for reading in elements
      const std::vector<component*>& inc() const throw();
      
      // recalculates output from inputs
      virtual bool update() throw() = 0;
      
      bool state; // state
      
    private:
      // list of input components
      // component gets its inputs from here
      // if there are no enough inputs
      // component must assume input is false
      std::vector<component*> in;
      
      // list of output components
      // if state changes component wakes up
      // these components
      std::vector<component*> out;
      
      bool ok;
      unsigned int counter;
      
      mutable pthread_mutex_t state_lock;
      mutable pthread_cond_t  state_change;
      
      pthread_t component_thread;
      
    public:
      // main component execution thread
      void threadloop();
      
    };
    
  }
}

#endif
