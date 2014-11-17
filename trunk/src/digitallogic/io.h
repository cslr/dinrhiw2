/*
 * input and output components
 * user can set value of input component
 * user can read value of output component
 */

#ifndef whiteice__digital_logic__io_h
#define whiteice__digital_logic__io_h

#include "component.h"


namespace whiteice
{
  namespace digital_logic
  {
    class input : public component
    {
    public:
      input();
      virtual ~input(){ }
      
      // sets value of input
      void set(bool value) throw();
      
      
    protected:
      // recalculates output from inputs
      bool update() throw();
      
      bool input_state;
      
    };
    
    
    class output : public component
    {
    public:
      output(){ }
      virtual ~output(){ }
      
    protected:
      // recalculates output from inputs
      bool update() throw();
      
    };

  };
};

#endif

