

#ifndef whiteice__nandgate__component_h
#define whiteice__nandgate__component_h

#include "component.h"


namespace whiteice
{
  namespace digital_logic
  {
    
    class nandgate : public component
    {
    public:
      nandgate(){ }
      virtual ~nandgate(){ }
      
    protected:
      // recalculates output from inputs
      bool update() throw();
      
    private:
      
    };
    
  };
};

#endif
