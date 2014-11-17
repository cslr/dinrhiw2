

#ifndef whiteice__andgate__component_h
#define whiteice__andgate__component_h

#include "component.h"


namespace whiteice
{
  namespace digital_logic
  {
    
    class andgate : public component
    {
    public:
      andgate(){ }
      virtual ~andgate(){ }
      
    protected:
      // recalculates output from inputs
      bool update() throw();
      
    private:
      
    };
    
  };
};

#endif
