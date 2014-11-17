
#include "component.h"

#ifndef whiteice__orgate__component_h
#define whiteice__orgate__component_h

namespace whiteice
{
  namespace digital_logic
  {
    class orgate : public component
    {
    public:
      orgate(){ }
      virtual ~orgate(){ }
      
    protected:
      // recalculates output from inputs
      bool update() throw();
      
    private:
      
    };
    
  };
};

#endif
