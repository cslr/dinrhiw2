
#include "component.h"

#ifndef whiteice__complement__component_h
#define whiteice__complement__component_h

namespace whiteice
{
  namespace digital_logic
  {
    class complement : public component
    {
    public:
      complement(){ }
      virtual ~complement(){ }
      
    protected:
      // recalculates output from inputs
      bool update() throw();
      
    private:
      
    };
    
  };
};

#endif
