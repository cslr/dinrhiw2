
#ifndef whiteice__digital_logic__xorgate_h
#define whiteice__digital_logic__xorgate_h

#include "component.h"

namespace whiteice
{
  namespace digital_logic
  {
    
    class xorgate : public component
    {
    public:
      xorgate(){ }
      virtual ~xorgate(){ }
      
    protected:
      // recalculates output from inputs
      bool update() throw();
      
    private:
      
    };
    
  };
};


#endif

