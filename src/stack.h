/*
 * stack interface
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */
#ifndef stack_h
#define stack_h

#include <exception>
#include <stdexcept>

#include "container.h"

namespace whiteice
{
  template <typename D, typename T=int>
    class stack : public container<D,T>
  {
    public:
    
    virtual ~stack(){ }
    
    /* returns true if data is pushed succesfully to stack */
    virtual bool push(const D& d)  = 0;
    
    /* returns pushed data from the top of stack or throws
     * logic_error exception if stack is empty */
    virtual D pop()  = 0;
    
  };

}

#endif

