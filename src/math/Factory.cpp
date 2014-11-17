


#include "Factory.h"
#include <typeinfo>

#ifndef Factory_cpp
#define Factory_cpp

namespace whiteice
{

  template <typename I>
  void* Factory<I>::createGenericInstance() 
    throw(std::bad_alloc, std::logic_error)
    
  {
    return dynamic_cast<void*>(createInstance());
  }
  
}

#endif
