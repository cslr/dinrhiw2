/*
 * Factory class interface (+ void casting case) for type I
 */

#include <exception>
#include <stdexcept>
#include "GenericFactory.h"

#ifndef Factory_h
#define Factory_h

namespace whiteice
{
  /*
   * typename I - interface
   */
  template <typename I>
    class Factory : public GenericFactory
    {
    public:
      
      /* interface that specific FactoryImpl<T,I>
       * (creating instances of T with interface I)
       * (must be dynamic_cast:able)
       */
      
      // inheritator implements
      virtual I* createInstance() throw(std::bad_alloc,
					std::logic_error) = 0;
      
      
      // needed by FactoryMapping
      void* createGenericInstance() throw(std::bad_alloc,
					  std::logic_error);
    };
  
}



#endif
