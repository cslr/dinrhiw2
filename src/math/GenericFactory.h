

#include <exception>
#include <stdexcept>

#ifndef GenericFactory_h
#define GenericFactory_h

namespace whiteice
{

  class GenericFactory
    {
    public:
      
      virtual void *createGenericInstance() throw(std::bad_alloc,
						  std::logic_error) = 0;
    };
}

#endif
