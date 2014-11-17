/*
 * FactoryImpl class that forces
 * creation to happen only
 * through it (when inherited)
 *
 * it implements Factory<I> interface.
 */

#include <stdexcept>
#include <exception>

#include "Singleton.h"
#include "Factory.h"

#ifndef FactoryImpl_h
#define FactoryImpl_h

namespace whiteice
{

  /*
   * typename T - type
   * typename I - interface
   */
  template <typename I, typename T>
    class FactoryImpl : 
    public Factory<I>,
    public Singleton< FactoryImpl<I, T> >
    {

    protected:
      
      FactoryImpl() throw(std::logic_error);
      
    public:
      
      I* createInstance() throw(std::bad_alloc,
				std::logic_error);  
      
    };
  
}




#endif
