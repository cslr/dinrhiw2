


#include <stdexcept>
#include <exception>
#include <typeinfo>
#include "FactoryImpl.h"

#ifndef FactoryImpl_cpp
#define FactoryImpl_cpp

namespace whiteice
{
  
  template <typename I, typename T>
  FactoryImpl<I,T>::FactoryImpl() throw(std::logic_error){ }
  
  
  template <typename I, typename T>
  I* FactoryImpl<I,T>::createInstance() throw(std::bad_alloc,
					      std::logic_error)
  {
    T* p = new T();
    
    try{ return dynamic_cast<I*>(p); }
    catch(std::exception& e){ throw std::logic_error(e.what()); }
  }

}

#endif


