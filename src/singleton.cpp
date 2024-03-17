

#ifndef singleton_cpp
#define singleton_cpp

#include "singleton.h"
#include "singleton_list.h"
#include <iostream>
#include <stdexcept>
#include <typeinfo>

namespace whiteice
{


  template <typename T>
  T* singleton<T>::instance_ptr = 0;
  
  template <typename T>
  bool singleton<T>::canCreate = false;
  
  
  
  template <typename T>
  singleton<T>::singleton() 
  {
    if(singleton<T>::canCreate == false)
      throw std::logic_error("singleton: unauthorized create attempt.");
    
  }
  
  
  template <typename T>
  singleton<T>::singleton(const singleton<T>& s)
    
  {
    throw std::logic_error("singleton: no copying allowed.");
  }
  
  
  template <typename T>
  singleton<T>& singleton<T>::operator=(const singleton<T>& s)
    
  {
    throw std::logic_error("singleton: no copying allowed.");
  }
  
  
  template <typename T>
  singleton<T>::~singleton() 
  {
    canCreate = false;
    
    // freeing instance is done by singletonList
    
    if(instance_ptr)
      instance_ptr = 0; // if not done by someone else
  }
  

  template <typename T>
  T& singleton<T>::instance() // throw(std::bad_alloc, std::logic_error)
  {
    if(instance_ptr)
      return (*(dynamic_cast<T*>(instance_ptr)));
    
    // should use locks (also with free_instance)
    // in order to make thread safe
    
    
    singleton<T>::canCreate = true; 
    instance_ptr = new T; // causes call to singleton() and singletonListable() ctors
    singleton<T>::canCreate = false;
    
    
    // checks T is subclass of singleton<T>
    try{
      if(typeid(dynamic_cast< singleton<T>* >(instance_ptr)) !=
	 typeid(singleton<T>*))
	{
	  delete instance_ptr; instance_ptr = 0;
	  throw std::logic_error("getInstance(): T is non-singleton class");
	}
    }
    catch(std::bad_cast& e){
      delete instance_ptr; instance_ptr = 0;
      throw std::logic_error("getInstance(): T is non-singleton class");
    }
    
    return (*dynamic_cast<T*>(instance_ptr));
  }
  
  
  
  template <typename T>
  bool singleton<T>::destroy() 
  {
    if(!instance_ptr)
      return false;
    
    delete instance_ptr;
    instance_ptr = 0;
    
    return true;
  }
  
  
}

#endif

