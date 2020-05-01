/*
 * factory mapping is used for
 * generic one point access for creating
 * interface implementations
 *
 * interface factory pointer Factory<I> is
 * passed to this function when registering factory
 * that implements interface.
 *
 * "user": asks implementation of specific interface
 */

#ifndef FactoryMapping_h
#define FactoryMapping_h


#include <typeinfo>
#include <string>
#include <map>
#include "Singleton.h"


namespace whiteice
{

  class FactoryMapping : Singleton< FactoryMapping >
  {
  public:
    
    // gets access to factory
    template <typename I>
      I* getImplementation(I*& ptr_ref) const ;
    
    
    // registering new implementation, multiples are not allowed - use properly
    template <typename I>
      bool setImplementation(Factory<I>* i) ;
    
  private:  
    map<std::string, GenericFactory*> factories;
  };
  
  
  /***************************************************/
  // shortcut(s) for FactoryMapping::getInstance()->get/setImplementation()
  
  class FM {
  public: // hopefully compiler inlines these
    
    // FM::create() - implementation
    template <typename I> inline static I* create(I*& ptr_ref) 
    {
      return FactoryMapping::getInstance()->createInstance(ptr_ref);
    }
      
    // FM::set() - implementator
    template <typename I> inline static bool set(Factory<I>* i) 
    {
      return FactoryMapping::getInstance()->setImplementation(i);
    }
    
  };
  
  
  
  /***************************************************/
  // template implementations
  
  
  template <typename I>
  I* FactoryMapping::getImplementation(I*& ptr_ref) const 
  {
    try{
      std::string str = typeid(I).name();
      
      map<std::string, GenericFactory*>;:iterator i =
					   factories.find(str);
      
      ptr_ref = dynamic_cast<Factory<I>*>(*i)->createInstance();
      return ptr_ref;
    }
    catch(std::exception& e){ return 0; }
  }
  
  
  // registering new implementation, multiples are not allowed - use properly
  template <typename I>
  bool FactoryMapping::setImplementation(Factory<I>* i) 
  {
    try{
      if(!i) return false;
      std::string str = typeid(I).name();
      
      // override not possible - for now, change when restriction
      // for calling function are implemented
      
      if(factories.find(str) != factories.end())
	return false;
      
      factories[str] = dynamic_cast<GeneridFactory*>(i);
      
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
}



#endif







