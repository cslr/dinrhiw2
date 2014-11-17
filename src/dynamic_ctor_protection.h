/*
 * dynamic constructor success/failure
 * class modifier [use by inheritance]
 */

#ifndef dynamic_ctor_protection_h
#define dynamic_ctor_protection_h

#include <stdexcept>
#include <typeinfo>

namespace whiteice
{
  
  template <typename T, typename K>
    class dynamic_ctor_protection_control;
  
  // templatization is for specifying
  // dynamic_ctor_protection_control
  // for specific class T
  template <typename T, typename K=unsigned int>
    class dynamic_ctor_protection
    {
      protected:
      
      dynamic_ctor_protection() throw(std::logic_error)
      {
	if(!canCreate)
	throw std::logic_error("Construction of class is disabled");
	
	if(dynamic_cast<T*>(this))
	throw std::logic_error("DynamicCtor inheritated by wrong class");
      }
      
      virtual ~dynamic_ctor_protection(){ }
      
      protected:
      
      friend class dynamic_ctor_protection_control<T,K>;
      
      // inheritator implements this to tell if
      // class with given key can change state
      
      virtual bool canChange(const K& key) throw() = 0;
      
      private:
      
      static bool canCreate;
    };
  
  
  template <typename T, typename K>
    bool dynamic_ctor_protection<T,K>::canCreate = false;
  
  
  // inheritating class have power to disable / enable template creation
  template <typename T, typename K=unsigned int>
    class dynamic_ctor_protection_control
    {
      protected:    
      
      bool enableCreation() throw()
      {	
	const K& key = getAuthorizationKey();
	
	if(dynamic_ctor_protection<T,K>::canChange(key)){
	  dynamic_ctor_protection<T>::canCreate = true;
	  return true;
	}
	else return false;
      }
      
      bool disableCreation() throw()
      {
	const K& key = getAuthorizationKey();
	
	if(dynamic_ctor_protection<T,K>::canChange(key)){
	  dynamic_ctor_protection<T>::canCreate = false;
	  return true;
	}
	else return false;
      }
      
      // inheritator implements to provide key for
      // a proof of authorizated control
      
      virtual const K& getAuthorizationKey() throw() = 0;
    };

}




#endif




