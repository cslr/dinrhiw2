
#include <iostream>
#include <vector>

#include "singleton_list.h"

using namespace std;


namespace whiteice
{

  /******************************************************/
  /* singleton listable */
  
  singleton_listable::singleton_listable() throw()
  { // -> not directly creatable
    
    singleton_list::add(*this);
  }
  
  
  singleton_listable::~singleton_listable() throw()
  {
    singleton_list::remove(*this);
  }
  
  
  /******************************************************/
  /* singleton_list */
  
  // static initialization
  bool singleton_list::created = false;
  bool singleton_list::no_inputs = true;
  std::vector<singleton_listable *> singleton_list::slist;
  
  static singleton_list slist;
  
  
  singleton_list::singleton_list() throw(std::logic_error)
  {
    if(created)
      throw std::logic_error("Only one singleton list can be created");
    
    created = true;
    no_inputs = false;
  }
  
  
  singleton_list::~singleton_list()
  {
    // (locks needed)
    
    no_inputs = true; // no adding / unregistering anymore
    
    std::vector<singleton_listable*>::iterator i =
      slist.begin();
    
    for(;i != slist.end();i++){
      
      if(*i)
	delete (*i);
    }
    
    slist.clear();  
    created = false;  
  }
  
  
  
  bool singleton_list::add(singleton_listable& sl) throw()
  {
    if(!created || no_inputs) return false;
    
    slist.push_back(&sl);
    return true;
  }
  
  
  bool singleton_list::remove(singleton_listable& sl) throw()
  {
    if(!created || no_inputs) return false;
    
    // direct search .. hashes would be better
    
    std::vector<singleton_listable*>::iterator i =
      slist.begin();
    
    for(;i != slist.end();i++){
      if(*i == &sl)
      {
	i = slist.erase(i);
	return true;
      }
    }
    
    return false;
  }
  
  
}


