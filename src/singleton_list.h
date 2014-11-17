/*
 * keeps list of singletons. and
 * deletes them when program ends
 * meant to be only used by singleton<T>
 * classes
 */

// no best possible solution, anyone can
// create singleton_list and inherit singleton_listable
// not thread-safe

#ifndef singleton_list_h
#define singleton_list_h

#include <vector>
#include <stdexcept>

namespace whiteice
{

  class singleton_list;

  // forces that singleton_list's objects are
  // registered to it + hides different template cases
  class singleton_listable
    {
    protected:
      
      singleton_listable() throw(); // -> not directly creatable
      virtual ~singleton_listable() throw();
      
    private:
      friend class singleton_list;  
      
    };
  
  
  // makes sure singleton(listable)s are
  // removed when program stops (and are removed
  // in order).
  
  class singleton_list
    {
    public:
      
      singleton_list() throw(std::logic_error);
      ~singleton_list();
      
    private:
      
      // only singleton_listables (or succesfully claiming to be)
      // can add pointers (private -> only friends = singleton_listable
      // can access.
      static bool add(singleton_listable& id) throw();
      
      static bool remove(singleton_listable& id) throw();
      
      friend class singleton_listable;
      
      
      static std::vector<singleton_listable*> slist;
      static bool created; // prevents multiple instances
      
      static bool no_inputs; // changing internal structure
      
    };
  
};

#endif

