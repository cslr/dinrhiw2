/*
 * singleton implements singleton design pattern
 * (also SingletonListable and SingletonList
 *  but unseeable from user point of view)
 *
 * usage:
 *  1. change "class X" -> "class X : public singleton<X>".
 *  2. provide default X() constructor that says to throw std::logic_error
 *  done (X is singleton now)
 *
 * use getInstance(). using constructor (new) throws std::logic_error.
 * (or getInstancePtr())
 *
 * instance can be deleted for freeInstance()d or 
 * they get automatically freed at the program end time
 * (in order).
 *
 * - not thread safe (implementable via semaphores)
 *
 */



#ifndef singleton_h
#define singleton_h

#include "singleton_list.h"

#include <stdexcept>


namespace whiteice
{

  /*
   * generic singleton<T> class.
   * implementation T must inherit this
   * in order to become 'singleton' = 
   * "actual singleton + global + automatic removal"
   */
  template <typename T>
    class singleton : public singleton_listable
    {
    private:
      
      static T* instance_ptr;
      static bool canCreate;
      
    protected: // only locally available
      
      singleton() ;
      virtual ~singleton() ;
      
      singleton(const singleton<T>& s)
	;
      
      singleton<T>& operator=(const singleton<T>& s)
	;
      
      
    public:
      
      static T& instance(); // throw(std::bad_alloc, std::logic_error);
      
      
      static bool destroy() ;
      
    };
  
  
}

// template must have code available (gcc)
#include "singleton.cpp"


#endif





