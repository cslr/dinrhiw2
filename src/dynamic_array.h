/*
 * dynamic array header
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */
#ifndef dynamic_array_h
#define dynamic_array_h

#include "array.h"
#include "stack.h"
#include "queue.h"

namespace whiteice
{
  template <typename D, typename T=int>
    class dynamic_array : public array<D,T>, public stack<D,T>, public queue<D,T>
  {
    public:
    
    dynamic_array(const T& n = 0);
    dynamic_array(const array<D,T>& a);
    virtual ~dynamic_array();
    
    unsigned int size() const ;
    bool resize(const T& n) ;
    
    void clear() ;
        
    D& operator[](const T& n) ;
    const D& operator[](const T& n) const ;
    
    bool push(const D& d) ;
    D pop() ;
    
    bool enqueue(const D& data) ;
    D dequeue() ;
    
    
    private:
    
    class darray_node
    {
    public:
      
      darray_node *prev, *next;
      D data;
    };
    
    darray_node *first, *last;
    T size_of_array;
    
    void add_empty_node() ;
    void add_node(const D& data) ;
    D remove_node() ;
    D remove_node(const T& n) ;
    
  };
  
}
  
#include "dynamic_array.cpp"


#endif

