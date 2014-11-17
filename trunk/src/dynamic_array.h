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
    
    unsigned int size() const throw();
    bool resize(const T& n) throw();
    
    void clear() throw();
        
    D& operator[](const T& n) throw(std::out_of_range);
    const D& operator[](const T& n) const throw(std::out_of_range);
    
    bool push(const D& d) throw();
    D pop() throw(std::logic_error);
    
    bool enqueue(const D& data) throw();
    D dequeue() throw(std::logic_error);
    
    
    private:
    
    class darray_node
    {
    public:
      
      darray_node *prev, *next;
      D data;
    };
    
    darray_node *first, *last;
    T size_of_array;
    
    void add_empty_node() throw(std::bad_alloc);
    void add_node(const D& data) throw(std::bad_alloc);
    D remove_node() throw();
    D remove_node(const T& n) throw();
    
  };
  
}
  
#include "dynamic_array.cpp"


#endif

