/*
 * static array implementation
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */
#ifndef static_array_h
#define static_array_h

#include <stdexcept>
#include <exception>
#include "array.h"

namespace whiteice
{
  template <typename D, typename T=int>
    class static_array : public array<D,T>
  {
    public:
    
    static_array(const T& n = 0);
    static_array(const array<D,T>& a);
    virtual ~static_array();
    
    unsigned int size() const ;
    bool resize(const T& n) ;
    
    void clear() ;
    
    D& operator[](const T& n) ;
    const D& operator[](const T& n) const ;
    
    private:
    
    D* array_memory;
    T size_of_array;
  };
  
}

#include "static_array.cpp"


#endif

