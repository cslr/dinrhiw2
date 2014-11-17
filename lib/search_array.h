/*
 * search array tree interface
 */

#ifndef search_array_h
#define search_array_h

#include "array.h"

namespace whiteice
{
 
  template <typename D, typename T>
    class search_array : array<D,T>
    {
    public:
      
      virtual bool insert(const T& key, D& data) = 0;
      virtual D    remove(const T& key) = 0;
      
      virtual D&   search(const T& key) = 0;
      
      // 
      // virtual bool resize(const T& n) = 0;
      // virtual const T& size() const = 0;
      // 
      // virtual D& operator[](const T& key) = 0;
      // virtual const D& operator[](const T& key) const = 0;
      // 
    };
};

#endif
 
 
 
