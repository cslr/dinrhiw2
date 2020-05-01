/*
 * hash table interface
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */
#ifndef hash_table_h
#define hash_table_h

#include "container.h"

namespace whiteice
{
  template <typename T, typename D>
    class hash_table : public container<T,T>
    {
      /* inserts key (and its data) into hash */
      virtual bool insert(const T& key, D& data)  = 0;
      
      /* removes key (and its data) from hash */
      virtual bool remove(const T& key, D& data)  = 0;
      
      /* finds given data for given key from hash or
       * throws exception if key isn't in a hash.
       */
      virtual D&   search(const T& key)  = 0;
      
      /* same as search() */
      virtual D& operator[](const T& key)  = 0;
    };
  
}


#endif








