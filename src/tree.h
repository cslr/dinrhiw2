/*
 * tree interface
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */

#ifndef tree_h
#define tree_h

#include <exception>
#include <stdexcept>

#include "container.h"

namespace whiteice
{
  template <typename T>
    class tree : public container<T,T>
    {
    public:      
      virtual ~tree(){ }
      
      // inserts value to search tree
      virtual bool insert(const T& t)  = 0;
      
      // remove()s value to tree
      virtual bool remove(T& t)  = 0;
      
      // search()es for given value, updates value and
      // returns true in case of success
      virtual bool search(T& value) const  = 0;
      
      // returns maximum & minimum keys/objects from the tree
      virtual T& maximum() const  = 0;
      virtual T& minimum() const  = 0;
      
    };
}


#endif
