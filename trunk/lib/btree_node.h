
#ifndef btree_node_h
#define btree_node_h

#include "tree.h"

namespace whiteice
{
  
  template <typename T>
    class binarytree;
  
  template <typename T>
    class btree_node
    {
    public:
      btree_node()
      {
	left = 0;
	right = 0;
	parent = 0;
      }
      
      btree_node(const T& key)
      {
	left = 0;
	right = 0;
	parent = 0;
	this->key = key;
      }
      
    protected:
      friend class binarytree<T>;
      
      btree_node<T> *left, *right, *parent;
      T key;
    };
}

#endif
