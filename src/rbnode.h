/*
 * red-black tree node
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 *
 */
#ifndef rbnode_h
#define rbnode_h

namespace whiteice
{
  template <typename T>
    class rbtree;
  
  
  template <typename T>
    class rbnode
    {
    public:
      rbnode(const T&);
      ~rbnode();    
      
      // returns true if node is leaf node
      bool leaf() const ;
      
      // is color of node
      bool red() const { return !black; }
      
      T& key() { return value; }
      const T& key() const { return value; }
      
    private:
      friend class rbtree<T>;
      
      bool left_rotate(rbtree<T>* tree) ;
      bool right_rotate(rbtree<T>* tree) ;
      
      bool black; // is the tree black? (otherwise red)
      T value;
      rbnode<T>* parent;
      rbnode<T>* left;
      rbnode<T>* right;
    };
}
    
#include "rbtree.h"

#include "rbnode.cpp"

#endif

