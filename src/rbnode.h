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
      bool leaf() const throw();
      
      // is color of node
      bool red() const throw(){ return !black; }
      
      T& key() throw(){ return value; }
      const T& key() const throw(){ return value; }
      
    private:
      friend class rbtree<T>;
      
      bool left_rotate(rbtree<T>* tree) throw();
      bool right_rotate(rbtree<T>* tree) throw();
      
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

