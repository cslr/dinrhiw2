/*
 * avl tree node
 * Tomas Ukkonen <tomas.ukkonen@hut.fi>
 */

#ifndef avlnode_h
#define avlnode_h

namespace whiteice
{
  template <typename T>
    class avltree;
  
  template <typename T>
    class avlnode
    {
    public:
      avlnode(const T&);
      ~avlnode();    
      
      // returns true if node is leaf node
      bool leaf() const ;
      
      // height of node rooted subtree
      unsigned int nodes() const { return numNodes; }
      
      T& key() { return value; }
      const T& key() const { return value; }	
      
      // returns height of this subtree
      unsigned int height() const ;
      
      // returns height difference between nodes
      // (this.height - node.height)
      int hdifference(const avlnode<T>& node) const ;
      
    private:
      friend class avltree<T>;
      
      bool equal_height(unsigned int a, unsigned int b)
	const ;
      
      unsigned int numNodes; // in this subtree
      T value;
      avlnode<T>* parent;
      avlnode<T>* left;
      avlnode<T>* right;	
    };
  
};

#include "avltree.h"
#include "avlnode.cpp"

#endif
