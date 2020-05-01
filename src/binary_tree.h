/*
 * binary tree implementation
 */
#ifndef binary_tree_h
#define binary_tree_h

#include "tree.h"
#include "btree_node.h"

namespace whiteice
{
  
  template <typename T>
    class binarytree : public tree<T>
    {
    public:	
      binarytree() ;
      ~binarytree() ;
      
      bool insert(const T& t) ;  
      bool remove(T& t) ;
      
      bool search(T& value) const ;
      T& maximum() const ;
      T& minimum() const ;
      
      void clear() ;
      
      unsigned int size() const ;
      
    protected:
      btree_node<T>* search(btree_node<T>* x, T& value) const ;
      
      btree_node<T>* minimum_node(btree_node<T>* node) const ;
      btree_node<T>* maximum_node(btree_node<T>* node) const ;
      
      bool insert(btree_node<T>* z) ;	
      bool remove(btree_node<T>* z) ;
      
      btree_node<T>* tree_successor(btree_node<T>* x) const ;
      void free_subtree(btree_node<T>* x) ;
      
      
      btree_node<T> *root;
      unsigned int numberOfNodes;
    };
  
}


#include "binary_tree.cpp"

#endif
