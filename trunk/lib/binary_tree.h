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
      binarytree() throw();
      ~binarytree() throw();
      
      bool insert(const T& t) throw();  
      bool remove(T& t) throw();
      
      bool search(T& value) const throw();
      T& maximum() const throw(std::logic_error);
      T& minimum() const throw(std::logic_error);
      
      void clear() throw();
      
      unsigned int size() const throw();
      
    protected:
      btree_node<T>* search(btree_node<T>* x, T& value) const throw();
      
      btree_node<T>* minimum_node(btree_node<T>* node) const throw();
      btree_node<T>* maximum_node(btree_node<T>* node) const throw();
      
      bool insert(btree_node<T>* z) throw();	
      bool remove(btree_node<T>* z) throw();
      
      btree_node<T>* tree_successor(btree_node<T>* x) const throw();
      void free_subtree(btree_node<T>* x) throw();
      
      
      btree_node<T> *root;
      unsigned int numberOfNodes;
    };
  
}


#include "binary_tree.cpp"

#endif
