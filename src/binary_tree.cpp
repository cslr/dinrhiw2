 
#ifndef binary_tree_cpp
#define binary_tree_cpp
 
#include "binary_tree.h"

namespace whiteice
{
  
  template <typename T>
  binarytree<T>::binarytree() 
  {
    root = 0;
    numberOfNodes = 0;
  }
  
  
  template <typename T>
  binarytree<T>::~binarytree() 
  {
    if(root) free_subtree(root);
  }
    
  
  // insert data value t to tree or returns false
  template <typename T>
  bool binarytree<T>::insert(const T& t) 
  {
    try{
      btree_node<T>* n = new btree_node<T>(t);
      
      if(insert(n))
	return true;
      
      delete n;
      return false;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  // removes data with value t
  template <typename T>
  bool binarytree<T>::remove(T& t) 
  {
    try{
      btree_node<T>* n = search(root, t);
      if(n == 0) return false;
      return remove(n);
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  bool binarytree<T>::search(T& value) const 
  {
    btree_node<T>* node = search(root, value);
    if(!node) return false;
    value = node->key;
    return true;
  }
  
  
  template <typename T>
  T& binarytree<T>::maximum() const 
  {
    btree_node<T>* node = maximum_node(root);
    if(!node) throw std::logic_error("no maximum node");
    return node->key;
  }
  
  
  template <typename T>
  T& binarytree<T>::minimum() const 
  {
    btree_node<T>* node = minimum_node(root);
    if(!node) throw std::logic_error("no minimum node");
    return node->key;
  }
  
  
  template <typename T>
  void binarytree<T>::clear() 
  {
    if(root) free_subtree(root);
    root = 0;
    numberOfNodes = 0;
  }
  
  
  template <typename T>
  unsigned int binarytree<T>::size() const 
  {
    return numberOfNodes;
  }
  
  
  /**************************************************/
  
  /* 
   * search for given key from subtree
   */
  template <typename T>
  btree_node<T>* binarytree<T>::search(btree_node<T>* x, T& value) const 
  {
    if(x == 0) return 0;
    if(x->key == value) return x;  
    
    while(1){
      if(value < x->key) x = x->left;
      else x = x->right;
      
      if(x == 0) return 0;
      if(x->key == value) return x;    
    }
  }
  
  
  /* finds node with minimum value from tree which root
   * is node or returns null 
   */
  template <typename T>
  btree_node<T>* binarytree<T>::minimum_node(btree_node<T>* node) const 
  {
    btree_node<T> *prev = 0, *ptr;
    ptr = node;
    
    while(ptr != 0){
      prev = ptr;
      ptr = ptr->left;
    }
    
    return prev;
  }
  
  
  /* finds maximum value of tree of subtree with node root node.
   * returns null if there's no maximum
   */
  template <typename T>
  btree_node<T>* binarytree<T>::maximum_node(btree_node<T>* node) const 
  {
    btree_node<T>* prev = 0, *ptr;
    ptr = node;
      
    while(ptr != 0){
      prev = ptr;
      ptr = ptr->right;
    }
    
    return prev;
  }
  
  
  /* inserts node to tree */
  template <typename T>
  bool binarytree<T>::insert(btree_node<T>* z) 
  {
    try{
      btree_node<T>* y = 0;
      btree_node<T>* x = root;
      
      z->left = 0;
      z->right = 0;
      
      
      while(x != 0){
	y = x;
	if(z->key < x->key) x = x->left;
	else x = x->right;
      }
      
      z->parent = y;
      
      if(y == 0){
	  root = z;
      }
      else{
	if(z->key < y->key) y->left = z;
	else y->right = z;
      }
      
      // node has been inserted
      numberOfNodes++;
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  /* deletes node from the tree or returns false */
  template <typename T>
  bool binarytree<T>::remove(btree_node<T>* z) 
  {
    btree_node<T> *y, *x;
    
    // y will be removed
    if(z->left == 0 || z->right == 0) y = z;
    else y = tree_successor(z);
    
    if(y->left) x = y->left;
    else x = y->right;
    
    if(x) x->parent = y->parent;
    
    if(y->parent == 0)
      root = x;
    else{
      if(y == y->parent->left)
	y->parent->left = x;
      else
	y->parent->right = x;	
    }
    
    if(y != z)
      z->key = y->key; // copies values
    
    delete y;
    numberOfNodes--;
    return true;
  }
  
  
  /* finds successor of node */
  template <typename T>
  btree_node<T>* binarytree<T>::tree_successor(btree_node<T>* x) const 
  {
    if(x == 0) return 0;
    if(x->right)
      return minimum_node(x->right);
    
    btree_node<T>* y = x->parent;
    
    if(y == 0) return 0;
    
    while(x == y->right){
      x = y;
      y = y->parent;
      
      if(y == 0) break;
    }
    
    return y;
  }
  
  
  /* frees subtree */
  template <typename T>    
  void binarytree<T>::free_subtree(btree_node<T>* x) 
  {
    if(x){
      free_subtree(x->left);
      free_subtree(x->right);
      
      delete x;
      numberOfNodes--;
    }
  }
  
  
}
    


#endif

