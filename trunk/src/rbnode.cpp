/*
 * red black tree node implementation
 * Tomas Ukkonen <tomas.ukkonen@hut.fi>
 */
#ifndef rbnode_cpp
#define rbnode_cpp

#include "rbnode.h"

namespace whiteice
{
  
  template <typename T>
  rbnode<T>::rbnode(const T& t)
  {
    value = t;
    parent = left = right = 0;
  }
  
  
  template <typename T>
  rbnode<T>::~rbnode()
  {
    
  }
  
  
  template <typename T>
  bool rbnode<T>::leaf() const throw()
  {
    if(!left & !right) return true;
    return false;
  }
  
  
  /*
   * (red-black tree) left rotates given node
   */
  template <typename T>
  bool rbnode<T>::left_rotate(rbtree<T>* tree) throw()
  {
    if(!right) return false;
    
    rbnode<T>* y = this->right;  
    this->right = y->left;
    
    if(y->left)
      y->left->parent = this;
    
    y->parent = this->parent;
    
    if(this->parent == 0)
      tree->root = y;
    else{
      if(this == (this->parent)->left)
	this->parent->left = y;
      else
	this->parent->right = y;
    }
    
    y->left = this;
    this->parent = y;
    return true;
  }
  
  /*
   * right rotates red black tree
   */
  template <typename T>
  bool rbnode<T>::right_rotate(rbtree<T>* tree) throw()
  {
    if(!left) return false;
    
    rbnode<T>* x = this->left;
    this->left = x->right;
    
    if(x->right)
      x->right->parent = this;
    
    x->parent = this->parent;        
    
    if(this->parent == 0)
      tree->root = x;
    else{
      if(this->parent->left == this)
	this->parent->left = x;
      else
	this->parent->right = x;
    }
    
    this->parent = x;
    x->right = this;
    
    return true;
  }
  
}
    
#endif




