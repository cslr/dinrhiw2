
#ifndef rbtree_cpp
#define rbtree_cpp

#include <string>
#include <exception>
#include "rbtree.h"

namespace whiteice
{
  /**************************************************/
  
  template <typename T>
  rbtree<T>::rbtree() throw()
  {
    root = 0;
    numberOfNodes = 0;
  }
  
  
  template <typename T>
  rbtree<T>::~rbtree() throw()
  {
    if(root) free_subtree(root);
  }
  
  
  template <typename T>
  bool rbtree<T>::insert(const T& t) throw()
  {
    try{
      rbnode<T>* n = new rbnode<T>(t);
      return insert(n);
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  bool rbtree<T>::remove(T& t) throw()
  {
    try{    
      rbnode<T>* n = iterative_search(root, t);
      if(n == 0)
	return false;
      
      return remove(n);
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  bool rbtree<T>::search(T& value) const throw()
  {
    try{
      rbnode<T>* t = iterative_search(root, value);
      if(t == 0)
	return false;
      
      value = t->value;
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  T& rbtree<T>::maximum() const throw(std::logic_error)
  {
    rbnode<T> *k = max_value(root);
    if(!k) throw std::logic_error("no maximum value: empty tree");
    return k->value;
  }
  
  template <typename T>
  T& rbtree<T>::minimum() const throw(std::logic_error)
  {
    rbnode<T> *k = min_value(root);  
    if(!k) throw std::logic_error("no minimum value: empty tree");
    return k->value;
  }
  
  
  template <typename T>
  void rbtree<T>::clear() throw()
  {
    if(root) free_subtree(root);
    root = 0;
    numberOfNodes = 0;
  }
  
  
  template <typename T>
  unsigned int rbtree<T>::size() const throw()
  {
    return numberOfNodes;
  }
  
  
  template <typename T>
  bool rbtree<T>::list() const throw()
  {
    std::vector<rbnode<T>*> A, B;      
    A.push_back(root);
    
    while(A.size() != 0 || B.size() != 0){
      
      if(A.size() > 0){
	
	for(unsigned int i=0;i<A.size();i++){
	  if(A[i]){
	    std::cout << A[i]->value << "|" << A[i]->black << "  ";
	    B.push_back(A[i]->left);
	    B.push_back(A[i]->right);
	  }
	  else std::cout << "0.0  ";
	}
	
	std::cout << std::endl;
	A.clear();
      }
      
      else if(B.size() > 0){	  
	for(unsigned int i=0;i<B.size();i++){
	  if(B[i]){
	    std::cout << B[i]->value << "|" << B[i]->black  << "  ";
	    A.push_back(B[i]->left);
	    A.push_back(B[i]->right);
	  }
	  else std::cout << "0.0  ";
	}
	
	std::cout << std::endl;	    
	B.clear();
      }
    }
    
    return true;
  }
  
  
  /**************************************************/
  
  
  /*
   * tree insert
   * uses left/right rotates to
   * maintain red-black property -> balanced trees
   */
  template <typename T>
  bool rbtree<T>::insert(rbnode<T>* x) throw()
  {
    try{
      if(!basic_insert(x)) return false;
      x->black = false;
      
      rbnode<T>* y = 0;	
      bool ok = true;
      
      if(x == root) ok = false;
      else if(x->parent->black) ok = false;
      
      while(ok){
	
	if(x->parent == x->parent->parent->left){
	  y = x->parent->parent->right;
	  
	  bool red = true;
	  if(y == 0) red = false;
	  else if(y->black) red = false; // black
	  
	  if(red){ // red
	    x->parent->black = true;
	    y->black = true;
	    x->parent->parent->black = false; 
	    x = x->parent->parent;
	  }
	  else{
	    if(x == x->parent->right){
	      x = x->parent;
	      x->left_rotate(this);
	    }
	    
	    x->parent->black = true;
	    x->parent->parent->black = false;
	    x->parent->parent->right_rotate(this);
	  }
	}
	else{
	  y = x->parent->parent->left;
	  
	  bool red = true;
	  if(y == 0) red = false;
	  else if(y->black) red = false; // black
	  
	  if(red){ // red
	    x->parent->black = true;
	    y->black = true;
	    x->parent->parent->black = false; 
	    x = x->parent->parent;
	  }
	  else{
	    if(x == x->parent->left){
	      x = x->parent;
	      x->right_rotate(this);
	    }
	    
	    x->parent->black = true;
	    x->parent->parent->black = false;
	    x->parent->parent->left_rotate(this);
	  }
	}
	
	
	root->black = true;
	if(x == root) ok = false;
	else if(x->parent->black) ok = false;
      }
      
      
      root->black = true;
      numberOfNodes++;
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  /*
   * normal binary tree insert
   */
  template <typename T>
  bool rbtree<T>::basic_insert(rbnode<T>* z) throw()
  { 
    try{
      rbnode<T>* y = 0;
      rbnode<T>* x = root;
      
      z->left = 0;
      z->right = 0;
      
      
      while(x != 0){
	y = x;
	if(z->value < x->value) x = x->left;
	else x = x->right;
      }
      
      z->parent = y;
      
      if(y == 0){
	root = z;
      }
      else{
	if(z->value < y->value) y->left = z;
	else y->right = z;
      }    
      
      // node has been inserted
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  rbnode<T>* rbtree<T>::tree_search(rbnode<T>* x, T& value) const throw()
  {
    if(x == 0) return 0;
    if(x->value == value) return x;
    
    if(value < x->value)
      return tree_search(x->left, value);
    else
      return tree_search(x->right, value);
  }
  
  
  template <typename T>
  rbnode<T>* rbtree<T>::iterative_search(rbnode<T>* x, T& value) const throw()
  {
    if(x == 0) return 0;
    if(x->value == value) return x;  
    
    while(1){
      
      if(value < x->value) x = x->left;
      else x = x->right;
      
      if(x == 0) return 0;
      if(x->value == value) return x;    
    }
  }
  
  
  template <typename T>
  rbnode<T>* rbtree<T>::min_value(rbnode<T>* x) const throw()
  {
    if(x == 0) return 0;
    while(x->left != 0) x = x->left;
    return x;
  }
  
  
  template <typename T>
  rbnode<T>* rbtree<T>::max_value(rbnode<T>* x) const throw()
  {
    if(x == 0) return 0;
    while(x->right != 0) x = x->right;
    return x;
  }
  
  
  template <typename T>
  rbnode<T>* rbtree<T>::tree_successor(rbnode<T>* x) const throw()
  {
    if(x == 0) return 0;
    if(x->right)
      return min_value(x->right);
    
    rbnode<T>* y = x->parent;
    
    if(y == 0) return 0;
    
    while(x == y->right){
      x = y;
      y = y->parent;
      
      if(y == 0) break;
    }
    
    return y;
  }
  
  
  template <typename T>
  bool rbtree<T>::remove(rbnode<T>* z) throw()
  {
    rbnode<T> *y, *x;
    
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
    
    
    if(y->black && x != 0)
      rb_delete_fixup(x);
    
    
    // copies correct values
    if(y != z){
      // exchanges node pointers, this doesn't copy values
      // OLD: 
      // z->key() = y->key(); // copies values
      
      // this way value for data will always be
      // in a same rbnode memory location/pointer
      // where it was when node/data inserted into tree.
      // this is useful if pointer for data is passed to some
      // other (friend) class. This also doesn't assume anything
      // about T's "=" operator.
      
      // updates parent pointers
      if(z->parent){
	if(z == z->parent->left)
	  z->parent->left = y;
	  else
	    z->parent->right = y;
      }
      
      if(z == root) root = y;
      y->parent = z->parent;
      
      // swaps children
      
      // children's parent pointers
      if(z->left)
	z->left->parent = y;
      if(z->right)
	z->right->parent = y;
      
      y->left = z->left;
      y->right = z->right;
      
      y->black = z->black;
      
      y = z;  // now:  y == z
    }
    
    delete y;
    
    numberOfNodes--;
    return true;      
  }
  
  
  /* keeps red/black property after delete. */
  template <typename T>
  void rbtree<T>::rb_delete_fixup(rbnode<T>* x) throw()
  {
    rbnode<T>* w;
    
    while(x != root && x->black){
      if(x == x->parent->left){      
	w = x->parent->right;
	
	bool wblack = true;
	if(w) wblack = w->black;
	
	if(wblack){
	  if(w) w->black = true;
	  x->parent->black = false;
	  x->parent->left_rotate(this);
	  w = x->parent->right;	
	}
	
	bool wlblack = true, wrblack = true;
	if(w) if(w->left) wlblack = w->left->black;
	if(w) if(w->right) wrblack = w->right->black;
	
	if(wlblack && wrblack){
	  if(w) w->black = false;
	  x = x->parent;
	}
	else{
	  if(w){
	    if(w->right){
	      if(w->right->black){
		if(w->left) w->left->black = true;
		w->black = false;
		w->right_rotate(this);
		w = x->parent->right;
	      }
	    }
	    
	    w->black = x->parent->black;
	    if(w->right) w->right->black = true;
	  }
	  
	  x->parent->black = true;
	  x->parent->left_rotate(this);
	  x = root;
	}
      }
      else{	  
	w = x->parent->left;
	
	bool wblack = true;
	if(w) wblack = w->black;
	
	if(wblack){	    
	  if(w) w->black = true;
	  x->parent->black = false;
	  x->parent->right_rotate(this);
	  w = x->parent->left;	
	}
	
	bool wlblack = true, wrblack = true;
	if(w) if(w->left) wlblack = w->left->black;
	if(w) if(w->right) wrblack = w->right->black;
	
	if(wlblack && wrblack){
	  if(w) w->black = false;
	  x = x->parent;
	}
	else{
	  if(w){
	    if(w->left){
	      if(w->left->black){
		if(w->right) w->right->black = true;
		w->black = false;
		w->left_rotate(this);
		w = x->parent->left;
	      }
	    }
	    
	    w->black = x->parent->black;
	    if(w->left) w->left->black = true;
	  }
	  
	  x->parent->black = true;	      
	  x->parent->right_rotate(this);
	  x = root;
	}
      }
    }
    
    x->black = true;
  }
  
  
  template <typename T>
  void rbtree<T>::free_subtree(rbnode<T>* x) throw()
  {
    if(x){
      free_subtree(x->left);
      free_subtree(x->right);
      delete x;
    }
  }
  
}
    
#endif



