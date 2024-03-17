/* 
 * AVL tree implementation
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */

#ifndef avltree_cpp
#define avltree_cpp

#include "avltree.h"
#include <iostream>
#include <vector>
#include <exception>
#include <stdexcept>

namespace whiteice
{
  
  template <typename T>
  avltree<T>::avltree()
  {      
    dummy = new avlnode<T>(T());
    dummy->parent = dummy;
    dummy->left   = dummy;
    dummy->right  = dummy;
    dummy->numNodes = 0;
    
    root = dummy;
  }
  
  
  template <typename T>
  avltree<T>::~avltree()
  {
    free_subtree(root);
    delete dummy;
    root = 0;      
  }
  
  
  // inserts value to search tree
  template <typename T>
  bool avltree<T>::insert(const T& t) 
  {
    try{
      avlnode<T>* n = new avlnode<T>(t);
      return insert(n);
    }
    catch(std::exception& e){ return false; }
  }
  
  
  // remove()s value to tree
  template <typename T>
  bool avltree<T>::remove(T& t) 
  {
    try{    
      avlnode<T>* n = iterative_search(root, t);
      if(n == 0) return false;
      
      return node_remove(n);
    }
    catch(std::exception& e){ return false; }
  }
  
  
  // remove()s minimum value and saves it to variable t
  // or returns false if tree is empty
  template <typename T>
  bool avltree<T>::remove_minimum(T& t) 
  {
    try{    
      avlnode<T>* n = min_value(root);
      if(n == 0) return false;
      t = n->value;
      
      return node_remove(n);
    }
    catch(std::exception& e){ return false; }      
  }
  
  
  // remove()s maximum value and saves it to variable t
  // or returns false if tree is empty
  template <typename T>
  bool avltree<T>::remove_maximum(T& t) 
  {
    try{    
      avlnode<T>* n = max_value(root);
      if(n == 0) return false;	
      t = n->value;
      
      return node_remove(n);
    }
    catch(std::exception& e){ return false; }      
  }
  
  
  // search()es for given value, updates value and
  // returns true in case of success
  template <typename T>
  bool avltree<T>::search(T& value) const 
  {
    try{
      avlnode<T>* t = iterative_search(root, value);
      if(t == 0)
	return false;
      
      value = t->value;
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
    
  template <typename T>
  bool avltree<T>::order_search(unsigned int order, T& value) const 
  {
    try{
      if(order >= root->numNodes)
	return false;
      
      // search must be successfull or there's a bug
      recursive_order_search(root, order, value);
      
      return true;
    }
    catch(std::exception& e){ return false; }      
  }
  
  // clear()s tree from any data
  template <typename T>
  void avltree<T>::clear() 
  {
    free_subtree(root);
    root = dummy;
  }
  
  
  // returns maximum & minimum keys/objects from the tree
  template <typename T>
  T& avltree<T>::maximum() const 
  {
    avlnode<T> *k = max_value(root);
    if(!k) throw std::logic_error("no maximum value: empty tree");
    return k->value;      
  }
  
  
  template <typename T>
  T& avltree<T>::minimum() const 
  {
    avlnode<T> *k = min_value(root);  
    if(!k) throw std::logic_error("no minimum value: empty tree");
    return k->value;      
  }
  
  
  template <typename T>
  unsigned int avltree<T>::size() const 
  {
    if(root)
      return root->numNodes;
    else
      return 0;
  }
  
  
  template <typename T>
  bool avltree<T>::list() const 
  {
    std::vector<avlnode<T>*> A, B;      
    A.push_back(root);
    
    while(A.size() != 0 || B.size() != 0){
      
      if(A.size() > 0){
	
	for(unsigned int i=0;i<A.size();i++){
	  if(A[i]){
	    std::cout << A[i]->value 
		      << "|" << A[i]->nodes();
	    
	    if(A[i]->left != dummy)
	      std::cout << "," << A[i]->left->value;
	    else
	      std::cout << ",*";
	    
	    if(A[i]->right != dummy)
	      std::cout << "," << A[i]->right->value;
	    else
	      std::cout << ",*";
	    
	    std::cout << "  ";

	    if(A[i]->left  != dummy) B.push_back(A[i]->left);
	    else B.push_back(0);
	    if(A[i]->right != dummy) B.push_back(A[i]->right);
	    else B.push_back(0);
	  }
	  else std::cout << "*|*,*,*  ";
	}
	
	std::cout << std::endl;
	A.clear();
      }
      
      else if(B.size() > 0){	  
	for(unsigned int i=0;i<B.size();i++){
	  if(B[i]){
	    std::cout << B[i]->value 
		      << "|" << B[i]->nodes();
	    
	    if(B[i]->left != dummy)
	      std::cout << "," << B[i]->left->value;
	    else
	      std::cout << ",*";
	    
	    if(B[i]->right != dummy)
	      std::cout << "," << B[i]->right->value;
	    else
	      std::cout << ",*";
	    
	    std::cout << "  ";    

	    // "|" << B[i]->black  << "  ";
	    if(B[i]->left  != dummy) A.push_back(B[i]->left);
	    else A.push_back(0);	      
	    if(B[i]->right != dummy) A.push_back(B[i]->right);
	    else A.push_back(0);
	  }
	  else std::cout << "*|*,*,*  ";
	}
	
	std::cout << std::endl;
	B.clear();
      }
    }
    
    return true;
  }
  
  
  // lists all elements in order (and checks all links are ok)
  template <typename T>
  bool avltree<T>::ordered_list() const 
  {
    
#if 0
    T value;
    
    // tests ordered search
    for(unsigned int i=0;i<size();i++){
      if(order_search(i, value))
	std::cout << value << " ";
      else
	std::cout << "*?* ";
    }
#endif
    
    avlnode<T>* n;
    
    n = min_value(root);
    
    while(n != 0 && n != dummy){
      std::cout << n->value << " ";
      n = tree_successor(n);
    }
    
    std::cout << std::endl;
    return true;
  }
  
  
  /************************************************************/
  
  template <typename T>
  bool avltree<T>::insert(avlnode<T>* n) 
  {
    try{
      // inserts node to tree + height updates
      if(!basic_insert(n)) return false;
      
      avlnode<T>* y = n;
      avlnode<T>* z = n->parent;
      
      if(n == root) return true;
      
      int d;
      
      // travels from inserted node parent up the tree 
      // and fixes unbalancenedness
      while(z != dummy){
	
	d = z->left->hdifference(*(z->right)); /* left minus right */
	
	if(n->value < z->value){ // inserted into left subtree
	  if(d > 1){ // deepness of left is too big
	    
	    if(y->left->hdifference(*(y->right)) >= 1){
	      
	      // updates left/right pointers
	      z->left = y->right;
	      y->right = z;
	      
	      // updates parent pointers
	      if(z->left != dummy) z->left->parent = z;
	      y->parent = z->parent;
	      z->parent = y;
	      
	      if(y->parent == dummy){
		root = y;
	      }
	      else{
		if(y->parent->left == z)
		  y->parent->left = y;
		else
		  y->parent->right = y;
	      }
	      
	      
	      // updates node numbers
	      y->numNodes = z->numNodes;
	      z->numNodes = z->left->numNodes + z->right->numNodes + 1;		
	      
	      // changes / updates z
	      z = y;
	      // y is not correct, not used untill next iter.
	    }
	    else{ // double rotation (other cases not possible (?!?!))
	      
	      avlnode<T>* yr = y->right;
	      
	      // updates left/right pointers
	      y->right = yr->left;
	      yr->left = y;
	      z->left  = yr->right;
	      yr->right = z;
		
	      // updates parent pointers
	      yr->parent = z->parent;
	      y->parent  = yr;
	      z->parent = yr;
	      if(y->right != dummy)  y->right->parent = y;		
	      if(z->left  != dummy)  z->left->parent = z;		
	      
	      if(yr->parent == dummy){
		root = yr;
	      }
	      else{
		if(yr->parent->left == z)
		  yr->parent->left = yr;
		else
		  yr->parent->right = yr;
	      }
	      
	      // updates node numbers
	      z->numNodes = z->left->numNodes + z->right->numNodes + 1;
	      y->numNodes = y->left->numNodes + y->right->numNodes + 1;
	      yr->numNodes = yr->left->numNodes + yr->right->numNodes + 1;
	      
	      // changes / updates z
	      z = yr;
	      // y is not correct, not used untill next iter.
	    }
	  }
	}
	else{ // node inserted into right subtree
	  
	  if(d < -1){ // deepness of right is too big
	    if(y->left->hdifference(*(y->right)) <= -1){
	      
	      // updates left/right pointers
	      z->right = y->left;
	      y->left = z;
	      
	      // updates parent pointers
	      if(z->right != dummy) z->right->parent = z;
	      y->parent = z->parent;
	      z->parent = y;
	      
	      if(y->parent == dummy){
		root = y;
	      }
	      else{
		if(y->parent->left == z)
		  y->parent->left = y;
		else
		  y->parent->right = y;
	      }
	      
	      
	      // updates node numbers
	      y->numNodes = z->numNodes;
	      z->numNodes = z->left->numNodes + z->right->numNodes + 1;		
	      
	      // changes / updates z
	      z = y;
	      // y is not correct, not used untill next iter.
	    }
	    else{
	      avlnode<T>* yl = y->left;
	      
	      // updates left/right pointers
	      y->left = yl->right;
	      yl->right = y;
	      z->right  = yl->left;
	      yl->left = z;
	      
	      // updates parent pointers
	      yl->parent = z->parent;
	      y->parent  = yl;
	      z->parent = yl;
	      
	      if(y->left   != dummy) y->left->parent = y;		
	      if(z->right  != dummy) z->right->parent = z;
	      
	      if(yl->parent == dummy){
		root = yl;
	      }
	      else{
		if(yl->parent->left == z)
		  yl->parent->left = yl;
		else
		  yl->parent->right = yl;
	      }
	      
	      // updates node numbers
	      z->numNodes = z->left->numNodes + z->right->numNodes + 1;
	      y->numNodes = y->left->numNodes + y->right->numNodes + 1;
	      yl->numNodes = yl->left->numNodes + yl->right->numNodes + 1;
	      
	      // changes / updates z
	      z = yl;
	      // y is not correct, not used untill next iter.
	    }
	  }
	}
	
	y = z;
	z = z->parent;
      }
      
      
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  bool avltree<T>::node_remove(avlnode<T>* z) 
  {
    try{	
      avlnode<T> *y = 0, *x = 0;
      
      // y will be removed this can
      // be z or successor of z
      if(z->left == dummy || z->right == dummy){
	if(z->left == dummy && z->right == dummy)
	  std::cout << "removal of leaf node." << std::endl;
	y = z;
      }
      else{
	y = tree_successor(z);
      }
      
      
      if(y == 0)
	return false;
      
      
      
      if(y->left != dummy)
	x = y->left;
      else
	x = y->right;
      
      
      if(y->parent == dummy)
	root = x;
      else{
	if(y == y->parent->left)
	  y->parent->left = x;
	else
	  y->parent->right = x;	
      }
      
      if(x != dummy){
	x->parent = y->parent;
	recalculate_numnodes(x);
      }
      else{
	recalculate_numnodes(y->parent);
	x = y->parent;
      }
      
      
      // copies correct values
      if(y != z){
	// exchanges node pointers, this doesn't copy values

	// updates parent pointers
	if(z->parent != dummy){
	  if(z == z->parent->left)
	    z->parent->left = y;
	  else
	    z->parent->right = y;
	}
	
	if(z == root) root = y;
	y->parent = z->parent;
	
	// swaps children
	
	// children's parent pointers
	if(z->left != dummy)
	  z->left->parent = y;
	if(z->right != dummy)
	  z->right->parent = y;
	
	y->left = z->left;
	y->right = z->right;
	
	y->numNodes = z->numNodes;

	//n = y;  // saves moved value's node pointer [for height fixup]
	y = z;  // now:  y == z
      }
      
      
      // correct numNodes and height differences
      // starts from the removed node's parent node x
      // and goes up to root
      
      z = x;
      x = y;
      y = z;
      z = z->parent;
      
      // z = y->parent;
      // x = y; // saves to be removed pointer
      
      
      int d;
      
      // almost same as in insert. [*] is added and n -> x
      while(z != dummy){
	// updates number of nodes [*]
	z->numNodes = z->left->numNodes + z->right->numNodes + 1;
	
	d = z->left->hdifference(*(z->right)); /* left minus right */
	
	std::cout << "d = " << d << std::endl;
	
	if(d > 1){ // deepness of left is too big
	  
	  if(y->left->hdifference(*(y->right)) >= 1){
	    
	    // updates left/right pointers
	    z->left = y->right;
	    y->right = z;
	    
	    // updates parent pointers
	    if(z->left != dummy)
	      z->left->parent = z;
	    
	    y->parent = z->parent;
	    z->parent = y;
	    
	    if(y->parent == dummy){
	      root = y;
	    }
	    else{
	      if(y->parent->left == z)
		y->parent->left = y;
	      else
		y->parent->right = y;
	    }
	    
	    
	    // updates node numbers
	    y->numNodes = z->numNodes;
	    z->numNodes = z->left->numNodes + z->right->numNodes + 1;		
	    
	    // changes / updates z
	    z = y;
	    // y is not correct, not used untill next iter.
	  }
	  else{ // double rotation (other cases not possible (?!?!))
	    avlnode<T>* yr = y->right;
	    
	    // updates left/right pointers
	    y->right = yr->left;
	    yr->left = y;
	    z->left  = yr->right;
	    yr->right = z;
	    
	    // updates parent pointers
	    yr->parent = z->parent;
	    y->parent  = yr;
	    z->parent = yr;
	    
	    if(y->right != dummy)
	      y->right->parent = y;		
	    if(z->left  != dummy)
	      z->left->parent = z;
	    
	    if(yr->parent == dummy){
	      root = yr;
	    }
	    else{
	      if(yr->parent->left == z)
		yr->parent->left = yr;
	      else
		yr->parent->right = yr;
	    }
	    
	    // updates node numbers
	    z->numNodes = z->left->numNodes + z->right->numNodes + 1;
	    y->numNodes = y->left->numNodes + y->right->numNodes + 1;
	    yr->numNodes = yr->left->numNodes + yr->right->numNodes + 1;
	    
	    // changes / updates z
	    z = yr;
	    // y is not correct, not used untill next iter.
	  }
	}
	
	if(d < -1){ // deepness of right is too big
	  if(y->left->hdifference(*(y->right)) <= -1){
	    
	    // updates left/right pointers
	    z->right = y->left;
	    y->left = z;
	    
	    // updates parent pointers
	    if(z->right != dummy)
	      z->right->parent = z;
	    
	    y->parent = z->parent;
	    z->parent = y;
	    
	    if(y->parent == dummy){
	      root = y;
	    }
	    else{
	      if(y->parent->left == z)
		y->parent->left = y;
	      else
		y->parent->right = y;
	    }
	    
	    
	    // updates node numbers
	    y->numNodes = z->numNodes;
	    z->numNodes = z->left->numNodes + z->right->numNodes + 1;		
	    
	    // changes / updates z
	    z = y;
	    // y is not correct, not used untill next iter.
	  }
	  else{
	    avlnode<T>* yl = y->left;
	    
	    // updates left/right pointers
	    y->left = yl->right;
	    yl->right = y;
	    z->right  = yl->left;
	    yl->left = z;
	    
	    // updates parent pointers
	    yl->parent = z->parent;
	    y->parent  = yl;
	    z->parent = yl;
	    
	    if(y->left   != dummy)
	      y->left->parent = y;		
	    if(z->right  != dummy)
	      z->right->parent = z;
	    
	    if(yl->parent == dummy){
	      root = yl;
	    }
	    else{
	      if(yl->parent->left == z)
		yl->parent->left = yl;
	      else
		yl->parent->right = yl;
	    }
	    
	    // updates node numbers
	    z->numNodes = z->left->numNodes + z->right->numNodes + 1;
	    y->numNodes = y->left->numNodes + y->right->numNodes + 1;
	    yl->numNodes = yl->left->numNodes + yl->right->numNodes + 1;
	    
	    // changes / updates z
	    z = yl;
	    // y is not correct, not used untill next iter.
	  }
	}
	
	
	// updates number of nodes [*]
	if(z != dummy)
	  z->numNodes = z->left->numNodes + z->right->numNodes + 1;
      
	y = z;
	z = z->parent;
      }
      
      delete x;
      
      return true;	
    }
    catch(std::exception& e){ return false; }
  }
  
  
  /* basic binary tree algorthms */
  
  // normal tree insert + height updating
  template <typename T>
  bool avltree<T>::basic_insert(avlnode<T>* z) 
  {
    try{
      avlnode<T>* y = dummy;
      avlnode<T>* x = root;
      
      while(x != dummy){
	y = x;
	x->numNodes++;
	if(z->value < x->value) x = x->left;
	else x = x->right;
      }
      
      z->parent = y;
      
      if(y == dummy){
	root = z;
      }
      else{
	if(z->value < y->value) y->left = z;
	else y->right = z;
      }
      
      z->left = dummy;
      z->right = dummy;
      
      // node has been inserted
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  avlnode<T>* avltree<T>::tree_search(avlnode<T>* x, T& value) const 
  {
    if(x == 0) return 0;
    if(x->value == value) return x;
    
    if(value < x->value)
      return tree_search(x->left, value);
    else
      return tree_search(x->right, value);
  }
  
  
  template <typename T>
  avlnode<T>* avltree<T>::iterative_search(avlnode<T>* x, T& value) const 
  {
    if(x == 0) return 0;
    if(x->value == value) return x;  
    
    while(1){
      if(value < x->value) x = x->left;
      else x = x->right;
      
      if(x == dummy) return 0;
      if(x->value == value) return x;    
    }      
  }
  
  
  
  template <typename T>
  bool avltree<T>::recursive_order_search(avlnode<T>* n,
					  unsigned int order,
					  T& value) const 
  {
    // note: tail recursion optimizations in gcc
    // should be able to make this iterative
    
    if(order < n->left->numNodes){
      return recursive_order_search(n->left, order, value);
    }
    else if(order >= n->left->numNodes + 1){
      // correct order in right subtree
      order -= (n->left->numNodes + 1);
      
      return recursive_order_search(n->right, order, value);
    }
    else{
      value = n->value;
      return true;
    }
  }
  
  
  template <typename T>
  avlnode<T>* avltree<T>::max_value(avlnode<T>* x) const 
  {
    if(x == 0) return 0;      
    while(x->right != dummy) x = x->right;
    return x;
  }
  
  
  template <typename T>
  avlnode<T>* avltree<T>::min_value(avlnode<T>* x) const 
  {
    if(x == 0) return 0;
    while(x->left != dummy) x = x->left;
    return x;
  }
  
  
  template <typename T>
  avlnode<T>* avltree<T>::tree_successor(avlnode<T>* x) const 
  {
    if(x == 0) return 0;
    if(x->right != dummy)
      return min_value(x->right);
    
    avlnode<T>* y = x->parent;
    
    if(y == dummy) return 0;
    
    while(x == y->right){
      x = y;
      y = y->parent;
      
      if(y == dummy) break;
    }
    
    if(y == dummy) return 0;
    return y;
  }
  
  
  template <typename T>
  void avltree<T>::free_subtree(avlnode<T>* x) 
  {
    if(x != dummy){
      free_subtree(x->left);
      free_subtree(x->right);
      try{ delete x; }catch(std::exception& e){ }
    }
  }    
  
  
  template <typename T>
  void avltree<T>::recalculate_numnodes(avlnode<T>* x)
  {
    if(x == 0)
      return;
    
    while(x != dummy){
      x->numNodes = x->left->numNodes + x->right->numNodes + 1;
      x = x->parent;
    }
  }
  
};


#endif
