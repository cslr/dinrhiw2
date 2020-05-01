/* 
 * avl tree supporting O(log n) Nth number search
 * 
 * this variation keeps track of exact number of nodes
 * in a given subtree. -> fast find(Nth number from list) O(log n).
 * 
 * cannot be done with red black trees. keeping track of number
 * of nodes in subtrees -> data can be also used for avl tree.
 */

#ifndef avltree_h
#define avltree_h

#include "tree.h"
#include "avlnode.h"
#include <exception>
#include <stdexcept>

namespace whiteice
{
  template <typename T>
    class avlnode;
  
  
  template <typename T>
    class avltree : public tree<T>
    {
    public:
      avltree();
      ~avltree();
      
      // inserts value to search tree
      bool insert(const T& t) ;
      
      // remove()s value from tree
      bool remove(T& t) ;
      
      // remove()s minimum value and saves it to variable t
      // or returns false if tree is empty
      bool remove_minimum(T& t) ;
      
      // remove()s maximum value and saves it to variable t
      // or returns false if tree is empty
      bool remove_maximum(T& t) ;
      
      // search()es for given value, updates value and
      // returns true in case of success
      bool search(T& value) const ;
      
      // search()es for 'order'th value from tree
      // this runs in O(log n) and is reason to use this
      // variant of avl search tree
      // order == 0 finds 1st smallest value
      // order == size()-1 finds biggest value
      bool order_search(unsigned int order, T& value) const ;
      
      // returns maximum & minimum keys/objects from the tree
      T& maximum() const ;
      T& minimum() const ;
      
      // clears tree from data
      void clear() ;
      
      unsigned int size() const ;
      
      // prints tree list
      bool list() const ;
      
      // prints tree data in order
      bool ordered_list() const ;
      
    private:
      friend class avlnode<T>;
      
      bool insert(avlnode<T>* n) ;
      bool node_remove(avlnode<T>* z) ;
      
      /* basic binary tree algorthms */	
      // normal tree insert
      bool basic_insert(avlnode<T>* z) ;
      
      avlnode<T>* tree_search(avlnode<T>* x, T& value) const ;  
      avlnode<T>* iterative_search(avlnode<T>* x, T& value) const ;
      
      bool recursive_order_search(avlnode<T>* n,
				  unsigned int order,
				  T& value) const ;
      
      avlnode<T>* max_value(avlnode<T>* x) const ;
      avlnode<T>* min_value(avlnode<T>* x) const ;
      
      avlnode<T>* tree_successor(avlnode<T>* x) const ;
      
      void free_subtree(avlnode<T>* x) ;
      
      // recalculates numNodes starting from given node
      // up to root
      void recalculate_numnodes(avlnode<T>* x);
      
      
      avlnode<T>* root;
      
      avlnode<T>* dummy;
    };
  
};

#include "avltree.cpp"

#endif


