/*
 * templated red-black tree
 * template type T must support 
 * "<, ==, !=, >, >=, <=" and "=" and ctor(), ctor(T t) operations
 *
 */
#ifndef rbtree_h
#define rbtree_h

#include "tree.h"
#include "rbnode.h"
          
namespace whiteice
{
  template <typename T>
    class rbtree : public tree<T>
    {
    public:
      rbtree() ;
      ~rbtree() ;
      
      bool insert(const T& t) ;  
      bool remove(T& t) ;
      
      bool search(T& value) const ;
      T& maximum() const ;
      T& minimum() const ;
      
      void clear() ;
      
      unsigned int size() const ;
      
      bool list() const ;
      
#if 0	
      // iterators
      iterator begin() ;
      iterator end() ;
      const_iterator begin() const ;
      const_iterator end() const ;
      
      iterator i;
      i++;
      i--;
      *i; (returns T)
	    
	    remove(i);  
#endif
      
    private:
      friend class rbnode<T>;
      
      bool insert(rbnode<T>* x) ;
      bool basic_insert(rbnode<T>* z) ; // normal tree insert
      
      rbnode<T>* tree_search(rbnode<T>* x, T& value) const ;  
      rbnode<T>* iterative_search(rbnode<T>* x, T& value) const ;
      
      rbnode<T>* max_value(rbnode<T>* x) const ;
      rbnode<T>* min_value(rbnode<T>* x) const ;
      
      rbnode<T>* tree_successor(rbnode<T>* x) const ;
      
      bool remove(rbnode<T>* z) ;
      void rb_delete_fixup(rbnode<T>* x) ;
      
      /* deletes subtrees of x and x itself */
      void free_subtree(rbnode<T>* x) ;
      
      unsigned int numberOfNodes;
      
      rbnode<T>* root;
    };
  
}
    
#include "rbtree.cpp"

#endif
