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
      rbtree() throw();
      ~rbtree() throw();
      
      bool insert(const T& t) throw();  
      bool remove(T& t) throw();
      
      bool search(T& value) const throw();
      T& maximum() const throw(std::logic_error);
      T& minimum() const throw(std::logic_error);
      
      void clear() throw();
      
      unsigned int size() const throw();
      
      bool list() const throw();
      
#if 0	
      // iterators
      iterator begin() throw();
      iterator end() throw();
      const_iterator begin() const throw();
      const_iterator end() const throw();
      
      iterator i;
      i++;
      i--;
      *i; (returns T)
	    
	    remove(i);  
#endif
      
    private:
      friend class rbnode<T>;
      
      bool insert(rbnode<T>* x) throw();
      bool basic_insert(rbnode<T>* z) throw(); // normal tree insert
      
      rbnode<T>* tree_search(rbnode<T>* x, T& value) const throw();  
      rbnode<T>* iterative_search(rbnode<T>* x, T& value) const throw();
      
      rbnode<T>* max_value(rbnode<T>* x) const throw();
      rbnode<T>* min_value(rbnode<T>* x) const throw();
      
      rbnode<T>* tree_successor(rbnode<T>* x) const throw();
      
      bool remove(rbnode<T>* z) throw();
      void rb_delete_fixup(rbnode<T>* x) throw();
      
      /* deletes subtrees of x and x itself */
      void free_subtree(rbnode<T>* x) throw();
      
      unsigned int numberOfNodes;
      
      rbnode<T>* root;
    };
  
}
    
#include "rbtree.cpp"

#endif
