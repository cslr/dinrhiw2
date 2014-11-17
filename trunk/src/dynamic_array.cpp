/*
 * dynamically sized array
 * uses double linked list.
 * Tomas Ukkonen <tomas.ukkonen@hut.fi>
 *
 * implements
 * array, stack and queue interfaces
 */

#ifndef dynamic_array_cpp
#define dynamic_array_cpp

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "dynamic_array.h"

namespace whiteice
{
  
  /* creates dynamic array and allocates n elements */
  template <typename D, typename T>
  dynamic_array<D,T>::dynamic_array(const T& n)
  {
    size_of_array = 0;
    first = new darray_node;
    last = new darray_node;
    
    /* dummy block */
    first->prev = first;
    last->prev  = first;
    last->next  = last;
    first->next = last;
    
    T i = n;
    
    while(i > 0){
      add_empty_node();
      i--;
    }
  }

  
  /* creates copy of dynamic array with "=" */
  template <typename D, typename T>
  dynamic_array<D,T>::dynamic_array(const array<D,T>& a)
  {
    size_of_array = 0;
    first = new darray_node;
    last = new darray_node;
    
    /* dummy block */
    first->prev = first;
    last->prev  = first;
    last->next  = last;
    first->next = last;
    
    unsigned int i = 0;
    const unsigned int N = a.size();
    
    while(i < N){
      add_node(a[i]);
      i++;
    }
  }
  
  /*
   * destructor of dynamic array
   */
  template <typename D, typename T>
  dynamic_array<D,T>::~dynamic_array(){
    resize(0);
    delete first;
    delete last;
  }
  
  
  /*
   * returns number of elements in dynamic_array
   */
  template <typename D, typename T>
  unsigned int dynamic_array<D,T>::size() const throw()
  {
    return size_of_array;
  }
  
  
  /*
   * resizes dynamic_array
   */
  template <typename D, typename T>
  bool dynamic_array<D,T>::resize(const T& n) throw()
  {
    if(n < 0){
      return false;
    }
    else if(n == size_of_array){
      return true;
    }
    else if(n > size_of_array){ /* adds nodes */
      
      T i = n - size_of_array;
      
      while(i > 0){
	add_empty_node();
	i--;
      }
      
      return true;
    }
    else if(n < size_of_array){
      /* removes nodes */
      
      T i = size_of_array - n;
      darray_node *dn = last->prev, *n;
      
      while(i > 0){
	n = dn->prev;
	delete dn;
	
	dn = n;
	
	i--;
	size_of_array--;
      }
      
      last->prev = dn;
      dn->next = last;
      
      return true;
    }
    
    
    return false;
  }
  
  
  /*
   * returns nth element of dynamic_array
   */
  template <typename D, typename T>
  void dynamic_array<D,T>::clear() throw()
  {
    resize(0);
  }
  
  
  /*
   * returns nth element of dynamic_array
   */
  template <typename D, typename T>
  D& dynamic_array<D,T>::operator[](const T& n) throw(std::out_of_range){
    
    if(n < 0 || n >= size_of_array)
      throw std::out_of_range("index out of range");
    
    /* simple method, optimize later */
    T i = n;
    
    darray_node* dn = first->next;
    
    while(i > 0){
      dn = dn->next;
      i--;
    }
    
    return dn->data;
  }
  
  
  /* returns nth element of const dynamic_array */
  template <typename D, typename T>
  const D& dynamic_array<D,T>::operator[](const T& n) const throw(std::out_of_range){
    
    if(n < 0 || n >= size_of_array)
      throw std::out_of_range("index out of range");
    
    /* simple method, optimize later */
    T i = n;
    
    darray_node* dn = first->next;
    
    while(i > 0){
      dn = dn->next;
      i--;
    }
    
    return dn->data;	
  }
  
  
  /*
   * adds data to the end of the list
   */
  template <typename D, typename T>
  bool dynamic_array<D,T>::push(const D& d) throw(){    
    try{
      add_node(d);
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  /*
   * pops data from the end of the list
   */
  template <typename D, typename T>
  D dynamic_array<D,T>::pop() throw(std::logic_error){    
    if(size_of_array <= 0)
      throw std::logic_error("cannot pop from empty stack");
    
    return remove_node();
  }
  
  
  /*
   * adds data element to the end of the list
   */
  template <typename D, typename T>
  bool dynamic_array<D,T>::enqueue(const D& data) throw(){
    try{
      add_node(data);
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  /*
   * removes data element from the beginning of the list
   */
  template <typename D, typename T>
  D dynamic_array<D,T>::dequeue() throw(std::logic_error){
    if(size_of_array <= 0)
      throw std::logic_error("cannot dequeue from empty queue");
    
    /* removes 1st node */    
    darray_node *dn = first->next;
    
    assert(dn != last);
    assert(dn != first);
    
    first->next = dn->next;
    dn->next->prev = first;
    D data = dn->data;
    size_of_array--;
    delete dn;
    
    return data;
  }
  
  
  /*
   * adds empty node to the end of list
   */
  template <typename D, typename T>
  void dynamic_array<D,T>::add_empty_node() throw(std::bad_alloc){
    darray_node *dn = new darray_node;
    last->prev->next = dn;
    last->prev->next->prev = last->prev;
    last->prev->next->next = last;
    last->prev = dn;
    
    size_of_array++;
  }
  
  
  /*
   * adds node with data to the end of list
   */
  template <typename D, typename T>
  void dynamic_array<D,T>::add_node(const D& data) throw(std::bad_alloc){
    darray_node *dn = new darray_node;
    dn->data = data;
    
    last->prev->next = dn;
    last->prev->next->prev = last->prev;
    last->prev->next->next = last;
    last->prev = dn;
    
    size_of_array++;
  }
  
  
  /* removes nth node from list */
  template <typename D, typename T>
  D dynamic_array<D,T>::remove_node(const T& n) throw(){
    T i = n;
    darray_node *dn = first->next;
    
    while(i > 0){
      dn = dn->next;
      i--;	
    }
    
    assert(dn != first);
    assert(dn != last);
    
    dn->prev->next = dn->next;
    dn->next->prev = dn->prev;
    D data = dn->data;
    delete dn;
    
    size_of_array--;
    
    return data;
  }
  
  
  /* removes last node from the list */
  template <typename D, typename T>
  D dynamic_array<D,T>::remove_node() throw(){
    return remove_node(size_of_array - 1);
  }
  
}
  
  
#endif

