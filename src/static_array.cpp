/* 
 * old code - 
 * coded when there wasn't totally freely
 * usable vector class on some platforms
 *
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */

#ifndef static_array_cpp
#define static_array_cpp

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "static_array.h"

namespace whiteice
{
  
  /*
   * creates and allocates memory for static array of size n
   */
  template <typename D, typename T>
  static_array<D,T>::static_array(const T& n)
  {
    if(n > 0){
      array_memory = (D*)calloc(n , sizeof(D));
      
      if(!array_memory) throw std::bad_alloc();
      
      
      size_of_array = n;
      
      for(int i=0;i<n;i++)
	(*this)[i] = T(0);
    }
    else{
      size_of_array = 0;
      array_memory = 0;
    }
  }
  
  
  /*
   * allocates memory and creates copy of given array.
   * copying of values are done with "=" operator of given datatype.
   */
  template <typename D, typename T>
  static_array<D,T>::static_array(const array<D,T>& a)
  {
    if(a.size() > 0){
      
      array_memory = (D*)calloc( a.size() , sizeof(D) );
      if(!array_memory) throw std::bad_alloc();
      
      for(int i = 0;i < a.size(); i++)
	array_memory[i] = a[i];
      
      size_of_array = a.size();
    }
    else{
      size_of_array = 0;
      array_memory = 0;
    }
  }
  
  
  /* destructor */
  template <typename D, typename T>
  static_array<D,T>::~static_array()
  {
    if(array_memory != 0) free(array_memory);
    size_of_array = 0;
  }
  
  
  /*
   * returns size of array
   */
  template <typename D, typename T>
  unsigned int static_array<D,T>::size() const throw()
  {
    return size_of_array;
  }
  
  
  /*
   * resizes array
   */
  template <typename D, typename T>
  bool static_array<D,T>::resize(const T& n) throw()
  {
    if(n < 0 || n == size_of_array) return true;
    
    D* ptr = (D*)realloc(array_memory, n * sizeof(D));
    
    if(ptr == 0){
      return false;
    }
    else{
      size_of_array = n;
      array_memory = ptr;
      return true;
    }
  }
  
  
  template <typename D, typename T>
  void static_array<D,T>::clear() throw(){ resize(0); }
  
  
  /*
   * returns Nth element
   */
  template <typename D, typename T>
  D& static_array<D,T>::operator[](const T& n) throw(std::out_of_range){
    if(n < 0 || n >= size_of_array) throw std::out_of_range("index out of range");
    return array_memory[n];
  }
  
  /*
   * returns Nth element
   */
  template <typename D, typename T>
  const D& static_array<D,T>::operator[](const T& n) const throw(std::out_of_range){
    if(n < 0 || n >= size_of_array) throw std::out_of_range("index out of range");
    return array_memory[n];
  }
  
}

#endif




