/*
 * bits based radix sort   O(n) = bits(T)*n
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 *
 * stats from most significant bits
 * before least significant bits (intel)
 * depth first sort
 *
 * => 
 * depth first sort and goes through the
 * the whole data only once so data sizes
 * of individual loops/sorts should be inside cache
 * limits as soon as possible.
 */

#ifndef radix_sort_h
#define radix_sort_h

#include <new>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>


template <typename T>
class radix_sort
{
 public:
  
  // allocates temporary memory, uses orig_table as a
  // data source, size is the size of table
  
  radix_sort(T* orig_table, unsigned int size){
    this->orig_table = orig_table;
    this->tmp_table = (T*)malloc(sizeof(T)*size);
    tableSize = size;
  }
  
  virtual ~radix_sort(){
    if(tmp_table)
      free(tmp_table);
  }

  /* does radix sort */
  virtual bool sort(){
    const int bitsize = sizeof(T)*8; // number of bits (intel)
    T mask = T(1<<(bitsize - 1)); // highest bit
    
    recursive_sort(0, tableSize, mask);
    return true;
  }
  

  unsigned int size() const throw(){ return tableSize; }
  
  /* changes source data table pointer and size of
   * the table. if new_table is NULL no changes to pointer
   * is made
   */
  bool set(T* new_table, unsigned int new_size){
    if(new_table == 0) return false;
    
    T* t = (T*)realloc(tmp_table, sizeof(T)*new_size);
    if(!t) return false;
    
    tmp_table = t;
    tableSize = new_size;
    orig_table = new_table;
    
    return true;
  }
  
 protected:
  T *orig_table;
  T *tmp_table;
  
  unsigned int tableSize;
  
  
  virtual void recursive_sort(unsigned int a, unsigned int b, T mask){
    if(mask == 0 || a == b) return; // run out of bits
    unsigned int min = a, max = b - 1;
    
    for(unsigned int i=a;i<b;i++){
      if(orig_table[i] & mask){
	tmp_table[max] = orig_table[i];
	max--;
      }
      else{
	tmp_table[min] = orig_table[i];
	min++;
      }
    }
    
    std::swap<T*>(orig_table, tmp_table);
    recursive_sort(a, min, ((unsigned)mask)>>1);
    recursive_sort(min, b, ((unsigned)mask)>>1);
    std::swap<T*>(orig_table, tmp_table);
  }
  
};



#endif
