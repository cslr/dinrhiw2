/*
 * bits based in-place forward radix sort
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 *
 * sorts table in place: starts from min and max limits
 * and goes up / down till both low part and max part
 * has found element with bit on/off, then swaps the values.
 *
 * using insertion sort to handle smallest cases/subproblems
 * seems to cause only slowdowns (contrary to some scientific
 *  sources from on the web - maybe their radix sort
 *  implementation wasn't optimized)
 */
#ifndef fast_sort_h
#define fast_sort_h

#include <new>
#include <algorithm>


template <typename T>
class fast_radix
{
 public:
  fast_radix(T* orig_table, unsigned int size){
    this->table = orig_table;
    tableSize = size;
  }  
  
  virtual ~fast_radix(){ }
  
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
    tableSize = new_size;
    table = new_table;
    
    return true;
  }
  
 protected:
  T *table;
  unsigned int tableSize;
  
  // radix sort
  virtual void recursive_sort(int a, int b, T mask)
  {
    if(!mask || a == b) return; // done
    int min = a, max = b-1;
    int i, j;
    
    while(1){
      // finds lower value for swap
      for(i=min;i<=max;i++)
	if(table[i] & mask) break;

      min = i;
      
      // finds higher value for swap
      for(j=max;j>=min;j--)
	if(!(table[j] & mask)) break;

      max = j;
      
      if(i < j) std::swap<T>(table[i], table[j]);
      else break;	
    }
    
    recursive_sort(a, min, ((unsigned)mask)>>1);
    recursive_sort(min, b, ((unsigned)mask)>>1);
  }
  
  
  /* sorts values { table[i] | b > i >= a } */
  /*
   * void insertion_sort(int a, int b){
   * int i;
   * T key;
   * 
   * for(int j=1;j<b;j++){
   * key = orig_table[j];
   * i = j-1;
   * 
   * while(i > a && orig_table[i] > key){
   * orig_table[i+1] = orig_table[i];
   * i--;
   * }
   * 
   * orig_table[i] = key;
   * }
   *
   */
    

#if 0
    // TOO COMPLICATED, SLOW? - DON'T NEED
    // OVER 2GB numbers at least right now
    
    // in reversed order (than normally) to
    // make special case of a = 0 fast when
    // using unsigned integers (handled as a special
    // case but no if()s in a main loop
    // 
    // cannot have: a<=j;j-- , because
    // j cannot never be negative (a = 0)
    
    if(a == b || a == b+1) return;
    if(a == tableSize-1) return;
    if(b == 1) return; // a = 0 || a = 1
    if(b == 2){
      if(orig_table[0] > orig_table[1]){
	std::swap<T>(orig_table[0], orig_table[1]);
	return;
      }
    }
    
    for(unsigned int j=b-2;a<j;j--){
      key = orig_table[j];
      i = j+1;
      
      while(i < b-1 && orig_table[i] < key){
	orig_table[i-1] = orig_table[i];
	i++;
      }
      
      orig_table[i] = key;
    }
    
    // finally handles case (j == a)
    key = orig_table[a];
    i = a+1;
    
    while(i < b-1 && orig_table[i] < key){
      orig_table[i-1] = orig_table[i];
      i++;
    }
    
    orig_table[i] = key;
    
#endif
  
};

#endif




