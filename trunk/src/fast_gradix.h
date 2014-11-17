/*
 * bits based in-place generic forward radix sort
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
 *
 */

#ifndef fast_gsort_h
#define fast_gsort_h

#include <new>
#include <algorithm>
#include "dynamic_bitset.h"


// interface class/datatype
class radixnode {
 public:
  virtual ~radixnode(){ }
  
  virtual const whiteice::dynamic_bitset& radixkey() throw() = 0;
};


/*
 * fast_gradix works with list of pointers
 * pointing to datastructures implementing radixnode interface
 */
class fast_gradix
{
 public:
  fast_gradix(){ table = 0; tableSize = 0; }
  
  fast_gradix(radixnode** orig_table, unsigned int size){
    table = orig_table;
    tableSize = size;
  }
  
  virtual ~fast_gradix(){ }
  
  /* does radix sort */
  virtual bool sort(){
    if(tableSize <= 0)
      return true;
    
    recursive_sort(0, tableSize,
		   table[0]->radixkey().size() - 1);
    
    return true;
  }
  
  
  unsigned int size() const throw(){ return tableSize; }
  
  
  /* changes source data table pointer and size of
   * the table. if new_table is NULL no changes to pointer
   * is made
   */
  bool set(radixnode** new_table, unsigned int new_size){
    tableSize = new_size;
    table = new_table;
    
    return true;
  }
  
 protected:
  radixnode** table;
  unsigned int tableSize;
  
  // radix sort
  virtual void recursive_sort(int a, int b, unsigned int bitnumber)
  {
    if(bitnumber == 0 || a == b) return; // done
    int min = a, max = b-1;
    int i, j;
    
    
    while(1){
      // finds lower value for swap
      for(i=min;i<=max;i++)
	if(!(table[i]->radixkey()[bitnumber]))
	  break;
      
      min = i;
      
      // finds higher value for swap
      for(j=max;j>=min;j--)
	if(table[j]->radixkey()[bitnumber])
	  break;

      max = j;
      
      if(i < j)
	std::swap(table[i], table[j]);
      else
	break;
    }
    
    recursive_sort(a, min, bitnumber-1);
    recursive_sort(min, b, bitnumber-1);
  }
  
};

#endif




