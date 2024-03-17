/*
 * bits based radix sort for std::vector<T> list
 *
 * see radix_sort.h for details
 */

#ifndef radix_sortv_h
#define radix_sortv_h

#include <new>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>


template <typename T>
class radix_sortv
{
 public:
  
  radix_sortv(){ tmparray = new std::vector<T>; }
  
  virtual ~radix_sortv(){ if(tmparray) delete tmparray; }
  
  /* non reentrant, mask must be number of bits in T long
   * and must have highest bit set
   * zero is value with not bits set
   */
  virtual bool sort(std::vector<T>& data, T mask, T zero)
  {
    this->array = &data;
    tmparray->resize(data.size());
    this->zero = zero;
    
    // (this works correctly with even number of bits)
    recursive_sort(0, data.size(), mask);
    tmparray->resize(0);
    
    return true;
  }
  
 protected:
  std::vector<T>* array;
  std::vector<T>* tmparray;
  T zero;
  
  
  virtual void recursive_sort(unsigned int a, unsigned int b, T mask)
  {
    if(mask == 0 || a == b)
      return; // run out of bits or data
    
    unsigned int min = a, max = b - 1;
    
    for(unsigned int i=a;i<b;i++){
      if(((*array)[i] & mask) != zero){
	(*tmparray)[max] = (*array)[i];
	max--;
      }
      else{
	(*tmparray)[min] = (*array)[i];
	min++;
      }
    }
    
    std::swap<std::vector<T>*>(array, tmparray);
    recursive_sort(a, min, mask >> 1);
    recursive_sort(min, b, mask >> 1);
    std::swap<std::vector<T>*>(array, tmparray);
  }
  
};



#endif
