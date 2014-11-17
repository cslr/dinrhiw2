/*
 * bits based in-place forward radix sort for std::vector<T>s
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 *
 * see "fast_radix.h" for details
 *
 * NOTE: zero thing is make things slow and
 * because only data structure which uses it is dynamic_bitset
 * it's probably good idea to write separated fast_radix for
 * dynamic_bitsets
 *
 */
#ifndef fast_sortv_h
#define fast_sortv_h

#include <new>
#include <algorithm>


template <typename T>
class fast_radixv
{
 public:
  fast_radixv(){ }  
  virtual ~fast_radixv(){ }

  /* non reentrant, mask must be number of bits in T long
   * and must have highest bit set
   * zero value with no bits set
   */
  virtual bool sort(std::vector<T>& data, T mask, T zero)
  {
    this->array = &data;
    this->zero  = zero;
    
    recursive_sort(0, data.size(), mask);
    return true;
  }

 protected:
  std::vector<T>* array;
  T zero;
  
  // in-place radix sort
  virtual void recursive_sort(int a, int b, T mask)
  {
    if(mask == zero || a == b) return; // done
    int min = a, max = b-1;
    int i, j;
    
    while(1){
      // finds lower value for swap
      for(i=min;i<=max;i++)
	if(((*array)[i] & mask) != zero) break;

      min = i;
      
      // finds higher value for swap
      for(j=max;j>=min;j--)
	if(((*array)[j] & mask) == zero) break;

      max = j;
      
      if(i < j) std::swap<T>((*array)[i], (*array)[j]);
      else break;	
    }
    
    recursive_sort(a, min, mask >> 1);
    recursive_sort(min, b, mask >> 1);
  }
  
};

#endif




