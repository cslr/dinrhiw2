/*
 * priority queue implementation
 * extracts maximum value
 *
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */
#ifndef priority_queue_h
#define priority_queue_h


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <stdexcept>

namespace whiteice
{
  template <typename T>
    class priority_queue
    {
    public:
      priority_queue(unsigned int initial_size = 1)
      {
	numElements = 0;
	tableSize = initial_size;
	table = (T*)malloc(sizeof(T)*(initial_size + 1));
	if(!table) tableSize = 0;
      }
      
      ~priority_queue(){ if(table) free(table); }
      
      /* returns number of elements in queue */
      unsigned int size() const { return numElements; }  
      bool empty() const { return (numElements == 0); }
      
      T& maximum() {
	if(!numElements)
	  throw std::logic_error("empty queue");
	return table[1];
      }
      
      const T& maximum() const {
	if(!numElements)
	  throw std::logic_error("empty queue");
	return table[1];
      }
      
      // returns false if operation fails
      bool insert(const T& value) {
	if(numElements == tableSize){
	  unsigned int numNew = (unsigned)(tableSize*0.10);
	  if(numNew < 256) numNew = 256;
	  
	  // with paging hardware realloc() should never fail.
	  // just map memory to different address space if there's
	  // other data after current max table address
	  
	  T* t = (T*)realloc(table, (tableSize + numNew)*sizeof(T));
	  if(!t) return false;
	  tableSize++;
	  table = t;
	}
	
	numElements++;
	int index = numElements;
	
	while(index > 1 && table[index>>1] < value){
	  table[index] = table[index>>1];
	  index = index >> 1;
	}
	
	table[index] = value;
	
	return true;
      }
      
      /* extracts maximum value */
      T extract() {
	if(numElements < 1)
	  throw std::logic_error("empty queue");
	
	T max = table[1];
	table[1] = table[numElements];
	numElements--;
	heapify(1);
	
	return max;
      }
      
    private:
      
      void heapify(unsigned int index){
	unsigned int left  = index<<1;
	unsigned int right = (index<<1) + 1;
	unsigned int largest = index;
	
	if(left  <= numElements && table[left] > table[index])
	  largest = left;
	
	if(right <= numElements && table[right] > table[largest])
	  largest = right;
	
	if(largest != index){
	  T tmp = table[largest];
	  table[largest] = table[index];
	  table[index] = tmp;
	  
	  heapify(largest);
	}
      }
      
      // for debugging
      void print_queue(){    
	int i, j, k, min;
	
	for(i=1,j=0;i<=(int)numElements;i++){
	  min = 1;
	  
	  for(k=0;k<j;k++)
	    min *= 2;
	  
	  if(i>=min){
	    std::cout << std::endl;
	    std::cout << j << ". ";
	    j++;
	  }
	  
	  std::cout << table[i] << " ";
	}
      }
      
      T* table;
      unsigned int numElements;
      unsigned int tableSize;
    };
  
}

#endif
