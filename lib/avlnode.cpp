
#ifndef avlnode_cpp
#define avlnode_cpp

#include "avlnode.h"

namespace whiteice
{
  
  template <typename T>
  avlnode<T>::avlnode(const T& v)
  {
    value = v;
    numNodes = 1;
    
    parent = 0;
    left = right = 0;
  }
  
  template <typename T>
  avlnode<T>::~avlnode()
  {
    this->numNodes = 0;
  }
  
  // returns true if node is leaf node
  template <typename T>
  bool avlnode<T>::leaf() const throw()
  {
    return (numNodes == 1);
  }
  
  // returns height of this subtree
  template <typename T>
  unsigned int avlnode<T>::height() const throw()
  {
    unsigned int height = 0;
    unsigned int n = numNodes;
    
    // O(log n)
    while(n != 0){
      n >>= 1;
      height++;
    }
    
    return height;
  }
  
  
  // returns height difference between nodes
  // (this.height - node.height)
  template <typename T>
  int avlnode<T>::hdifference(const avlnode<T>& node) const throw()
  {
    unsigned int a = numNodes;
    unsigned int b = node.numNodes;
    
    int hdelta = 0;
    
    if(a < b){
      if(b == 0) return 0;
      
      while(!equal_height(b,a)){
	hdelta++;
	a = (a << 1) + 1;
      }
      
      return -hdelta;
    }
    else{      
      if(a == 0) return 0;
      
      while(!equal_height(a,b)){
	hdelta++;
	b = (b << 1) + 1;
      }
      
      return hdelta;
    }
  }
  
  /************************************************************/
  
  
  // it is assumed a >= b
  template <typename T>
  bool avlnode<T>::equal_height(unsigned int a, unsigned int b)
    const throw()
  {
    if(a < b)
      return (bool)( (b & (~a)) <= (b & a) );    
    else
      return (bool)( (a & (~b)) <= (a & b) );
  }
  
};


#endif
