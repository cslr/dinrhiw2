
#include "hypervolume.h"
#include <new>

#ifndef hypervolume_cpp
#define hypervolume_cpp

/*********************************************************************************/

namespace whiteice
{
  
  template <typename T>
  hypervolume<T>::hypervolume(unsigned int dim,
			      unsigned int len) throw()
  {
    dimension = 0;
    root = new hypervolume_node<T>(len, dim);
    dimension = dim;
  }
  
  
  template <typename T>
  hypervolume<T>::hypervolume(const std::vector<unsigned int>& lengths) throw()
  {
    dimension = 0;
    root = new hypervolume_node<T>(lengths);
    dimension = lengths.size();
  }
  

  template <typename T>
  hypervolume<T>::~hypervolume() throw()
  {
    if(root) delete root;
  }
  
  
  template <typename T>
  inline hypervolume_node<T>& hypervolume<T>::operator[](unsigned int index) throw()
  {
    return (*root)[index];
  }
  
  
  template <typename T>
  inline const hypervolume_node<T>& hypervolume<T>::operator[](unsigned int index) const throw()
  {
    return (*root)[index];
  }
  
  /*********************************************************************************/
  
  // hypervolume node ctor
  template <typename T> // 0 = data, 1... = nodelist
  hypervolume_node<T>::hypervolume_node(unsigned int node_size,
					unsigned int dimension) throw()
  {
    if(dimension == 0){
      nodelist = 0;
      leaf_node = true;
      data = new T;
    }
    else{
      data = 0;
      leaf_node = false;
      nodelist = new std::vector<hypervolume_node*>;
      nodelist->resize(node_size);
      
      for(unsigned int i=0;i<node_size;i++)
	(*nodelist)[i] = 
	  new hypervolume_node(node_size, dimension-1);
    }
  }
  
  
  template <typename T>
  hypervolume_node<T>::hypervolume_node(const std::vector<T>& lengths,
					unsigned int index) throw()
  {
    if(index == lengths.size()){
      nodelist = 0;
      leaf_node = true;
      data = new T;
    }
    else{
      data = 0;
      leaf_node = false;
      nodelist = new std::vector<hypervolume_node*>;
      nodelist->resize(lengths[index]);
      
      for(unsigned int i=0;i<lengths[index];i++){
	(*nodelist)[i] = 
	  new hypervolume_node(lengths, index+1);
      }
    }
  }
  
  
  template <typename T>
  hypervolume_node<T>::~hypervolume_node() throw()
  {
    if(leaf_node && data != 0) delete data;
    if(!leaf_node && nodelist != 0) delete nodelist;
  }
  
  
  // hypervolume data [] access
  template <typename T>
  inline hypervolume_node<T>& hypervolume_node<T>::operator[](unsigned int index) throw()
  {
    if(leaf_node)
      return *this;
    else 
      return *((*nodelist)[index]);
  }
  
  
  template <typename T>
  inline const hypervolume_node<T>& hypervolume_node<T>::operator[](unsigned int index) const throw()
  {
    if(leaf_node) return *this;
    else return *((*nodelist)[index]);
  }
  

  template <typename T>
  unsigned int hypervolume_node<T>::size() const throw()
  {
    if(leaf_node) return 0;
    else return nodelist->size();
  }

  
  // data value access
  template <typename T>
  T& hypervolume_node<T>::value() throw(std::logic_error)
  {
    if(!data) throw std::logic_error("hypervolume_node: only leafs has values");
    return *data;
  }
  
  

  template <typename T>
  const T& hypervolume_node<T>::value() const throw(std::logic_error)
  {
    if(!data) throw std::logic_error("hypervolume_node: only leafs has values");
    return *data;
  }
  
}
  
#endif


