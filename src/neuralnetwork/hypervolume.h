/*
 * generic implementation
 * of hypervolumes which consist of
 * 1-d std::vertex containers (which n-dim ouput surface som etc. can use)
 * atomic unit is a template parameter
 */

#include <vector>
#include <stdexcept>

#ifndef hypervolume_h
#define hypervolume_h


namespace whiteice
{

  template <typename T>
    class hypervolume_node; // either 1-d container or atomic unit
  
  
  template <typename T>
    class hypervolume
    {
    public:
      // creates hypercube
      hypervolume(unsigned int dim, unsigned int len) ;
      // hyperrectangle [0..L0-1][0..L1-1] ..
      hypervolume(const std::vector<unsigned int>& lengths) ;
      ~hypervolume() ;
      
      T& operator[](const std::vector<unsigned int>& index) ;
      const T& operator[](const std::vector<unsigned int>& index) const ;
      
      unsigned int size(unsigned int index) const ;
      
      inline hypervolume_node<T>& operator[](unsigned int index) ;
      inline const hypervolume_node<T>& operator[](unsigned int index) const ;
      
    private:
      
      hypervolume_node<T> *root;  
      unsigned int dimension;  
    };
  
  /**************************************************/
  
  template <typename T>
    class hypervolume_node
    {
    public:
      hypervolume_node(unsigned int node_size,
		       unsigned int dimension) ; // 0 = data, 1... = nodelist
      hypervolume_node(const std::vector<T>& lengths,
		       unsigned int index = 0) ;
      ~hypervolume_node() ;
      
      inline hypervolume_node<T>& operator[](unsigned int index) ;
      inline const hypervolume_node<T>& operator[](unsigned int index) const ;
      
      unsigned int size() const ;
      
      T& value() ;
      const T& value() const ;
      
    private:
      
      union{
	std::vector<hypervolume_node<T>*> *nodelist;
	T *data;
      };
      
      bool leaf_node; // atomic unit part
    };
  
}
  
#include "hypervolume.cpp"

#endif
