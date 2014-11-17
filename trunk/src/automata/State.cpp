
#ifndef AutomataState_cpp
#define AutomataState_cpp

#include <exception>
#include "State.h"


namespace automata
{
  /**********************************************************************/
  
  template <typename T>
  State<T>::State() throw(){ }

  /**********************************************************************/
  
  template <typename T>
  bool State<T>::add(const Edge<T>& e) throw()
  {
    try{
      edges.push_back(e);
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  /**********************************************************************/
  
  template <typename T>
  bool State<T>::remove(unsigned int index) throw()
  {
    try{
      std::vector<T>::iterator i = 
	edges.find(edges[index]);

      if(i == edges.end())
	return false;
      else
	edges.erase(i);

      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  /**********************************************************************/
  
  template <typename T>
  const Edge<T>& State<T>::getEdge(unsigned int index) const throw(std::out_of_range)
  {
    try{
      if(index >= edges.size())
	throw std::out_of_range("Edge<T>::getEdge() - index out of range");
      
      return edges[index];
    }
    catch(std::exception& e){
      throw std::out_of_range("Edge<T>::getEdge() - access error.");
    }
  }
  
  /**********************************************************************/
  
  template <typename T>
  unsigned int State<T>::getNumberOfEdges() const throw()
  {
    return edges.size();
  }
  
  
};


#endif
