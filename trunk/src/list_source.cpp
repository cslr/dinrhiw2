
#ifndef list_source_cpp
#define list_source_cpp

#include <vector>
#include <stdexcept>
#include <exception>

#include "list_source.h"


namespace whiteice
{
  
  template <typename datum>
  list_source<datum>::list_source(std::vector<datum>& _list) : list(_list) { }
  
  template <typename datum>
  list_source<datum>::~list_source(){ }
  
  template <typename datum>
  datum& list_source<datum>::operator[](unsigned int index) throw(std::out_of_range)
  {
    if(index >= list.size())
      throw std::out_of_range("list source: index too big");
    
    return list[index];
  }
  
  template <typename datum>
  const datum& list_source<datum>::operator[](unsigned int index) const throw(std::out_of_range)
  {
    if(index >= list.size())
      throw std::out_of_range("list source: index too big");
    
    return list[index];
  }
  
  template <typename datum>
  unsigned int list_source<datum>::size() const throw()
  {
    return list.size();
  }
  
  
  template <typename datum>
  bool list_source<datum>::good() const throw()
  {
    return true;
  }
  
  template <typename datum>
  void list_source<datum>::flush() const { }
  
}


#endif
