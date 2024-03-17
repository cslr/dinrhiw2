/*
 * interface for graph implementations
 *
 * todo/fix: add support weights (after primitive first version is done and working)
 */

#ifndef graph_interface_h
#define graph_interface_h

namespace blade
{
  template <typename T>
    class graph_access_interface;
  
  template <typename T>
    class graphnode_interface;


  class graph_interface
  {
  public:
    
    // adds node, returns 0 if there's error, otherwise returns id numbe
    virtual unsigned int add(graphnode_interface<T>* node) throw() = 0;
    
    // adds adjacencies to already inserted nodes
    virtual bool add(unsigned int id,
		     const std::list<unsigned int>& adjacencies,
		     bool unidirectional = false) throw() = 0;
    
    virtual graph_access_interface<T> access() throw() = 0;
    virtual const graph_access_interface<T> access() const throw() = 0;
  };
  
};


#include "graph_access_interface.h"
#include "graphnode_interface.h"


#endif // graph_interface_h

