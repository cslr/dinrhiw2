/*
 * node interface (pure virtual) for (type T) graphs
 */

#ifndef graphnode_interface_h
#define graphnode_interface_h

#include <list>

namespace blade
{

  template <typename T>
    class graph<T>;

  template <typename T>
    class graph_access<T>;
  

  template <typename T>
    class graphnode_interface
    {
    public:
      
      // access for adjacency list    
      const std::list<graphnode_interface<T>*>& adjacency() const throw() = 0;
      
      // access for data
      virtual T& data() throw() = 0;
      virtual const T& data() const throw() = 0;
      
    private:

      friend blade::graph<T>;
      friend blade::graph_access<T>;
      
      std::list<graphnode_interface<T>*>& adjacency() throw() = 0;
    };

};


#include "graph.h"
#include "graph_access.h"

#endif // graphnode_inteface_h


