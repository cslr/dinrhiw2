

#ifndef graph_h
#define graph_h

#include "graph_interface.h"
#include "graphnode_interface.h"
#include "graph_access_interface.h"

#include "unique_id.h"

#include <map>
#include <list>


namespace blade
{
  template <typename T>
    class graph : 
      public graph_interface<T>,
      public unique_id
    {
    public:

      graph() throw();
      ~graph() throw();
      
      // adds node, returns 0 if there's error, otherwise returns id numbe
      unsigned int add(graphnode_interface<T>* node) throw();

      // adds adjacencies to already inserted nodes
      bool add(unsigned int id,
	       const std::list<unsigned int>& adjacencies,
	       bool unidirectional = false) throw();
      
      bool remove(unsigned int id) throw();
      
      graph_access_interface<T> access() throw();
      const graph_access_interface<T> access() const throw();

      // after this one indexes in graph are topologically sorted
      // (if acyclic and/or returns true)
      bool topological_sort() throw();

      bool cyclic() throw();
      bool acyclic() throw();
      
      

    private:

      std::string nodedomain;
      
      // list of nodes
      std::map<unsigned int,
	graphnode_interface<T>*> nodes;
      
    };
  
};


#include "graph.cpp"


#endif /* graph_h */

