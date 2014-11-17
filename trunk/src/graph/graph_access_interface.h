

#ifndef graph_access_interface_h
#define graph_access_interface_h

namespace blade
{

  #include "graphnode_inteface.h"

  template <typename T>
  class graph_access_interface
  {
  public:
    
    // starts breadth first search
    bool bfs_begin() throw(); // 

    // returns current one and goes to next one
    graphnode_inteface<T>& bfs_next() throw(std::logic_error);
    
    // is there nodes left?
    bool bfs_end() throw();


    // starts depth first search
    bool dfs_begin() throw(); // 

    // returns current one and goes to next one
    graphnode_inteface<T>& dfs_next() throw(std::logic_error);
    
    // is there nodes left?
    bool dfs_end() throw();
    
    unsigned int distance(const graph_access_interface<T>& p1);
    unsigned int distance(const graph_access_interface<T>& p1,
			  const graph_access_interface<T>& p2);
    
  };
};


#endif // graph_access_interface_h

