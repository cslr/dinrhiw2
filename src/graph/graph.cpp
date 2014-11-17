

#ifndef graph_cpp
#define graph_cpp

#include "graph.h"
#include <map>
#include <list>
#include <string>


namespace blade
{

  template <typename T>
  graph<T>::graph() throw()
  {
    nodedomain = "nodes";

    // unlimited number of node numbers
    unique_id::create(nodedomain);
  }

  
  template <typename T>
  graph<T>::~graph() throw()
  {
    unique_id::free(nodedomain);
  }
  
  
  // adds node, returns 0 if there's error, otherwise returns id number for node
  template <typename T>
  unsigned int graph<T>::add(graphnode_interface<T>* node)
  {
    if(!node) return 0;
    
    unsigned int id = unique_id::get();
    if(!id) return 0;
    
    nodes[id] = node;
    return id;
  }
  
  
  // adds node, returns 0 if there's error, otherwise returns id number for node
  template <typename T>
  unsigned int graph<T>::add(unsigned int id,
			     const std::list<unsigned int>& adjacencies,
			     bool unidirectional = false) throw()
  {
    if(!node) return 0;
    
    unsigned int id = unique_id::get();
    if(id == 0) return false;

    std::list<graphnode_interface<T>*>& a = 
      node->adjacency();

    
    std::list<graphnode_interface<T>*>::iterator i;

    std::map<unsigned int,
      graphnode_interface<T>*>::iterator j;

    // adds edges
    i = adjacencies.begin();

    for(;i!=adjacencies.end();i++)
    {
      j = nodes.find(*i); // finds pointer of given node

      if(j != nodes.end()) a.insert(*j); // inserts edge
    }


    // adds links also in other direction
    if(unidirectional == false)
    {
      i = adjacencies.begin();
      
      for(;i!=adjacencies.end();i++)
      {
	j = nodes.find(*i); // finds pointer of given node

	if(j != nodes.end())
	  (*j)->adjacency().insert(node); // inserts edge
      }
    }
    
    nodes[id] = node;        
  }
  
  
  template <typename T>
  bool graph<T>::remove(unsigned int id) throw()
  {
    if(!id) return false;
    
    std::map<unsigned int,
      graphnode_interface<T>*>::iterator k;
    
    k = nodes.find(id);

    if(k == nodes.end()) return false;
    
    
    // removes node from others adjacency lists

    std::list<graphnode_interface<T>*>::iterator i;

    std::map<unsigned int,
      graphnode_interface<T>*>::iterator j;
    
    std::list<graphnode_interface<T>*>& a = 
      (k->second)->adjacency();

    i = a.begin();
      
    for(;i!=a.end();i++)
    {
      j = nodes.find(*i); // finds pointer of given node
      
      if(j != nodes.end()){
	
	std::list<graphnode_interface<T>*>::iterator l;
	l = (*j)->adjacency().find(k->second);

	if(l != (*j)->adjancency().end()){
	  // erases pointer to 'to be removed node'
	  (*j)->adjancency().erase(l);
	}
	
      }
    }
    
    nodes.erase(k);
    
    return true;
  }

  
  template <typename T>
  graph_access_interface<T> graph<T>::access() throw()
  {
    return graph_access<T>(nodes);
  }

  
  template <typename T>
  const graph_access_interface<T> graph<T>::access() const throw()
  {
    return graph_access<T>(nodes);
  }
  
  
};

#endif // graph.cpp

