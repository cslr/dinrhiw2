

#ifndef AutomataState_h
#define AutomataState_h

#include <vector>
#include "Edge.h"
#include "State.h"

namespace automata
{
  template <typename T>
  class State
  {
  public:
    State() throw();
    
    bool add(const Edge<T>& e) throw();
    bool remove(unsigned int index) throw();
    
    const Edge<T>& getEdge(unsigned int index) const throw(std::out_of_range);
    unsigned int getNumberOfEdges() const throw();
    
  private:
    // output edges
    std::vector< Edge<T> > edges;
    
  };
  
};

#include "State.cpp"

#endif

