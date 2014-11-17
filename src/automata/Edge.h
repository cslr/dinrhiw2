

#ifndef AutomataEdge_h
#define AutomataEdge_h

#include <stdexcept>
#include <exception>
#include <string>

#include "Automata.h"
#include "State.h"


namespace automata
{
  template <typename T>
    class Edge
    {
      friend class Automata<T>;

      //private:
    public:
      
      // constraintless edge with given target state
      Edge(const State<T>& target) throw();
      
      // constrained edge with given state
      Edge(const T& character, const State<T>& target) throw();

      // constrained list, must match some character (matchAny == true) or
      // must not match any (matchAny == false)
      Edge(const std::string<T>& exceptlist,
	   bool matchAny,
	   const State<T>& target) throw();

      // creates copy of edge
      Edge(const Edge<T>& e) throw();
      
    public:
      bool constraintlessEdge() const throw();
      bool accept(const T& character) const throw();
      
      const T& getConstraint() const throw(std::logic_error);
      const std::string<T>& getConstraintList() const throw(std::logic_error);

      const State<T>& getTransitionState() const throw();

      Edge<T>& operator=(const Edge<T>& s) throw();
      bool operator==(const Edge<T>& s) const throw();
      bool operator!=(const Edge<T>& s) const throw();
      
    private:
      State<T>* state; // target state
      T constraint;
      bool hasConstraint; // has single constraint
      
      std::string<T> constraint_list;
      bool matchAny; // match any in a list (true) or don't match any in the list
      bool hasConstraintList; // has constraint list
    };
  
  
};


#include "Edge.cpp"

#endif


