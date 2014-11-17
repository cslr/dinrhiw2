

#ifndef AutomataAutomata_h
#define AutomataAutomata_h

#include <string>
#include <vector>
#include <stdexcept>
#include <exception>



namespace automata
{

  template <typename T>
    class Edge;

  template <typename T>
    class State;
  
  
  
  template <typename T>
    class Automata
    {
    public:
      Automata() throw();
      // Automata(const std::string<T> regexp) throw();
      
      // add state to automata
      bool add(const State<T>& state,
	       bool start = false, 
	       bool final = false) throw();

      bool check(const std::string<T>& input) throw();
            
      bool numberOfMarkedStrings() throw();
      const std::string<T> getMarkedString(unsigned int i) const throw(std::out_of_range);
      
    private:
      
      bool checkAccept(State<T>* s, unsigned int index);

      
      // starting and final states
      State<T>* start;
      State<T>* final;
            
      std::vector< State<T> > states; // list of all states
      
      std::vector<T> input;
      std::vector<unsigned int> start_marks;
      std::vector<unsigned int> end_marks;
    };


#include "Automata.cpp"
    
};


#include "State.h"
#include "Edge.h"


#endif



