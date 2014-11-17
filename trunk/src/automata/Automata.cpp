

#ifndef AutomataAutomata_cpp
#define AutomataAutomata_cpp

#include "Automata.h"
#include <string>


namespace automata
{
  template <typename T>
  Automata<T>::Automata() throw()
  {
    start = 0;
    final = 0;
  }
  
 
  // add state to automata
  template <typename T>
  bool Automata<T>::add(const State<T>& state,
			bool start = false, 
			bool final = false) throw()
  {
    try{
      states.push_back(state);
      
      State<T>* pointer = &(states[states.size()-1]);
      
      if(start)
	this->start = pointer;
      if(final)
	this->start = pointer;
      
      return true;
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  bool Automata<T>::check(const std::string<T>& input) throw()
  {
    try{
      if(!start) return false;
      if(!final) return false;
      
      this->input.resize(input.length());
      for(unsigned int i=0;i<input.length();i++)
	this->input[i] = input[i];

      return checkAccept(start, 0);
    }
    catch(std::exception& e){ return false; }
  }
  
  
  template <typename T>
  bool Automata<T>::checkAccept(State<T>* s, unsigned int index)
  {
    bool result = false;
    bool startMark = false;
    bool endMark = false;

    if(input[index] == '['){
      startMark = true;
      index++;
    }
    else if(input[index] == ']'){
      endMark = true;
      index++;
    }

    if(index >= input.size())
    {
      if(s == final) result = true;
      else return false;
    }
    

    
    for(unsigned int i=0;i<s->getNumberOfEdges() && result == false;i++)
    {
      Edge<T>* e = &(s->getEdge(i));
      
      if(e->accept(input[index]))
      {
	if(e->constraintlessEdge())
	{
	  if(checkAccept(&(e->getTransitionState()), index))
	  {
	    result = true;
	    break;
	  }
	}
	else
	{
	  if(checkAccept(r, index+1))
	  {
	    result = true;
	    break;
	  }
	}
      }
      
    }
    
    
    if(result)
    {
      if(startMark)    start_marks.push_back(index-1);
      else if(endMark) end_marks.push_back(index-1);
      
      return true;
    }
    else return false;
    
  }
  
  
  template <typename T>
  bool Automata<T>::numberOfMarkedStrings() throw()
  {
    return start_marks.size();
  }

  
  template <typename T>
  const std::string<T> Automata<T>::getMarkedString(unsigned int index) const
    throw(std::out_of_range)
  {
    int start_pos = start_marks.at(index);
    int end_pos = end_marks.at(index);
    
    return input.substring(start_pos, end_pos - start_pos);
  }
  
};


#endif






