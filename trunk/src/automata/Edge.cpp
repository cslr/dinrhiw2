
#ifndef AutomataEdge_cpp
#define AutomataEdge_cpp

#include <stdexcept>
#include <exception>

#include "Automata.h"
#include "State.h"


namespace automata
{
  
  template <typename T>
  Edge<T>::Edge(const State<T>& target) throw() : 
    state((State<T>*)&target)
  {
    hasConstraint = false;
    hasConstraintList = false;
    matchAny = false;
    
    state = 0;    
  }

  
  template <typename T>
  Edge<T>::Edge(const T& character, const State<T>& target) throw() : 
    state((State<T>*)&target)
  {
    hasConstraint = true;
    hasConstraintList = false;
    constraint = character;
  }


  
  template <typename T>
  Edge<T>::Edge(const std::string<T>& exceptlist,
		bool matchAny,
		const State<T>& target) throw() : 
    state((State<T>*)&target)
  {
    hasConstraint = false;
    hasConstraintList = true;
    this->constraint_list = exceptlist;
    this->matchAny = matchAny;
  }
  

  template <typename T>  
  Edge<T>::Edge(const Edge<T>& e) throw() : 
    state((State<T>*)&(e.state))
  {
    this->hasConstraint = e.hasConstraint;
    this->hasConstraintList = e.hasConstraintList;
    
    this->constraint = e.constraint;
    this->state = e.state;
    this->matchAny = e.matchAny;
    this->constraint_list = e.constraint_list;
  }


  template <typename T>
  bool Edge<T>::constraintlessEdge() const throw()
  {
    return (!hasConstraint && !hasConstraintList);
  }
  
  
  
  template <typename T>
  bool Edge<T>::accept(const T& character) const throw()
  {
    if(!hasConstraint && !hasConstraintList) return true;
    
    if(hasConstraint){
      if(character == constraint) return true;
      else return false;
    }

    if(hasConstraintList)
    {      
      for(unsigned int i=0;i<constraint_list.size();i++){
	if(constraint_list[i] == character)
	  return matchAny;
      }

      return !matchAny;
    }
    
    return false;
  }
      
  
  
  template <typename T>
  const T& Edge<T>::getConstraint() const throw(std::logic_error)
  {
    if(!hasConstraint)
      throw std::logic_error("automata::Edge has no constraint");

    return constraint;
  }


  template <typename T>
  const std::string<T>& Edge<T>::getConstraintList() const throw(std::logic_error)
  {
    if(!hasConstraintList)
      throw std::logic_error("automata::Edge has no constraint");

    return constraint_list;
  }

  
  template <typename T>
  const State<T>& Edge<T>::getTransitionState() const throw()
  {
    return *state;
  }


  template <typename T>
  Edge<T>& Edge<T>::operator=(const Edge<T>& s) throw()
  {
    this->constraint = s.constraint;
    this->hasConstraint = s.hasConstraint;
    this->state = (State<T>*)(s.state);

    this->constraint_list = s.constraint_list;
    this->matchAny = s.matchAny;
    this->hasConstraintList = s.hasConstraintList;
    
    return *this;
  }

  
  template <typename T>
  bool Edge<T>::operator==(const Edge<T>& s) const throw()
  {    
    if(this->state != (State<T>*)(s.state)) return false;
    if(this->hasConstraint != s.hasConstraint) return false;
    if(this->hasConstraintList != s.hasConstraintList) return false;

    if(this->hasConstraint)
      if(this->constraint != s.constraint) return false;

    if(this->hasConstraintList){
      if(this->matchAny != s.matchAny) return false;
      if(this->constraint_list != s.constraint_list) return false;
    }
    
    
    return true;
  }

  
  template <typename T>
  bool Edge<T>::operator!=(const Edge<T>& s) const throw()
  {    
    if(this->state == (State<T>*)(s.state)) return false;
    if(this->hasConstraint == s.hasConstraint) return false;
    if(this->hasConstraintList == s.hasConstraintList) return false;

    if(this->hasConstraint)
      if(this->constraint == s.constraint) return false;
    
    if(this->hasConstraintList){
      if(this->matchAny == s.matchAny) return false;
      if(this->constraint_list == s.constraint_list) return false;
    }      
    
    return true;    
  }
  
  
  
};


#endif









