
#include <iostream>
#include "complement.h"


namespace whiteice
{
  namespace digital_logic
  {
    
    bool complement::update() throw()
    {
      try{
	const std::vector<component*>& IN = inc();
	std::vector<component*>::const_iterator i;
	i = IN.begin();
	
	if(i != IN.end())
	  state = !((*i)->get());
	else
	  state = false;
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }      
    }
    
  };
};
