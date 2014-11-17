
#include <iostream>
#include "orgate.h"


namespace whiteice
{
  namespace digital_logic
  {

    bool orgate::update() throw()
    {
      try{
	state = false;
	
	const std::vector<component*>& IN = inc();
	std::vector<component*>::const_iterator i;
	i = IN.begin();
	
	while(i != IN.end()){
	  state |= (*i)->get();
	  i++;
	}
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
  }
}
