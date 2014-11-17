
#include <iostream>
#include "andgate.h"

namespace whiteice
{
  namespace digital_logic
  {

    bool andgate::update() throw()
    {
      try{
	state = true;
	
	const std::vector<component*>& IN = inc();
	std::vector<component*>::const_iterator i;
	i = IN.begin();
	
	while(i != IN.end()){
	  state &= (*i)->get();
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
