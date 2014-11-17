
#include <iostream>
#include "nandgate.h"

namespace whiteice
{
  namespace digital_logic
  {

    bool nandgate::update() throw()
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
	
	state = !state;
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
  }
}
