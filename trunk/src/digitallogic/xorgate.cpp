

#include "xorgate.h"


namespace whiteice
{
  namespace digital_logic
  {
    
    bool xorgate::update() throw()
    {
      try{
	state = false;
	
	const std::vector<component*>& IN = inc();
	std::vector<component*>::const_iterator i;
	i = IN.begin();
	
	if(i != IN.end()){
	  state = (*i)->get();
	  i++;
	}
	
	while(i != IN.end()){
	  state ^= (*i)->get();
	  i++;
	}
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }      
    }
    
  };
};


