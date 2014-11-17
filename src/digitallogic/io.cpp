
#include "io.h"

namespace whiteice
{
  namespace digital_logic
  {
    input::input(){
      input_state = false;
      
      // inputs has one fake input
      // which makes generic code in
      // component class work even
      // in case of inputs
      
      register_input(0);
    }
    
    
    bool input::update() throw()
    {
      state = input_state;
      return true;
    }
    
    
    void input::set(bool value) throw()
    {
      input_state = value;
      signal(); // signals change in inputs
    }
    
    
    bool output::update() throw()
    {
      try{
	const std::vector<component*>& IN = inc();
	std::vector<component*>::const_iterator i;
	i = IN.begin();
	
	if(i != IN.end())
	  state = ((*i)->get());
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
