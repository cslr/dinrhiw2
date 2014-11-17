
#include <iostream>
#include "components.h"

int main(int argc, char** argv)
{
  using namespace whiteice::digital_logic;
  
  // constructs simple circuit:
  // (X,Y) -> (NOT(X AND Y), X OR Y)
  
  std::vector<complement*> complements;
  std::vector<input*> inputs;
  std::vector<output*> outputs;
  std::vector<andgate*> ands;
  std::vector<orgate*> ors;
  
  inputs.push_back(new input());
  inputs.push_back(new input());
  
  complements.push_back(new complement());
  
  ands.push_back(new andgate());
  ors.push_back(new orgate());
  
  outputs.push_back(new output());
  outputs.push_back(new output());
  
  
  inputs[0]->register_output(ands[0]);
  inputs[1]->register_output(ands[0]);
  ands[0]->register_input(inputs[0]);
  ands[0]->register_input(inputs[1]);
  
  complements[0]->register_input(ands[0]);
  ands[0]->register_output(complements[0]);
  
  outputs[0]->register_input(complements[0]);
  complements[0]->register_output(outputs[0]);
  
  
  inputs[0]->register_output(ors[0]);
  inputs[1]->register_output(ors[0]);
  ors[0]->register_input(inputs[0]);
  ors[0]->register_input(inputs[1]);
  
  outputs[1]->register_input(ors[0]);
  ors[0]->register_output(outputs[1]);
  
  // signals all components so that circuit is in coherent state
  for(unsigned int i=0;i<inputs.size();i++)  inputs[i]->signal();
  for(unsigned int i=0;i<outputs.size();i++) outputs[i]->signal();
  for(unsigned int i=0;i<ands.size();i++)    ands[i]->signal();
  for(unsigned int i=0;i<ors.size();i++)     ors[i]->signal();
  for(unsigned int i=0;i<complements.size();i++) complements[i]->signal();
    
  unsigned int c1, c2;
  
  
  while(1){
    
    c1 = outputs[0]->activation();
    c2 = outputs[1]->activation();
    
    inputs[0]->set(false);
    inputs[1]->set(false);
    
    // waits till changes have propagated to outputs
    while(c1 == outputs[0]->activation() ||
	  c2 == outputs[1]->activation());
    
    std::cout << "("      << inputs[0]->get()
	      << " , "    << inputs[1]->get()
	      << ") => (" << outputs[0]->get()
	      << " , "    << outputs[1]->get()
	      << ")" << std::endl;
    
    
    c1 = outputs[0]->activation();
    c2 = outputs[1]->activation();
    
    inputs[0]->set(true);
    inputs[1]->set(false);
    
    while(c1 == outputs[0]->activation() ||
	  c2 == outputs[1]->activation());
    
    std::cout << "("      << inputs[0]->get()
	      << " , "    << inputs[1]->get()
	      << ") => (" << outputs[0]->get()
	      << " , "    << outputs[1]->get()
	      << ")" << std::endl;
    
    c1 = outputs[0]->activation();
    c2 = outputs[1]->activation();
    
    inputs[0]->set(false);
    inputs[1]->set(true);
    
    while(c1 == outputs[0]->activation() ||
	  c2 == outputs[1]->activation());
    
    std::cout << "("      << inputs[0]->get()
	      << " , "    << inputs[1]->get()
	      << ") => (" << outputs[0]->get()
	      << " , "    << outputs[1]->get()
	      << ")" << std::endl;
    
    c1 = outputs[0]->activation();
    c2 = outputs[1]->activation();
    
    inputs[0]->set(true);
    inputs[1]->set(true);
    
    while(c1 == outputs[0]->activation() ||
	  c2 == outputs[1]->activation());  
    
    std::cout << "("      << inputs[0]->get()
	      << " , "    << inputs[1]->get()
	      << ") => (" << outputs[0]->get()
	      << " , "    << outputs[1]->get()
	      << ")" << std::endl;
    
  }
  
  return 0;
}


