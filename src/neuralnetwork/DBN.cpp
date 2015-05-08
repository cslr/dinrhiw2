
#include "DBN.h"

namespace whiteice
{

  template <typename T>
  bool RBM<T>::reconstructData(unsigned int iters = 2)
  {
    if(iters == 0) return false;
    if(machines.size() <= 0) return false;
    
    while(iters > 0){
      for(unsigned int i=0;i<machines.size();i++){
	machines[i].reconstructData(1); // from visible to hidden
	if(i+1 < machines.size())
	  machines[i+1].setVisible(machines[i].getHidden()); // hidden -> to the next visible
      }
      
      iters--;
      if(iters <= 0) return true;
      
      // now we have stimulated RBMs all the way to the last hidden layer and now we need to get back
      
      for(int i=machines.size()-1;i>=0;i--){
	machines[i].reconstructDataHidden(1); // from hidden to visible
	if(i-1 >= 0)
	  machines[i-1].setHidden(machines[i].getVisible()); // visible -> to the previous hidden
      }
      
      iters--;
      if(iters <= 0) return true;
    }
    
    return true;
  }
  
  
  template <typename T>
  bool DBN<T>::initializeWeights(){
    for(auto m : machines)
      m.initializeWeights();
  }
  
  // learns stacked RBM layer by layer, each RBM is trained one by one
  template <typename T>
  bool DBN<T>::learnWeights(const std::vector< math::vertex<T> >& samples, const T& dW)
  {
    if(dW < T(0.0)) return false;
    
    std::vector< math::vertex<T> > input = samples;
    std::vector< math::vertex<T> > output;
    
    for(unsigned int i=0;i<machines.size();i++){
      // learns the current layer from input
      
      unsigned int iters = 0;
      while(machines[i].learnWeights(input) >= dW){
	iters++;
	if(iters >= 1000) break; // stop at this step
      }
      
      // maps input into output
      for(auto v : input){
	machines[i].setVisible(v);
	machines[i].reconstructData(1);
	output.push_back(machines[i].getHidden());
      }
      
      input = output; // NOTE we could do shallow copy and use just pointers here.. 
                      // [OPTIMIZE ME!!]
      output.clear();
    }
    
    return true;
  }
  
};
