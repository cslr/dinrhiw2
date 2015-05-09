
#include "DBN.h"

namespace whiteice
{
  
  template <typename T>
  DBN<T>::DBN()
  {
    machines.resize(1);
    machines[0].resize(1,1); // 1 => 1 pseudonetwork
  }
  
  
  // constructs stacked RBM netwrok with the given architecture
  template <typename T>
  DBN<T>::DBN(std::vector<unsigned int>& arch)
  {
    if(arch.size() <= 1)
      throw std::invalid_argument("Invalid network architechture.");
    
    for(auto a : arch)
      if(a <= 0)
	throw std::invalid_argument("Invalid network architechture.");

    machines.resize(arch.size() - 1);
    
    for(unsigned int i=0;i<(arch.size()-1);i++){
      machines[i].resize(arch[i],arch[i+1]);
    }
  }

  template <typename T>
  DBN<T>::DBN(const DBN<T>& dbn)
  {
    machines = dbn.machines;
  }
  

  template <typename T>
  bool DBN<T>::resize(std::vector<unsigned int>& arch)
  {
    if(arch.size() <= 1)
      return false;
    
    for(auto a : arch)
      if(a <= 0) return false;
    
    machines.resize(arch.size() - 1);
    
    for(unsigned int i=0;i<(arch.size()-1);i++){
      if(machines[i].resize(arch[i],arch[i+1]) == false)
	return false;
    }
    
    return true;
  }
  
  
  ////////////////////////////////////////////////////////////
  
  // visible neurons/layer of the first RBM
  template <typename T>
  math::vertex<T> DBN<T>::getVisible() const
  {
    math::vertex<T> v;
    if(machines.size() > 0)
      v = machines[0].getVisible();
    
    return v;
  }
  
  template <typename T>
  bool DBN<T>::setVisible(const math::vertex<T>& v)
  {
    if(machines.size() > 0){
      return machines[0].setVisible(v);
    }
    else return false;
  }
  
  
  // hidden neurons/layer of the last RBM
  template <typename T>
  math::vertex<T> DBN<T>::getHidden() const
  {
    math::vertex<T> h;
    if(machines.size() > 0)
      h = machines[machines.size()-1].getHidden();
    
    return h;
  }
  
  template <typename T>
  bool DBN<T>::setHidden(const math::vertex<T>& h)
  {
    if(machines.size() > 0){
      return machines[machines.size()-1].setHidden(h);
    }
    else return false;
  }
  
  
  template <typename T>
  bool DBN<T>::reconstructData(unsigned int iters)
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
    
    return true;
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


  template class DBN< float >;
  template class DBN< double >;  
  template class DBN< math::blas_real<float> >;
  template class DBN< math::blas_real<double> >;
  
};
