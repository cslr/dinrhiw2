
#include "DBN.h"

namespace whiteice
{
  
  template <typename T>
  DBN<T>::DBN()
  {
    input.resize(1,1); // 1 => 1 pseudonetwork
    input.initializeWeights();
  }
  
  
  // constructs stacked RBM netwrok with the given architecture
  template <typename T>
  DBN<T>::DBN(std::vector<unsigned int>& arch)
  {
    if(arch.size() <= 1)
      throw std::invalid_argument("Invalid network architechture.");
    
    for(auto a : arch)
      if(a <= 0) throw std::invalid_argument("Invalid network architechture.");
       
    layers.resize(arch.size() - 2);

    input.resize(arch[0], arch[1]);
    input.initializeWeights();
    
    for(unsigned int i=0;i<(arch.size()-2);i++){
      layers[i].resize(arch[i+1],arch[i+2]);
      layers[i].initializeWeights();
    }
  }

  template <typename T>
  DBN<T>::DBN(const DBN<T>& dbn)
  {
    input = dbn.input;
    layers = dbn.layers;
  }


  template <typename T>
  unsigned int DBN<T>::getNumberOfLayers() const
  {
    return (1 + layers.size());
  }
  

  template <typename T>
  bool DBN<T>::resize(std::vector<unsigned int>& arch)
  {
    if(arch.size() <= 1)
      return false;
    
    for(auto a : arch)
      if(a <= 0) return false;

    layers.resize(arch.size() - 2);

    input.resize(arch[0], arch[1]);
    input.initializeWeights();
    
    for(unsigned int i=0;i<(arch.size()-2);i++){
      layers[i].resize(arch[i+1],arch[i+2]);
      layers[i].initializeWeights();
    }
    
    return true;
  }
  
  
  ////////////////////////////////////////////////////////////
  
  // visible neurons/layer of the first RBM
  template <typename T>
  math::vertex<T> DBN<T>::getVisible() const
  {
    math::vertex<T> v;

    input.getVisible(v);
    
    return v;
  }
  
  template <typename T>
  bool DBN<T>::setVisible(const math::vertex<T>& v)
  {
    return input.setVisible(v);
  }
  
  
  // hidden neurons/layer of the last RBM
  template <typename T>
  math::vertex<T> DBN<T>::getHidden() const
  {
    math::vertex<T> h;
    
    if(layers.size() > 0){
      layers[layers.size()-1].getHidden(h);
      return h;
    }
    else{
      input.getHidden(h);
      return h;
    }
  }
  
  template <typename T>
  bool DBN<T>::setHidden(const math::vertex<T>& h)
  {
    if(layers.size() > 0){
      return layers[layers.size()-1].setHidden(h);
    }
    else{
      return input.setHidden(h);
    }
  }
  
  
  template <typename T>
  bool DBN<T>::reconstructData(unsigned int iters)
  {
    if(iters == 0) return false;
    
    while(iters > 0){
      input.reconstructData(1);

      for(unsigned int i=0;i<layers.size();i++){
	math::vertex<T> h;

	if(i == 0){
	  input.getHidden(h);
	  layers[0].setVisible(h);
	}
	else{
	  layers[i-1].getHidden(h);
	  layers[i].setVisible(h);
	}

	layers[i].reconstructData(1); // from visible to hidden
      }

      iters--;

      if(iters <= 0) return true;
      
      // now we have stimulated RBMs all the way to the last hidden layer and now we need to get back
      
      for(int i=(int)(layers.size()-1);i>=0;i--){
	math::vertex<T> v;
	layers[i].reconstructDataHidden(1); // from hidden to visible
	if(i >= 1){
	  layers[i].getVisible(v);
	  layers[i-1].setHidden(v); // visible -> to the previous hidden
	}
	else{
	  layers[0].getVisible(v);
	  input.setHidden(v);
	}
      }

      input.reconstructDataHidden(1);
      
      iters--;
    }
    
    return true;
  }


  template <typename T>
  bool DBN<T>::reconstructData(std::vector< math::vertex<T> >& samples, unsigned int iters)
  {
    for(auto& s : samples){
      if(!setVisible(s)) return false;
      if(!reconstructData(iters)) return false;
      s = getVisible();
    }

    return true;
  }
  
  
  template <typename T>
  bool DBN<T>::initializeWeights()
  {
    input.initializeWeights();
    
    for(auto l :layers)
      l.initializeWeights();
    
    return true;
  }
  
  
  // learns stacked RBM layer by layer, each RBM is trained one by one
  template <typename T>
  bool DBN<T>::learnWeights(const std::vector< math::vertex<T> >& samples,
			    const T& dW, bool verbose)
  {
    if(dW < T(0.0)) return false;

    const unsigned int ITERLIMIT = 100;
    
    std::vector< math::vertex<T> > in = samples;
    std::vector< math::vertex<T> > out;

    unsigned int iters = 0;
    while(input.learnWeights(in, 1, verbose) >= dW){
      // learns also variance
      iters++;

      if(verbose)
	std::cout << "GB-RBM INPUT LAYER ITER "
		  << iters << std::endl;
      
      if(iters >= ITERLIMIT)
	break; // stop at this step
    }

    // maps input to output
    out.clear();
    
    for(auto v : in){
      input.setVisible(v);
      input.reconstructData(1);

      math::vertex<T> h;
      input.getHidden(h);

      out.push_back(h);
    }

    in = out;
    
    
    for(unsigned int i=0;i<layers.size();i++){
      // learns the current layer from input
      
      unsigned int iters = 0;
      while(layers[i].learnWeights(in) >= dW){
	iters++;

	if(verbose)
	  std::cout << "BB-RBM LAYER " << i << " ITER "
		    << iters << std::endl;
	
	if(iters >= ITERLIMIT)
	  break; // stop at this step
      }
      
      // maps input into output
      out.clear();
      
      for(auto v : in){
	layers[i].setVisible(v);
	layers[i].reconstructData(1);

	math::vertex<T> h;
	layers[i].getHidden(h);
	out.push_back(h);
      }
      
      in = out;
    }
    
    return true;
  }


  template class DBN< float >;
  template class DBN< double >;  
  template class DBN< math::blas_real<float> >;
  template class DBN< math::blas_real<double> >;
  
};
