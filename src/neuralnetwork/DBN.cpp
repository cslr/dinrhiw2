
#include "DBN.h"
#include <list>


namespace whiteice
{
  
  template <typename T>
  DBN<T>::DBN(bool binary)
  {
    gb_input.resize(1,1); // 1 => 1 pseudonetwork
    bb_input.resize(1,1);
    binaryInput = binary;
    
    gb_input.initializeWeights();
    bb_input.initializeWeights();
  }
  
  
  // constructs stacked RBM netwrok with the given architecture
  template <typename T>
  DBN<T>::DBN(std::vector<unsigned int>& arch, bool binary)
  {
    if(arch.size() <= 1)
      throw std::invalid_argument("Invalid network architechture.");
    
    for(auto a : arch)
      if(a <= 0) throw std::invalid_argument("Invalid network architechture.");
       
    layers.resize(arch.size() - 2);

    binaryInput = binary;
    
    if(!binary){
      gb_input.resize(arch[0], arch[1]);
      gb_input.initializeWeights();
    }
    else{
      bb_input.resize(arch[0], arch[1]);
      bb_input.initializeWeights();
    }

    for(unsigned int i=0;i<(arch.size()-2);i++){
      layers[i].resize(arch[i+1],arch[i+2]);
      layers[i].initializeWeights();
    }
  }

  template <typename T>
  DBN<T>::DBN(const DBN<T>& dbn)
  {
    gb_input = dbn.gb_input;
    bb_input = dbn.bb_input;
    binaryInput = dbn.binaryInput;
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

    gb_input.resize(arch[0], arch[1]);
    gb_input.initializeWeights();
    bb_input.resize(arch[0], arch[1]);
    bb_input.initializeWeights();
    
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

    if(!binaryInput)
      gb_input.getVisible(v);
    else
      bb_input.getVisible(v);
    
    return v;
  }
  
  template <typename T>
  bool DBN<T>::setVisible(const math::vertex<T>& v)
  {
    if(!binaryInput)
      return gb_input.setVisible(v);
    else
      return bb_input.setVisible(v);
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
      if(!binaryInput)
	gb_input.getHidden(h);
      else
	bb_input.getHidden(h);
      
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
      if(!binaryInput)
	return gb_input.setHidden(h);
      else
	return bb_input.setHidden(h);
    }
  }
  
  
  template <typename T>
  bool DBN<T>::reconstructData(unsigned int iters)
  {
    if(iters == 0) return false;
    
    while(iters > 0){
      if(!binaryInput)
	gb_input.reconstructData(1);
      else
	bb_input.reconstructData(1);

      for(unsigned int i=0;i<layers.size();i++){
	math::vertex<T> h;

	if(i == 0){
	  if(!binaryInput)
	    gb_input.getHidden(h);
	  else
	    bb_input.getHidden(h);
	  
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

	  if(!binaryInput)
	    gb_input.setHidden(v);
	  else
	    bb_input.setHidden(v);
	}
      }

      if(!binaryInput)
	gb_input.reconstructDataHidden(1);
      else
	bb_input.reconstructDataHidden(1);
      
      
      iters--;
    }
    
    return true;
  }


  template <typename T>
  bool DBN<T>::reconstructData(std::vector< math::vertex<T> >& samples)
  {
    for(auto& s : samples){
      if(!setVisible(s)) return false;
      if(!reconstructData(2)) return false;
      s = getVisible();
    }

    return true;
  }
  
  
  template <typename T>
  bool DBN<T>::initializeWeights()
  {
    gb_input.initializeWeights();
    bb_input.initializeWeights();
    
    for(auto l :layers)
      l.initializeWeights();
    
    return true;
  }
  
  
  // learns stacked RBM layer by layer, each RBM is trained one by one
  template <typename T>
  bool DBN<T>::learnWeights(const std::vector< math::vertex<T> >& samples,
			    const T& errorLevel, bool verbose)
  {
    if(errorLevel < T(0.0)) return false;

    // increase after the whole learning process works..
    const unsigned int ITERLIMIT = 100;
    const unsigned int EPOCH_STEPS = 2;
    
    std::vector< math::vertex<T> > in = samples;
    std::vector< math::vertex<T> > out;
    T error = T(INFINITY);
    std::list<T> errors;

    if(!binaryInput){ // GBRBM
      
      unsigned int iters = 0;
      while((error = gb_input.learnWeights(in, EPOCH_STEPS, verbose)) >= errorLevel
	    && iters < ITERLIMIT)
      {
	// learns also variance
	iters += EPOCH_STEPS;
	errors.push_back(error);
	
	if(verbose)
	  std::cout << "GBRBM LAYER OPTIMIZATION "
		    << iters << "/" << ITERLIMIT << ": "
		    << error << "/" << errorLevel 
		    << std::endl;
	
	while(errors.size() > 10)
	  errors.pop_front();
	
	// calculates mean and variance and stops if variance is
	// within 1% of mean (convergence) during latest
	// EPOCH_STEPS*10 iterations
	
	if(errors.size() >= 10){
	  T m = T(0.0), v = T(0.0);
	  
	  for(auto& e : errors){
	    m += e;
	    v += e*e;
	  }
	  
	  m /= errors.size();
	  v /= errors.size();
	  v -= m*m;
	  
	  v = sqrt(v);
	  
	  if(v/m <= 1.01) 
	    break; // stdev is less than 1% of mean
	}
      }

      // maps input to output
      out.clear();
      
      for(auto v : in){
	gb_input.setVisible(v);
	gb_input.reconstructData(1);
	
	math::vertex<T> h;
	gb_input.getHidden(h);
	
	out.push_back(h);
      }
      
      in = out;
      
    }
    else{ // BBRBM
      
      unsigned int iters = 0;
      while((error = bb_input.learnWeights(in, EPOCH_STEPS, verbose)) >= errorLevel
	    && iters < ITERLIMIT)
      {
	// learns also variance
	iters += EPOCH_STEPS;
	errors.push_back(error);
	
	if(verbose)
	  std::cout << "BBRBM LAYER OPTIMIZATION "
		    << iters << "/" << ITERLIMIT << ": "
		    << error << "/" << errorLevel 
		    << std::endl;
	
	while(errors.size() > 10)
	  errors.pop_front();
	
	// calculates mean and variance and stops if variance is
	// within 1% of mean (convergence) during latest
	// EPOCH_STEPS*10 iterations
	
	if(errors.size() >= 10){
	  T m = T(0.0), v = T(0.0);
	  
	  for(auto& e : errors){
	    m += e;
	    v += e*e;
	  }
	  
	  m /= errors.size();
	  v /= errors.size();
	  v -= m*m;
	  
	  v = sqrt(v);
	  
	  if(v/m <= 1.01) 
	    break; // stdev is less than 1% of mean
	}
      }

      // maps input to output
      out.clear();
      
      for(auto v : in){
	bb_input.setVisible(v);
	bb_input.reconstructData(1);
	
	math::vertex<T> h;
	bb_input.getHidden(h);
	
	out.push_back(h);
      }
      
      in = out;
      
    }

    
    

    for(unsigned int i=0;i<layers.size();i++){
      // learns the current layer from input

      unsigned int iters = 0;
      while((error = layers[i].learnWeights(in, EPOCH_STEPS, verbose)) >= errorLevel &&
	    iters < ITERLIMIT)
      {
	iters += EPOCH_STEPS;

	errors.push_back(error);
	
	if(verbose)
	  std::cout << "BBRBM LAYER OPTIMIZATION "
		    << iters << "/" << ITERLIMIT << ": "
		    << error << "/" << errorLevel 
		    << std::endl;
	
	while(errors.size() > 10)
	  errors.pop_front();
	
	// calculates mean and variance and stops if variance is
	// within 1% of mean (convergence) during latest
	// EPOCH_STEPS*10 iterations
	
	if(errors.size() >= 10){
	  T m = T(0.0), v = T(0.0);
	  
	  for(auto& e : errors){
	    m += e;
	    v += e*e;
	  }
	  
	  m /= errors.size();
	  v /= errors.size();
	  v -= m*m;
	  
	  v = sqrt(v);
	  
	  if(v/m <= 1.01) 
	    break; // stdev is less than 1% of mean
	}
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

  
  /* converts DBN to supervised nnetwork by using training samples
   * (cluster 0 = input) and (cluster 1 = output) and
   * by adding linear outputlayer which is optimized locally using linear optimization
   * returned nnetwork contains layer by layer optimized values which
   * can be further optimized across all layers using nnetwork optimizers
   */
  template <typename T>
  bool DBN<T>::convertToNNetwork(const whiteice::dataset<T>& data,
				 whiteice::nnetwork<T>*& net)
  {
    if(data.getNumberOfClusters() < 2)
      return false;

    // cluster 0 input values must match input layer dimensions
    if(!binaryInput){
      if(data.access(0,0).size() != gb_input.getVisibleNodes()) 
	return false;
    }
    else{
      if(data.access(0,0).size() != bb_input.getVisibleNodes()) 
	return false;      
    }

    // cluster 0 (input) and cluster 1 (output) have different sizes
    if(data.size(0) != data.size(1))
      return false;

    // output layer dimensions
    const unsigned int outputDimension = data.access(1,0).size();

    // 1. process input data and calculates the deepest hidden values h
    // 2. optimizes hidden values h to map output values:
    //    min ||Ah + b - y||^2 (final linear layer)
    // 3. creates lreg_nnetwork<T> and sets parameters (weights)
    //    appropriately

    // calculates hidden values
    std::vector< math::vertex<T> > hidden;
    
    for(unsigned int i=0;i<data.size(0);i++){
      this->setVisible(data.access(0,i));
      this->reconstructData(1);
      hidden.push_back(this->getHidden());
    }

    /////////////////////////////////////////////////////////////////
    // final layers linear optimizer (hidden(i) -> data.access(1, i))
    // 
    // solves min E{ 0.5*||Ah + b - y||^2 }
    // 
    // A = Syh * Shh^-1
    // b = m_y - A*m_h
    
    math::matrix<T> A;
    math::vertex<T> b;

    const unsigned int hDimension = this->getHidden().size();

    math::matrix<T> Shh(hDimension, hDimension);
    math::matrix<T> Syh(outputDimension, hDimension);
    math::vertex<T> mh(hDimension);
    math::vertex<T> my(outputDimension);

    Shh.zero();
    Syh.zero();
    mh.zero();
    my.zero();

    for(unsigned int i=0;i<hidden.size();i++){
      const auto& h = hidden[i];
      const auto& y = data.access(1, i);

      mh += h/T(hidden.size());
      my += y/T(hidden.size());

      Shh += h.outerproduct(h)/T(hidden.size());
      Syh += y.outerproduct(h)/T(hidden.size());
    }

    Shh -= mh.outerproduct(mh);
    Syh -= my.outerproduct(mh);

    // calculate inverse of Shh + regularizes it by adding terms
    // to diagonal if inverse fails
    // (TODO calculate pseudoinverse instead)
    
    T mindiagonal = Shh(0,0);

    for(unsigned int i=0;i<outputDimension;i++)
      if(Shh(i,i) < mindiagonal)
	mindiagonal = Shh(i, i);

    double k = 0.01;

    while(Shh.inv() == false){
      for(unsigned int i=0;i<outputDimension;i++){
	T p = T(pow(2.0, k));
	Shh(i,i) += p*mindiagonal;
      }

      k = 2.0*k;
    }

    A = Syh*Shh;
    b = my - A*mh;
    
    //////////////////////////////////////////////////////////
    // creates feedforward neural network

    std::vector<unsigned int> arch; // architecture

    if(!binaryInput)
      arch.push_back(gb_input.getVisibleNodes());
    else
      arch.push_back(bb_input.getVisibleNodes());

    if(layers.size() > 0){
      for(unsigned int i=0;i<layers.size();i++)
	arch.push_back(layers[i].getVisibleNodes());

      arch.push_back(layers[layers.size()-1].getHiddenNodes());
    }
    else{
      if(!binaryInput)
	arch.push_back(gb_input.getHiddenNodes());
      else
	arch.push_back(bb_input.getHiddenNodes());
    }

    arch.push_back(outputDimension);

    net = new whiteice::nnetwork<T>(arch);

    // copies DBN parameters as nnetwork parameters..
    if(!binaryInput){
      if(net->setWeights(gb_input.getWeights().transpose(), 0) == false){ delete net; net = nullptr; return false; }
      if(net->setBias(gb_input.getBValue(), 0) == false){ delete net; net = nullptr; return false; }
    }
    else{
      if(net->setWeights(bb_input.getWeights(), 0) == false){ delete net; net = nullptr; return false; }
      if(net->setBias(bb_input.getBValue(), 0) == false){ delete net; net = nullptr; return false; }
    }

    for(unsigned int l=0;l<layers.size();l++){
      if(net->setWeights(layers[l].getWeights(), l+1) == false){ delete net; net = nullptr; return false; }
      if(net->setBias(layers[l].getBValue(), l+1) == false){ delete net; net = nullptr; return false; }
    }

    if(net->setWeights(A, layers.size()+1) == false){ delete net; net = nullptr; return false; }
    if(net->setBias(b, layers.size()+1) == false){ delete net; net = nullptr; return false; }

    net->setNonlinearity(whiteice::nnetwork<T>::sigmoidNonLinearity);
    net->setStochastic(true); // stochastic DBN activation
    
    return true;
  }


  // converts trained DBN to autoencoder which can be trained using LBFGS
  template <typename T>
  bool DBN<T>::convertToAutoEncoder(whiteice::nnetwork<T>*& net) const
  {
    // we invert DBN in the middle so we have DBN><invDBN


    if(!binaryInput){
      //////////////////////////////////////////////////////////
      // creates feedforward neural network
      
      std::vector<unsigned int> arch; // architecture
      arch.push_back(gb_input.getVisibleNodes());
      
      if(layers.size() > 0){
	for(unsigned int i=0;i<layers.size();i++)
	  arch.push_back(layers[i].getVisibleNodes());
	
	arch.push_back(layers[layers.size()-1].getHiddenNodes());
      }
      else{
	arch.push_back(gb_input.getHiddenNodes());
      }
      
      // we have arch all the way to hidden layer, now we invert it back
      if(layers.size() > 0 ){
	for(int i=layers.size()-1;i>=0;i--)
	  arch.push_back(layers[i].getVisibleNodes());
      }
      
      arch.push_back(gb_input.getVisibleNodes());
      
      net = new whiteice::nnetwork<T>(arch);
      
      for(unsigned int l=0;l<net->getLayers();l++){
	whiteice::math::matrix< T > W;
	whiteice::math::vertex< T > bias;
	
	net->getBias(bias, l);
	net->getWeights(W, l);      
      }
      
      try {
	// copies DBN parameters as nnetwork parameters.. (forward step) [encoder]
	if(net->setWeights(gb_input.getWeights().transpose(), 0) == false) throw "error setting input layer W";
	
	if(net->setBias(gb_input.getBValue(), 0) == false) throw "error setting input layer b";
	
	for(unsigned int l=0;l<layers.size();l++){
	  if(net->setWeights(layers[l].getWeights(), l+1) == false) throw "error setting encoder layer W";
	  if(net->setBias(layers[l].getBValue(), l+1) == false) throw "error setting encoder layer b";
	}
	
	// copies DBN parameters as nnetwork parameters.. (forward step) [decoder]
	int ll = layers.size();
	for(int l=layers.size()-1;l>=0;l--,ll++){
	  if(net->setWeights(layers[l].getWeights().transpose(), ll+1) == false) throw "error setting decoder layer W^t";
	  if(net->setBias(layers[l].getAValue(), ll+1) == false) throw "error setting decoder layer a";
	}
	
	if(net->setWeights(gb_input.getWeights(), ll+1) == false) throw "error setting decoder output layer W^t ";
	if(net->setBias(gb_input.getAValue(), ll+1) == false) throw "error setting decoder output layer a";
	
	net->setStochastic(true);
	net->setNonlinearity(whiteice::nnetwork<T>::sigmoidNonLinearity);
      }
      catch(const char* msg){
	printf("ERROR: %s\n", msg);
	return false;
      }
      
      return true;

    }
    else{ // binary input

      //////////////////////////////////////////////////////////
      // creates feedforward neural network
      
      std::vector<unsigned int> arch; // architecture
      arch.push_back(bb_input.getVisibleNodes());
      
      if(layers.size() > 0){
	for(unsigned int i=0;i<layers.size();i++)
	  arch.push_back(layers[i].getVisibleNodes());
	
	arch.push_back(layers[layers.size()-1].getHiddenNodes());
      }
      else{
	arch.push_back(bb_input.getHiddenNodes());
      }
      
      // we have arch all the way to hidden layer, now we invert it back
      if(layers.size() > 0 ){
	for(int i=layers.size()-1;i>=0;i--)
	  arch.push_back(layers[i].getVisibleNodes());
      }
      
      arch.push_back(bb_input.getVisibleNodes());
      
      net = new whiteice::nnetwork<T>(arch);
      
      for(unsigned int l=0;l<net->getLayers();l++){
	whiteice::math::matrix< T > W;
	whiteice::math::vertex< T > bias;
	
	net->getBias(bias, l);
	net->getWeights(W, l);      
      }
      
      try {
	// copies DBN parameters as nnetwork parameters.. (forward step) [encoder]
	if(net->setWeights(bb_input.getWeights().transpose(), 0) == false) throw "error setting input layer W";
	
	if(net->setBias(bb_input.getBValue(), 0) == false) throw "error setting input layer b";
	
	for(unsigned int l=0;l<layers.size();l++){
	  if(net->setWeights(layers[l].getWeights(), l+1) == false) throw "error setting encoder layer W";
	  if(net->setBias(layers[l].getBValue(), l+1) == false) throw "error setting encoder layer b";
	}
	
	// copies DBN parameters as nnetwork parameters.. (forward step) [decoder]
	int ll = layers.size();
	for(int l=layers.size()-1;l>=0;l--,ll++){
	  if(net->setWeights(layers[l].getWeights().transpose(), ll+1) == false) throw "error setting decoder layer W^t";
	  if(net->setBias(layers[l].getAValue(), ll+1) == false) throw "error setting decoder layer a";
	}
	
	if(net->setWeights(bb_input.getWeights(), ll+1) == false) throw "error setting decoder output layer W^t ";
	if(net->setBias(bb_input.getAValue(), ll+1) == false) throw "error setting decoder output layer a";
	
	net->setStochastic(true);
	net->setNonlinearity(whiteice::nnetwork<T>::sigmoidNonLinearity);
      }
      catch(const char* msg){
	printf("ERROR: %s\n", msg);
	return false;
      }
      
      return true;
      
    }
    
  }


  template class DBN< float >;
  template class DBN< double >;  
  template class DBN< math::blas_real<float> >;
  template class DBN< math::blas_real<double> >;
  
};
