
#include "DBN.h"
#include "dataset.h"
#include "Log.h"

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
  unsigned int DBN<T>::getInputDimension() const
  {
    if(binaryInput) return bb_input.getVisibleNodes();
    else return gb_input.getVisibleNodes();
  }

  
  template <typename T>
  unsigned int DBN<T>::getHiddenDimension() const
  {
    if(layers.size() == 0){
      if(binaryInput) return bb_input.getHiddenNodes();
      else return gb_input.getHiddenNodes();
    }
    else{
      return layers[layers.size()-1].getHiddenNodes();
    }
  }

  template <typename T>
  whiteice::GBRBM<T>& DBN<T>::getInputGBRBM() throw(std::invalid_argument)
  {
    if(binaryInput)
      throw std::invalid_argument("DBN<T>::getInputGBRBM() - but network is BBRBM network");
    return gb_input;
  }

  template <typename T>
  whiteice::BBRBM<T>& DBN<T>::getInputBBRBM() throw(std::invalid_argument)
  {
    if(binaryInput == false)
      throw std::invalid_argument("DBN<T>::getInputBBRBM() - but network is GBRBM network");
    return bb_input;
  }

  template <typename T>
  const whiteice::GBRBM<T>& DBN<T>::getInputGBRBM() const throw(std::invalid_argument)
  {
    if(binaryInput)
      throw std::invalid_argument("DBN<T>::getInputGBRBM() - but network is BBRBM network");
    return gb_input;
  }

  template <typename T>
  const whiteice::BBRBM<T>& DBN<T>::getInputBBRBM() const throw(std::invalid_argument)
  {
    if(binaryInput == false)
      throw std::invalid_argument("DBN<T>::getInputBBRBM() - but network is GBRBM network");
    return bb_input;
  }
  
  // true if GBRBM<T> is used as input otherwise BBRBM<T>
  template <typename T>
  bool DBN<T>::getGaussianInput() const
  {
    return (binaryInput == false);
  }
  
  // layer is [0..getNumberOfLayers-2]
  template <typename T>
  whiteice::BBRBM<T>& DBN<T>::getHiddenLayer(unsigned int layer) throw(std::invalid_argument)
  {
    if(layer >= layers.size())
      throw std::invalid_argument("DBN<T>::getHiddenLayer() - layer out of range");
    return layers[layer];
  }


  template <typename T>
  const whiteice::BBRBM<T>& DBN<T>::getHiddenLayer(unsigned int layer) const throw(std::invalid_argument)
  {
    if(layer >= layers.size())
      throw std::invalid_argument("DBN<T>::getHiddenLayer() - layer out of range");
    return layers[layer];
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
      if(binaryInput == false){
	gb_input.reconstructDataHidden(1); //GBRBM: from visible to hidden
      }
      else{
	bb_input.reconstructData(1);  // BBRBM: v->h
      }
      
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
      
      if(!binaryInput){
	gb_input.reconstructDataHidden2Visible();
	/*
	  math::vertex<T> h,v;
	  gb_input.getHidden(h);
	  gb_input.sampleVisible(v, h);
	  gb_input.setVisible(v);
	*/
      }
      else{
	bb_input.reconstructDataHidden(1);
      }
      
      
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


  // calculates hidden responses (v->h)
  template <typename T>
  bool DBN<T>::calculateHidden(std::vector< math::vertex<T> >& samples)
  {
    for(auto& s : samples){
      if(!setVisible(s)) return false;
      if(!reconstructData(1)) return false;
      s = getHidden();
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
			    const T& errorLevel, const int verbose, const bool* running)
  {
    if(errorLevel < T(0.0)) return false;

    // increase after the whole learning process works..
    const unsigned int ITERLIMIT = 25;
    const unsigned int EPOCH_STEPS = 2;
    
    std::vector< math::vertex<T> > in = samples;
    std::vector< math::vertex<T> > out;
    T error = T(INFINITY);
    std::list<T> errors;

    if(!binaryInput){ // GBRBM
      
      unsigned int iters = 0;
      while((error = gb_input.learnWeights(in, EPOCH_STEPS, verbose, running)) >= errorLevel
	    && iters < ITERLIMIT)
      {
	if(running) if(*running == false) return false; // stop execution
	
	// learns also variance
	iters += EPOCH_STEPS;
	errors.push_back(error);
	
	if(verbose == 1){
	  std::cout << "GBRBM LAYER 0 OPTIMIZATION "
		    << iters << "/" << ITERLIMIT << ": "
		    << error << "/" << errorLevel 
		    << std::endl;
	}
	else if(verbose == 2){
	  char buffer[80];
	  double tmp;
	  whiteice::math::convert(tmp, errorLevel);
	  snprintf(buffer, 80, "DBN::learnWeights: GBRBM layer 0 optimization %d/%d: %f",
		   iters, ITERLIMIT, tmp);
	  whiteice::logging.info(buffer);
	}
	
	while(errors.size() > 5)
	  errors.pop_front();
	
	// calculates mean and variance and stops if variance is
	// within 1% of mean (convergence) during latest
	// EPOCH_STEPS*10 iterations
	
	if(errors.size() >= 5){
	  T m = T(0.0), v = T(0.0);
	  
	  for(auto& e : errors){
	    m += e;
	    v += e*e;
	  }
	  
	  m /= errors.size();
	  v /= errors.size();
	  v -= m*m;
	  
	  v = sqrt(v);

	  if(v/m <= 0.05)
	    break; // stdev is less than 5% of mean
	    
	}
      }

      if(verbose == 2)
	gb_input.diagnostics();

      // maps input to output
      out.clear();
      
      for(auto v : in){
	gb_input.setVisible(v);
	gb_input.reconstructDataHidden(1);
	
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
	if(running) if(*running == false) return false; // stop execution
	
	// learns also variance
	iters += EPOCH_STEPS;
	errors.push_back(error);
	
	if(verbose == 1){
	  std::cout << "BBRBM LAYER 0 OPTIMIZATION "
		    << iters << "/" << ITERLIMIT << ": "
		    << error << "/" << errorLevel 
		    << std::endl;
	}
	else if(verbose == 2){
	  char buffer[80];
	  double tmp;
	  whiteice::math::convert(tmp, errorLevel);
	  snprintf(buffer, 80, "DBN::learnWeights: BBRBM layer 0 optimization %d/%d: %f",
		   iters, ITERLIMIT, tmp);
	  whiteice::logging.info(buffer);	  
	}
	
	while(errors.size() > 5)
	  errors.pop_front();
	
	// calculates mean and variance and stops if variance is
	// within 1% of mean (convergence) during latest
	// EPOCH_STEPS*10 iterations
	
	if(errors.size() >= 5){
	  T m = T(0.0), v = T(0.0);
	  
	  for(auto& e : errors){
	    m += e;
	    v += e*e;
	  }
	  
	  m /= errors.size();
	  v /= errors.size();
	  v -= m*m;
	  
	  v = sqrt(v);

	  if((v/m) <= T(0.05))
	    break; // stdev is less than 5% of mean
	}
      }

      if(verbose == 2)
	bb_input.diagnostics();

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
      while((error = layers[i].learnWeights(in, EPOCH_STEPS, verbose, running)) >= errorLevel &&
	    iters < ITERLIMIT)
      {
	if(running) if(*running == false) return false; // stops execution
	
	iters += EPOCH_STEPS;

	errors.push_back(error);
	
	if(verbose == 1){
	  std::cout << "BBRBM LAYER " << (i+1) << " OPTIMIZATION "
		    << iters << "/" << ITERLIMIT << ": "
		    << error << "/" << errorLevel 
		    << std::endl;
	}
	else if(verbose == 2){
	  char buffer[80];
	  double tmp;
	  whiteice::math::convert(tmp, errorLevel);
	  
	  snprintf(buffer, 80, "DBN::learnWeights: BBRBM layer %d optimization %d/%d: %f",
		   i+1, iters, ITERLIMIT, tmp);
	  whiteice::logging.info(buffer);	  
	}
	
	while(errors.size() > 5)
	  errors.pop_front();
	
	// calculates mean and variance and stops if variance is
	// within 1% of mean (convergence) during latest
	// EPOCH_STEPS*10 iterations
	
	if(errors.size() >= 5){
	  T m = T(0.0), v = T(0.0);
	  
	  for(auto& e : errors){
	    m += e;
	    v += e*e;
	  }
	  
	  m /= errors.size();
	  v /= errors.size();
	  v -= m*m;
	  
	  v = sqrt(v);

	  if(v/m <= 0.05) 
	    break; // stdev is less than 5% of mean
	}
      }

      
      if(verbose == 2)
	layers[i].diagnostics();
      
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

    if(data.size(0) <= 0)
      return false;

    // cluster 0 input values must match input layer dimensions
    if(binaryInput == false){
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
    const unsigned int outputDimension = data.dimension(1);
    const unsigned int hDimension = this->getHidden().size();

    /////////////////////////////////////////////////////////////////
    // final layers linear optimizer (hidden(i) -> data.access(1, i))
    // 
    // solves min E{ 0.5*||Ah + b - y||^2 }
    // 
    // A = Syh * Shh^-1
    // b = m_y - A*m_h

    math::matrix<T> A;
    math::vertex<T> b;

    if(data.size(0) > 10){ // have enough data
      
      // 1. process input data and calculates the deepest hidden values h
      // 2. optimizes hidden values h to map output values:
      //    min ||Ah + b - y||^2 (final linear layer)
      // 3. creates nnetwork<T> and sets parameters (weights)
      //    appropriately
      
      // calculates hidden values
      std::vector< math::vertex<T> > hidden;
      
#if 0
      std::cout << "size(visible) = " << data.size(0) << std::endl;
      std::cout << "x(0) = " << data.access(0,0) << std::endl;
      
      if(binaryInput == false)
	std::cout << "W = " << gb_input.getWeights() << std::endl;
      else
	std::cout << "W = " << bb_input.getWeights() << std::endl;
#endif
      
      math::vertex<T> h;
      T maxabs_h = T(-INFINITY);
      
      for(unsigned int i=0;i<data.size(0);i++){
	// calculates mean-field response without discretization..
	if(this->calculateHiddenMeanField(data.access(0, i), h) == false)
	  return false;

	for(unsigned int k=0;k<h.size();k++)
	  if(maxabs_h < abs(h[k])) maxabs_h = abs(h[k]);
	
	hidden.push_back(h);
      }

      {
	double v = 0.0;
	whiteice::math::convert(v, maxabs_h);

	char buffer[80];
	snprintf(buffer, 80, "DBN::convertToNNetwork(): max abs hidden value %f", v);
	whiteice::logging.info(buffer);
      }
      
      math::matrix<T> Shh(hDimension, hDimension);
      math::matrix<T> Syh(outputDimension, hDimension);
      math::vertex<T> mh(hDimension);
      math::vertex<T> my(outputDimension);
      
      Shh.zero();
      Syh.zero();
      mh.zero();
      my.zero();
      
#if 0
      std::cout << "size(hidden) = " << hidden.size() << std::endl;
      std::cout << "h(0) = " << hidden[0] << std::endl;
      std::cout << "y(0) = " << data.access(1, 0) << std::endl;
#endif
      T maxabs_y = T(-INFINITY);
      
      for(unsigned int i=0;i<hidden.size();i++){
	const auto& h = hidden[i];
	const auto& y = data.access(1, i);
	
	mh += h/T(hidden.size());
	my += y/T(hidden.size());
	
	Shh += h.outerproduct(h)/T(hidden.size());
	Syh += y.outerproduct(h)/T(hidden.size());

	for(unsigned int k=0;k<y.size();k++)
	  if(maxabs_y < abs(y[k])) maxabs_y = abs(y[k]);
      }
      
      Shh -= mh.outerproduct(mh);
      Syh -= my.outerproduct(mh);

      {
	double v = 0.0;
	whiteice::math::convert(v, maxabs_y);

	char buffer[80];
	snprintf(buffer, 80, "DBN::convertToNNetwork(): max abs output value %f", v);
	whiteice::logging.info(buffer);

	std::string shh_str;
	std::string syh_str;

	Shh.toString(shh_str);
	Syh.toString(syh_str);

	std::string line = "DBN::convertToNNetwork(): Shh = " + shh_str;
	whiteice::logging.info(line);

	line = "DBN::convertToNNetwork(): Syh = " + syh_str;
	whiteice::logging.info(line);

	T detShh = Shh.det();
	double detShh_ = 0.0;
	whiteice::math::convert(detShh_, detShh);

	snprintf(buffer, 80, "DBN::convertToNNetwork(): det(Shh) = %e",
		 detShh_);

	whiteice::logging.info(buffer);

	T normSyh = norm_inf(Syh);
	double normSyh_ = 0.0;
	whiteice::math::convert(normSyh_, normSyh);

	snprintf(buffer, 80, "DBN::convertToNNetwork(): norm_inf(Syh) = %e",
		 normSyh_);

	whiteice::logging.info(buffer);
      }
      

      // pseudoinverse always exists..
      // specifies own low precision of singular values: we only keep clearly non-singular elements/data..
      // (values that are set to zero and not inverted)
      Shh.symmetric_pseudoinverse(T(0.001));
      // Shh.pseudoinverse(T(0.0001));
      
      A = Syh*Shh;
      b = my - A*mh;

      {
	std::string A_str;
	A.toString(A_str);
	std::string b_str;
	b.toString(b_str);
	
	std::string line = "DBN::convertToNNetwork() output layer A = " +
	  A_str + " b = " + b_str;
	whiteice::logging.info(line);

	T normA = norm_inf(A);
	double normA_ = 0.0;
	whiteice::math::convert(normA_, normA);
	
	char buffer[80];
	snprintf(buffer, 80, "DBN::convertToNNetwork() output norm_inf(A) = %e",
		 normA_);

	whiteice::logging.info(buffer);
      }
    }
    else{
      // no enough data fills output matrices with random noise to be optimized..
      
      A.resize(outputDimension, hDimension);
      b.resize(outputDimension);

      gb_input.rng.normal(b);
      
      for(unsigned int j=0;j<outputDimension;j++)
	for(unsigned int i=0;i<hDimension;i++)
	  A(j,i) = gb_input.rng.normal();
    }
    
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

    net = new whiteice::nnetwork<T>();
    net->setArchitecture(arch); // we create new net with given architecture (resets all variables)

    // copies DBN parameters as nnetwork parameters..
    if(!binaryInput){
      auto W = gb_input.getWeights().transpose();

      math::vertex<T> v;
      gb_input.getVariance(v);

      for(unsigned int i=0;i<v.size();i++)
	v[i] = T(1.0)/(math::sqrt(v[i]) + T(10e-5)); // no div by zeros..

      assert(v.size() == W.xsize());

      for(unsigned int r=0;r<W.ysize();r++)
	for(unsigned int c=0;c<W.xsize();c++)
	  W(r,c) *= v[c];
	
      if(net->setWeights(W, 0) == false){ delete net; net = nullptr; return false; }
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

#if 0
    std::cout << "DEBUG (last layer A and b)" << std::endl;
    std::cout << "A = " << A << std::endl;
    std::cout << "b = " << b << std::endl;
#endif

    // net->setNonlinearity(whiteice::nnetwork<T>::stochasticSigmoid);
    net->setNonlinearity(whiteice::nnetwork<T>::sigmoid); // mean-field activation
    // last layer is always linear
    net->setNonlinearity(net->getLayers()-1, whiteice::nnetwork<T>::pureLinear);
    
    // printf("DBN convertNNetwork() exit\n");
    // fflush(stdout);

    // internal logging (prints max value of nnetwork)
    {
      whiteice::logging.info("DBN::convertToNNetwork(): DBN->nnetwork analysis");
      net->diagnosticsInfo();
    }
    
    return true;
  }

  
  template <typename T>
  bool DBN<T>::convertToNNetwork(whiteice::nnetwork<T>*& net)
  {
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

    net = new whiteice::nnetwork<T>();
    net->setArchitecture(arch); // we create new net with given architecture (resets all variables)

    // copies DBN parameters as nnetwork parameters..
    if(!binaryInput){
      auto W = gb_input.getWeights().transpose();

      math::vertex<T> v;
      gb_input.getVariance(v);

      for(unsigned int i=0;i<v.size();i++)
	v[i] = T(1.0)/(math::sqrt(v[i]) + T(10e-10)); // no div by zeros..

      assert(v.size() == W.xsize());

      for(unsigned int r=0;r<W.ysize();r++)
	for(unsigned int c=0;c<W.xsize();c++)
	  W(r,c) *= v[c];
	
      if(net->setWeights(W, 0) == false){ delete net; net = nullptr; return false; }
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

    
    net->setNonlinearity(whiteice::nnetwork<T>::sigmoid);
    
    return true;
  }


  // converts inverse (from hidden to visible) DBN to nnetwork
  template <typename T>
  bool DBN<T>::convertInverseToNNetwork(whiteice::nnetwork<T>*& net)
  {

    if(!binaryInput){
      //////////////////////////////////////////////////////////
      // creates feedforward neural network
      
      std::vector<unsigned int> arch; // architecture

      if(layers.size() > 0)
	arch.push_back(layers[layers.size()-1].getHiddenNodes());
      else
	arch.push_back(gb_input.getHiddenNodes());
      
      // inverted network
      if(layers.size() > 0){
	for(int i=layers.size()-1;i>=0;i--)
	  arch.push_back(layers[i].getVisibleNodes());
      }
      
      arch.push_back(gb_input.getVisibleNodes()); // output
      
      net = new whiteice::nnetwork<T>(arch);
      
      try {
	
	// copies DBN parameters as nnetwork parameters.. (forward step) [decoder]
	int ll = 0;
	for(int l=layers.size()-1;l>=0;l--,ll++){
	  if(net->setWeights(layers[l].getWeights().transpose(), ll) == false) throw "error setting decoder layer W^t";
	  if(net->setBias(layers[l].getAValue(), ll) == false) throw "error setting decoder layer a";
	}
	
	{
	  auto W = gb_input.getWeights();
	  
	  math::vertex<T> v;
	  gb_input.getVariance(v);
	  
	  for(unsigned int i=0;i<v.size();i++)
	    v[i] = math::sqrt(v[i]);

	  assert(v.size() == W.ysize());
	  
	  for(unsigned int r=0;r<W.ysize();r++)
	    for(unsigned int c=0;c<W.xsize();c++)
	      W(r,c) = v[r] * W(r,c);
	  
	  if(net->setWeights(W, ll) == false) throw "error setting decoder output layer W^t ";
	}
	
	if(net->setBias(gb_input.getAValue(), ll) == false) throw "error setting decoder output layer a";
	
	net->setNonlinearity(whiteice::nnetwork<T>::sigmoid);

	// output layer is not stochastic sigmoid but pure linear (gaussian output) ???
	net->setNonlinearity(ll, whiteice::nnetwork<T>::pureLinear);
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
      
      if(layers.size() > 0){
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
      
      try {
	// copies DBN parameters as nnetwork parameters.. (forward step) [decoder]
	int ll = 0;
	for(int l=layers.size()-1;l>=0;l--,ll++){
	  if(net->setWeights(layers[l].getWeights().transpose(), ll) == false) throw "error setting decoder layer W^t";
	  if(net->setBias(layers[l].getAValue(), ll) == false) throw "error setting decoder layer a";
	}
	
	if(net->setWeights(bb_input.getWeights(), ll) == false) throw "error setting decoder output layer W^t ";
	if(net->setBias(bb_input.getAValue(), ll) == false) throw "error setting decoder output layer a";
	
	net->setNonlinearity(whiteice::nnetwork<T>::sigmoid);
      }
      catch(const char* msg){
	printf("ERROR: %s\n", msg);
	return false;
      }
      
      return true;
    }

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

      
      try {
	// copies DBN parameters as nnetwork parameters.. (forward step) [encoder]

	{	  
	  // if(net->setWeights(gb_input.getWeights().transpose(), 0) == false) throw "error setting input layer W";
	  
	  auto W = gb_input.getWeights().transpose();

	  math::vertex<T> v;
	  gb_input.getVariance(v);
	  
	  for(unsigned int i=0;i<v.size();i++)
	    v[i] = T(1.0)/(math::sqrt(v[i]) + T(10e-10)); // no div by zeros..

	  assert(v.size() == W.xsize());
	  
	  for(unsigned int r=0;r<W.ysize();r++)
	    for(unsigned int c=0;c<W.xsize();c++)
	      W(r,c) *= v[c];
	  
	  if(net->setWeights(W, 0) == false) throw "error setting input layer W";
	}
	
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
	
	{
	  // if(net->setWeights(gb_input.getWeights(), ll+1) == false) throw "error setting decoder output layer W^t ";
	  auto W = gb_input.getWeights();
	  
	  math::vertex<T> v;
	  gb_input.getVariance(v);
	  
	  for(unsigned int i=0;i<v.size();i++)
	    v[i] = math::sqrt(v[i]);

	  assert(v.size() == W.ysize());
	  
	  for(unsigned int r=0;r<W.ysize();r++)
	    for(unsigned int c=0;c<W.xsize();c++)
	      W(r,c) = v[r] * W(r,c);
	  
	  if(net->setWeights(W, ll+1) == false)
	    throw "error setting decoder output layer W^t ";
	}
	
	if(net->setBias(gb_input.getAValue(), ll+1) == false)
	  throw "error setting decoder output layer a";
	
	net->setNonlinearity(whiteice::nnetwork<T>::sigmoid);

	// output layer is not stochastic sigmoid but pure linear (gaussian output) ???
	net->setNonlinearity(ll+1, whiteice::nnetwork<T>::pureLinear);
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
      
      try {
	// copies DBN parameters as nnetwork parameters.. (forward step) [encoder]
	// if(net->setWeights(bb_input.getWeights().transpose(), 0) == false) throw "error setting input layer W";
	if(net->setWeights(bb_input.getWeights(), 0) == false) throw "error setting input layer W";
	
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
	
	net->setNonlinearity(whiteice::nnetwork<T>::sigmoid);
      }
      catch(const char* msg){
	printf("ERROR: %s\n", msg);
	return false;
      }
      
      return true;
      
    }
    
  }

  
  // prints to log max/min values of DBN network
  template <typename T>
  bool DBN<T>::diagnostics() const
  {
    whiteice::logging.info("DBN::diagnostics()");

    if(binaryInput == false){
      if(gb_input.diagnostics() == false) return false;
    }
    else{
      if(bb_input.diagnostics() == false) return false;
    }

    for(unsigned int i=0;i<layers.size();i++){
      if(layers[i].diagnostics() == false) return false;
    }

    return true;
  }

  
  template <typename T>
  bool DBN<T>::save(const std::string& basefilename) const
  {
    char buffer[256];

    // saves metadata
    {
      snprintf(buffer, 256, "%s", basefilename.c_str());

      whiteice::dataset<T> conf;

      conf.createCluster("DBN type", 1);

      whiteice::math::vertex<T> v;
      v.resize(1);

      if(binaryInput) v[0] = 1.0f;
      else v[0] = 0.0f;

      conf.add(0, v);

      conf.createCluster("DBN layers", 1);
      v[0] = layers.size();

      conf.add(1, v);

      if(conf.save(buffer) == false) return false;
    }


    // saves input layer
    {
      snprintf(buffer, 256, "%s-input", basefilename.c_str());
      
      if(binaryInput == false){      
	if(gb_input.save(buffer) == false)
	  return false;
      }
      else{
	if(bb_input.save(buffer) == false)
	  return false;
      }
    }
    
    // saves other layers
    for(unsigned int i=0;i<layers.size();i++){
      snprintf(buffer, 256, "%s-layer-%d", basefilename.c_str(), i);
      if(layers[i].save(buffer) == false)
	return false;
    }

    return true;
  }

  
  template <typename T>
  bool DBN<T>::load(const std::string& basefilename)
  {
    bool loadedBinaryInput;
    unsigned int numLayers;

    char buffer[256];
    
    // loads metadata
    {
      snprintf(buffer, 256, "%s", basefilename.c_str());

      whiteice::dataset<T> conf;

      if(conf.load(buffer) == false)
	return false;

      if(conf.getNumberOfClusters() != 2)
	return false;

      if(conf.size(0) != conf.size(1) && conf.size(0) != 1)
	return false;

      whiteice::math::vertex<T> v;

      v = conf.access(0, 0);

      if(v[0] > 0.5f) loadedBinaryInput = true;
      else loadedBinaryInput = false;

      v = conf.access(1, 0);

      auto tmp = floor(v[0]);

      whiteice::math::convert(numLayers, tmp);

      if(numLayers < 0) return false;
    }

    whiteice::GBRBM<T> gb;
    whiteice::BBRBM<T> bb;

    std::vector< whiteice::BBRBM<T> > bb_layers;
    bb_layers.resize(numLayers);

    
    // tries to load input layer
    {
      snprintf(buffer, 256, "%s-input", basefilename.c_str());
      
      if(loadedBinaryInput == false){      
	if(gb.load(buffer) == false)
	  return false;
      }
      else{
	if(bb.load(buffer) == false)
	  return false;
      }
    }
    
    // loads other layers
    for(unsigned int i=0;i<bb_layers.size();i++){
      snprintf(buffer, 256, "%s-layer-%d", basefilename.c_str(), i);
      if(bb_layers[i].load(buffer) == false)
	return false;
    }


    // finally copies loaded data structures from temporal data structures
    {
      binaryInput = loadedBinaryInput;
      
      if(binaryInput) bb_input = bb;
      else gb_input = gb;
      
      layers = bb_layers;
    }

    return true;
  }


  // calculates mean field sigmoid response of input
  // (used by convertToNNetwork())
  template <typename T>
    bool DBN<T>::calculateHiddenMeanField(const math::vertex<T>& v,
					  math::vertex<T>& h) const
  {
    if(binaryInput == false){
      // GBRBM: v->h
      if(gb_input.calculateHiddenMeanField(v, h) == false)
	return false;
    }
    else{
      // BBRBM: v->h
      if(bb_input.calculateHiddenMeanField(v, h) == false)
	return false;
    }

    for(unsigned int i=0;i<layers.size();i++){
      // BBRBM: v->h
      auto tmp_input = h;
      if(layers[i].calculateHiddenMeanField(tmp_input, h) == false)
	return false;
    }

    return true;
  }


  // calculates mean field response to input to input h->v
  template <typename T>
    bool DBN<T>::calculateVisibleMeanField(const math::vertex<T>& h,
					   math::vertex<T>& v) const
  {
    auto input = h;
    
    for(unsigned int i=0;i<layers.size();i++){
      // BBRBM: h->v
      if(layers[i].calculateVisibleMeanField(input, v) == false)
	return false;
      input = v;
    }
    
    if(binaryInput == false){
      // GBRBM: h->v
      if(gb_input.calculateVisibleMeanField(input, v) == false)
	return false;
    }
    else{
      // BBRBM: h->v
      if(bb_input.calculateVisibleMeanField(input, v) == false)
	return false;
    }

    return true;
  }
  

  template class DBN< float >;
  template class DBN< double >;  
  template class DBN< math::blas_real<float> >;
  template class DBN< math::blas_real<double> >;
  
};
