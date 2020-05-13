/*
 * Variational Autoencoder code
 * Tomas Ukkonen <tomas.ukkonen@iki.fi> 2020
 * 
 * Implementation research paper:
 * "Auto-Encoding Variational Bayes. Diedrik Kingma and Max Welling."
 * Implementation extra math notes in docs directory:
 * [variational-bayes-encoder.tm]
 *
 * NOTE: the input data x ~ p(x) should be roughly distributed as Normal(0,I)
 * because it is assumed that p(x|z) don't estimate variance term but
 * fixes it to be Var[x_i] = 1 for each dimension.
 */

#include "VAE.h"
#include <list>


namespace whiteice
{

  template <typename T>
  VAE<T>::VAE(const VAE<T>& vae)
  {
    this->encoder = vae.encoder;
    this->decoder = vae.decoder;

    this->minibatchMode = vae.minibatchMode;
  }
  
  template <typename T>
  VAE<T>::VAE(const nnetwork<T>& encoder, // x -> (z_mean, z_var)
	      const nnetwork<T>& decoder) // z -> (x_mean)
    
  {
    if(encoder.output_size() & 1) // odd value
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");
    if(encoder.input_size() != decoder.output_size()) // dim(x) != dim(d(e(x)))
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");
    if(encoder.output_size() != 2*decoder.input_size()) // dim(e(x)/2) != dim(z)
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");

    if(encoder.input_size() <= 0 || decoder.output_size() <= 0 ||
       encoder.output_size() <= 0 || decoder.input_size() <= 0)
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");

    this->encoder = encoder;
    this->decoder = decoder;

    this->minibatchMode = false;
  }

  
  template <typename T>
  VAE<T>::VAE(const std::vector<unsigned int> encoderArchitecture, // x -> (z_mean, z_var)
	      const std::vector<unsigned int> decoderArchitecture) // z -> (x_mean)
    
    
  {
    if(encoderArchitecture.size() < 2 || decoderArchitecture.size() < 2)
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");

    
    if(encoderArchitecture[encoderArchitecture.size()-1] & 1) // odd value
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");
    
    if(encoderArchitecture[0] !=
       decoderArchitecture[decoderArchitecture.size()-1]) // dim(x) != dim(d(e(x)))
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");

    if(2*decoderArchitecture[0] !=
       encoderArchitecture[encoderArchitecture.size()-1]) // dim(e(x)/2) != dim(z)
      throw std::invalid_argument("bad encoder/decoder dimensions/arch");

    for(unsigned int i=0;i<encoderArchitecture.size();i++){
      if(encoderArchitecture[i] <= 0)
	throw std::invalid_argument("bad encoder/decoder dimensions/arch");
    }

    for(unsigned int i=0;i<decoderArchitecture.size();i++){
      if(decoderArchitecture[i] <= 0)
	throw std::invalid_argument("bad encoder/decoder dimensions/arch");
    }
    
    this->encoder.setArchitecture(encoderArchitecture, nnetwork<T>::halfLinear);
    this->decoder.setArchitecture(decoderArchitecture, nnetwork<T>::halfLinear);
  }


  template <typename T>
  void VAE<T>::getModel(nnetwork<T>& encoder,
			nnetwork<T>& decoder)
  {
    encoder = this->encoder;
    decoder = this->decoder;
  }
  
  
  template <typename T>
  bool VAE<T>::setModel(const nnetwork<T>& encoder,
			const nnetwork<T>& decoder)
  {
    if(encoder.output_size() & 1) // odd value
      return false;
    if(encoder.input_size() != decoder.output_size()) // dim(x) != dim(d(e(x)))
      return false;
    if(encoder.output_size() != 2*decoder.input_size()) // dim(e(x)) != 2*dim(z)
      return false;

    if(encoder.input_size() <= 0 || decoder.output_size() <= 0 ||
       encoder.output_size() <= 0 || decoder.input_size() <= 0)
      return false;

    this->encoder = encoder;
    this->decoder = decoder;

    return true;
  }


  template <typename T>
  bool VAE<T>::setModel(const std::vector<unsigned int> encoderArchitecture,
			const std::vector<unsigned int> decoderArchitecture)
  {
    if(encoderArchitecture.size() < 2 || decoderArchitecture.size() < 2)
      return false;
    
    if(encoderArchitecture[encoderArchitecture.size()-1] & 1) // odd value
      return false;
    
    if(encoderArchitecture[0] !=
       decoderArchitecture[decoderArchitecture.size()-1]) // dim(x) != dim(d(e(x)))
      return false;

    if(2*decoderArchitecture[0] !=
       encoderArchitecture[encoderArchitecture.size()-1]) // dim(e(x)/2) != dim(z)
      return false;

    for(unsigned int i=0;i<encoderArchitecture.size();i++){
      if(encoderArchitecture[i] <= 0)
	return false;
    }

    for(unsigned int i=0;i<decoderArchitecture.size();i++){
      if(decoderArchitecture[i] <= 0)
	return false;
    }
    
    this->encoder.setArchitecture(encoderArchitecture, nnetwork<T>::halfLinear);
    this->decoder.setArchitecture(decoderArchitecture, nnetwork<T>::halfLinear);

    return true;
  }
  
  
  // x -> z (hidden) [returns N(zm,zv) distribution parameters]
  template <typename T>
  bool VAE<T>::encode(const math::vertex<T>& x,
		      math::vertex<T>& zmean,
		      math::vertex<T>& zstdev) const
  {
    math::vertex<T> result;
    if(encoder.calculate(x, result) == false)
      return false;

    zmean.resize(decoder.input_size());
    zstdev.resize(decoder.input_size());

    if(decoder.input_size()*2 != result.size())
      return false;
    
    for(unsigned int i=0;i<zmean.size();i++){
      zmean[i] = result[i];
      zstdev[i] = result[i+zmean.size()];

      T value = zstdev[i];

      if(value >= T(20.0)) // restrict values so we don't go to infinity
	value = T(20.0);
      else if(value <= T(-20.0))
	value = T(-20.0);
      
      zstdev[i] = ::exp(value.c[0]);
    }

    return true;
  }


  // x -> z (hidden) [returns sample from ~ Normal(zm(x),zv(x))]
  template <typename T>
  bool VAE<T>::encodeSample(const math::vertex<T>& x,
			    math::vertex<T>& zsample) const
  {
    math::vertex<T> result;
    if(encoder.calculate(x, result) == false)
      return false;

    RNG<T> rng;
    zsample.resize(decoder.input_size());

    if(decoder.input_size()*2 != result.size())
      return false;
    
    for(unsigned int i=0;i<zsample.size();i++){
      auto zmean = result[i];
      T value = result[i+zsample.size()];
      
      if(value >= T(20.0)) // restrict values so we don't go to infinity
	value = T(20.0);
      else if(value <= T(-20.0))
	value = T(-20.0);
      
      auto zstdev = ::exp(value.c[0]);
      
      zsample[i] = zmean + rng.normal()*zstdev;
    }

    return true;
  }
  
  
  // z (hidden) -> xmean (variance = I)
  template <typename T>
  bool VAE<T>::decode(const math::vertex<T>& z,
		      math::vertex<T>& xmean) const
  {
    if(decoder.calculate(z, xmean) == false)
      return false;
    
    return true;
  }

  
  template <typename T>
  unsigned int VAE<T>::getDataDimension() const
  {
    return encoder.input_size();
  }

  
  template <typename T>
  unsigned int VAE<T>::getEncodedDimension() const
  {
    return encoder.output_size();
  }
  

  template <typename T>
  void VAE<T>::getParameters(math::vertex<T>& p) const
  {
    math::vertex<T> p_encoder, p_decoder;
    if(encoder.exportdata(p_encoder) == false || decoder.exportdata(p_decoder) == false)
      return;

    p.resize(p_encoder.size() + p_decoder.size());
    p.write_subvertex(p_encoder, 0);
    p.write_subvertex(p_decoder, p_encoder.size());
  }


  template <typename T>
  bool VAE<T>::setParameters(const math::vertex<T>& p)
  {
    math::vertex<T> p_encoder, p_decoder;
    if(encoder.exportdata(p_encoder) == false || decoder.exportdata(p_decoder) == false)
      return false;

    if(p.size() != p_encoder.size() + p_decoder.size())
      return false;

    if(p.subvertex(p_encoder, 0, p_encoder.size()) == false)
      return false;

    if(p.subvertex(p_decoder, p_encoder.size(), p_decoder.size()) == false)
      return false;

    if(encoder.importdata(p_encoder) == false ||
       decoder.importdata(p_decoder) == false)
      return false;
    
    return true;
  }

  // to set minibatch mode in which we use only sample of 30 data points when calculating gradient
  template <typename T>
  void VAE<T>::setUseMinibatch(bool use_minibatch)
  {
    minibatchMode = use_minibatch;
  }

  template <typename T>
  bool VAE<T>::getUseMinibatch()
  {
    return minibatchMode;
  }

  template <typename T>
  void VAE<T>::initializeParameters()
  {
    encoder.randomize();
    decoder.randomize();
  }


  template <typename T>
  T VAE<T>::getError(const std::vector< math::vertex<T> >& xsamples) const
  {
    T error = T(0.0);

#pragma omp parallel shared(error)
    {
      math::vertex<T> zmean, zstdev, xmean;
      T e = T(0.0);

      whiteice::RNG<T> rng;
      math::vertex<T> epsilon;
      epsilon.resize(encoder.output_size()/2);

#pragma omp for nowait schedule(dynamic)
      for(unsigned int i=0;i<xsamples.size();i++){
	encode(xsamples[i], zmean, zstdev);

	auto zi = zmean;
	for(unsigned int k=0;k<zmean.size();k++){
	  zi[k] += zstdev[k]*epsilon[k];
	}
	
	decode(zi, xmean);
	
	auto delta = xsamples[i] - xmean;
	e += delta.norm();
      }

#pragma omp critical
      {
	error += e;
      }
    }

    if(xsamples.size() > 0)
      error /= T(xsamples.size());

    return error;
  }
    
  
  
  // x = samples, learn x = decode(encode(x))
  // optimizes using gradient descent,
  // divides data to teaching and test
  // datasets (50%/50%)
  //
  // model_stdev_error(last N iters)/model_mean_error(last N iters) < convergence_ratio 
  //
  template <typename T>
  bool VAE<T>::learnParameters(const std::vector< math::vertex<T> >& xsamples,
			       T convergence_ratio, bool verbose, LoggingInterface* messages,
			       bool* running)
  {
    // implements gradient descent
    if(convergence_ratio <= T(0.0) || convergence_ratio >= T(1.0))
      return false;

    if(running){
      if(*running == false){
	return false;
      }
    }
    

    std::list<T> errors;
    T error = getError(xsamples);
    T lrate = T(0.0001);
    unsigned int counter = 0;
    const int BUFLEN = 1024;
    char buf[BUFLEN];
    
    
    while(1){
      if(running){
	if(*running == false){
	  printf("VAE::learn() aborting computation\n");
	  if(messages) messages->printMessage("VAE::learn() aborting computation\n");
	  break;
	}
      }
      
      // gradient search of better solution
      math::vertex<T> grad;
      if(calculateGradient(xsamples, grad, messages, running) == false){
	if(verbose){
	  std::cout << "calculateGradient() returns false!" << std::endl;
	  if(messages) messages->printMessage("VAE::calculateGradient() returns false!\n");
	  std::cout << std::flush;;
	}
	return false;
      }
      
      if(verbose){
	std::cout << "norm(grad) == " << grad.norm() << std::endl;
	std::cout << std::flush;;
      }
      
      // FIXME write proper line search in grad direction
      math::vertex<T> params, p;
      getParameters(params);

      if(verbose){
	std::cout << "[before update] norm(params) = " << params.norm() << std::endl;
	std::cout << std::flush;;
      }
      
      bool found = false;
      
      T eprev = error;
      // lrate = T(1.0);
      lrate *= T(4.0);
      
      while(eprev <= error && lrate > T(10e-30)){
          if(running){
    	if(*running == false){
    	  printf("VAE::learn() aborting computation\n");
    	  if(messages) messages->printMessage("VAE::learn()/gradient-search: aborting computation\n");
    	  break;
    	}
          }


	p = params;
	p += lrate*grad;
	setParameters(p);

	error = getError(xsamples);
	
	if(error < eprev){
	  lrate *= T(2.0);
	  found = true;
	}
	else if(error >= eprev){
	  lrate *= T(0.50);
	}
      }

      if(found){
	params = p;
	setParameters(p);
      }
      else{
	error = getError(xsamples);
	setParameters(params);
      }
      
      counter++;

      if(verbose){
	std::cout << "[after update] norm(params) = " << params.norm() << std::endl;
	std::cout << "ERROR " << counter << ": " << error << std::endl;

	float errf = 10e10;
	whiteice::math::convert(errf, error);

	snprintf(buf, BUFLEN, "VAE::learn() iter %d error: %f\n", counter, errf);
	if(messages) messages->printMessage(buf);
	
	std::cout << std::flush;
      }

      errors.push_back(error);

      while(errors.size() > 20)
	errors.pop_front();

      // check for convergence
      if(errors.size() >= 20){
	T m = T(0.0), v = T(0.0);
	  
	for(const auto& e : errors){
	  m += e;
	}
	
	m /= errors.size();
	
	for(const auto& e : errors){
	  v += (e - m)*(e - m);
	}
	  
	v /= (errors.size() - 1);
	  
	v = sqrt(abs(v));
	
	if(true){
	  if(verbose){
	    T ratio = T(100.0)*v/m;
	    
	    std::cout << "ERROR CONVERGENCE " << ratio << "%" << std::endl;

	    float ratiof = 10e10;
	    whiteice::math::convert(ratiof, ratio);

	    float convf  = 10e10;
	    T convp = T(100.0)*convergence_ratio;
	    whiteice::math::convert(convf, convp);
	    
	    snprintf(buf, BUFLEN, "ERROR CONVERGENCE: %f (> %f)\n", ratiof, convf);
	    if(messages) messages->printMessage(buf);
	    
	    std::cout << std::flush;;
	  }
	      
	}

	if(v/m <= convergence_ratio)
	  break; // stdev is less than c% of mean (5%)
      }
      
    }
    
    return true;
  }

  
  // calculates gradient of parameter p using all samples
  template <typename T>
  bool VAE<T>::calculateGradient(const std::vector< math::vertex<T> >& xsamples,
				 math::vertex<T>& pgradient,
				 LoggingInterface* messages,
				 bool* running)
  {
    pgradient.resize(encoder.gradient_size() + decoder.gradient_size());
    pgradient.zero();
    
    bool failure = false;
    const bool verbose = false;
    unsigned int MINIBATCHSIZE = 0; // number of samples used to estimate gradient

    const int BUFLEN = 1024;
    char buffer[BUFLEN];

    if(minibatchMode)
      MINIBATCHSIZE = 30;
    else
      MINIBATCHSIZE = xsamples.size(); // use all samples

    if(xsamples.size() <= 0) failure = true;

    if(verbose){
      printf("CALCULATE GRADIENT CALLED\n");
      fflush(stdout);
    }

#pragma omp parallel shared(pgradient)
    {
      whiteice::RNG<T> rng;
      math::vertex<T> pgrad;
      pgrad.resize(pgradient.size());
      pgrad.zero();

#pragma omp for nowait schedule(dynamic)
      for(unsigned int i=0;i<MINIBATCHSIZE;i++)
      {
	if(failure) continue; // do nothing after first failure
	if(running){
		if(*running == false){
			failure = true;
			continue; // do nothing if computation has been stopped
		}
	}

  	if(verbose){
  		snprintf(buffer, BUFLEN, "Calculating gradient sample %d/%d..\n", i, MINIBATCHSIZE);
  	  if(messages) messages->printMessage(buffer);
  	  printf(buffer);
  	  fflush(stdout);
  	}


	unsigned int index = 0;
	
	if(minibatchMode){
	  index = rng.rand() % ((unsigned int)xsamples.size());
	}
	else{
	  index = i;
	}
	
	
	if(verbose){
	  printf("PROCESSING SAMPLE %d/%d\n", index, (int)xsamples.size());
	  fflush(stdout);
	}
	
	const auto& x = xsamples[index];
	// 1. first calculate gradient of "-D_KL" term
	
	// encoder values
	math::vertex<T> zmean, zstdev;
	math::vertex<T> output;
	math::vertex<T> ones;
	
	if(encoder.calculate(x, output) == false){
	  failure = true;
	  continue;
	}
	
	zmean.resize(output.size()/2);
	zstdev.resize(output.size()/2);
	ones.resize(output.size()/2);
	
	if(output.subvertex(zmean, 0, zmean.size()) == false){
	  failure = true;
	  continue;
	}
	
	if(output.subvertex(zstdev, zmean.size(), zstdev.size()) == false){
	  failure = true;
	  continue;
	}

	for(unsigned int j=0;j<zstdev.size();j++){
	  T value = zstdev[j];
	  
	  if(value >= T(20.0)) // restrict values so we don't go to infinity
	    value = T(20.0);
	  else if(value <= T(-20.0))
	    value = T(-20.0);
	  
	  zstdev[j] = ::exp(value.c[0]); // convert s = Log(stdev) to proper standard deviation
	  ones[j] = T(1.0);
	}
	
	// gradient of encoder
	math::matrix<T> J; // Jacobian matrix
	math::matrix<T> Jmean, Jstdev;
	
	if(encoder.gradient(x, J) == false){
	  failure = true;
	  continue;
	}
	
	assert(J.ysize() == zmean.size()+zstdev.size());
	assert(J.xsize() == encoder.gradient_size());
	
	if(J.submatrix(Jmean,
		       0, 0,
		       encoder.gradient_size(), zmean.size()) == false)
	{
	  failure = true;
	  continue;
	}
	
	
	if(J.submatrix(Jstdev,
		       0, zmean.size(),
		       encoder.gradient_size(), zstdev.size()) == false)
	{
	  failure = true;
	  continue;
	}


	auto zvar = zstdev;
	for(unsigned int j=0;j<zvar.size();j++)
	  zvar[j] = zvar[j]*zvar[j];
	
	auto g1 = T(-1.0)*zmean*Jmean + (ones + T(-1.0)*zvar)*Jstdev;
	
	
	// 2. second calculate gradient
	math::vertex<T> decoder_gradient, encoder_gradient;
	decoder_gradient.resize(decoder.gradient_size());
	encoder_gradient.resize(encoder.gradient_size());
	decoder_gradient.zero();
	encoder_gradient.zero();
	
	const unsigned int N = 15;
	
	math::vertex<T> epsilon;
	epsilon.resize(zmean.size());
	

	if(verbose){
	  printf("%d/%d: CALCULATING 2ND GRADIENT\n", index, (int)xsamples.size());
	  fflush(stdout);
	}

	for(unsigned int j=0;j<N;j++){
	  // epsilon is ~ Normal(0,I)
	  rng.normal(epsilon);
	  
	  auto zi = zmean;
	  for(unsigned int k=0;k<zmean.size();k++){
	    zi[k] += zstdev[k]*epsilon[k];
	  }
	  
	  auto xmean = x;
	  if(decoder.calculate(zi, xmean) == false){
	    failure = true;
	    continue;
	  }
	  
	  math::matrix<T> grad_meanx;
	  if(decoder.gradient(zi, grad_meanx) == false){
	    failure = true;
	    continue;
	  }
	  
	  decoder_gradient += (x - xmean)*grad_meanx;
	  
	  math::matrix<T> J_meanx_value;
	  if(decoder.gradient_value(zi, J_meanx_value) == false){
	    failure = true;
	    continue;
	  }

	  
	  auto gzx = (x - xmean)*J_meanx_value; // first part of gradient
	  
	  auto Jstdev_epsilon = Jstdev;
	  for(unsigned int y=0;y<Jstdev.ysize();y++)
	    for(unsigned int x=0;x<Jstdev.xsize();x++)
	      Jstdev_epsilon(y,x) = epsilon[y]*zstdev[y]*Jstdev(y,x);
	  
	  // encoder_gradient += gzx * (Jmean + epsilon*Jstdev); // second part of gradient
	  encoder_gradient += gzx * (Jmean + Jstdev_epsilon); // second part of gradient
	}

	encoder_gradient *= T(1.0/N);
	decoder_gradient *= T(1.0/N);

	if(verbose){
	  printf("%d/%d: CALCULATING 2ND GRADIENT: SUM CALCULATED\n", i, (int)xsamples.size());
	  fflush(stdout);
	}
	
	// calculates gradient
	auto pg = pgradient;
	pg.zero();
	
	if(pg.write_subvertex(g1, 0) == false){
	  failure = true;
	  continue;
	}
	
	pgrad += pg;

	// HEURISTICS: scales "dimensionality" of p(x) to be same as in p(z) which are both
	// guessed to be Normal(0,I) distributed.
	// DISABLED:
	// encoder_gradient *= T((float)zmean.size()/((float)x.size()));
	// decoder_gradient *= T((float)zmean.size()/((float)x.size()));
	
	pg.zero();
	if(pg.write_subvertex(encoder_gradient, 0) == false){
	  failure = true;
	  continue;
	}
	
	if(pg.write_subvertex(decoder_gradient, encoder_gradient.size()) == false){
	  failure = true;
	  continue;
	}
	
	pgrad += pg;
      }
      

#pragma omp critical
      if(!failure){
	pgradient += pgrad;
      }

    }

    if(failure) return false;

    if(xsamples.size() > 0)
      pgradient /= T(MINIBATCHSIZE);

    T nrm = pgradient.norm();

    if(nrm > T(0.0)){
      pgradient /= nrm; // is this really needed?
    }
    
    return true;
  }
  

  template <typename T>
  bool VAE<T>::load(const std::string& filename) 
  {
    if(filename.size() <= 0) return false;
    
    const std::string encoderfile = filename + ".vae-encoder";
    const std::string decoderfile = filename + ".vae-decoder";
    
    auto en = encoder;
    auto de = decoder;

    if(en.load(encoderfile) == false || de.load(decoderfile) == false)
      return false;

    if(en.input_size() != de.output_size() || en.output_size() != 2*de.input_size())
      return false;

    encoder = en;
    decoder = de;

    return true;
  }


  template <typename T>
  bool VAE<T>::save(const std::string& filename) const 
  {
    if(filename.size() <= 0) return false;
    
    const std::string encoderfile = filename + ".vae-encoder";
    const std::string decoderfile = filename + ".vae-decoder";
    
    if(encoder.save(encoderfile) == false || decoder.save(decoderfile) == false)
      return false;

    return true;
  }
  
  
  
  // template class VAE< float >;
  // template class VAE< double >;  
  template class VAE< math::blas_real<float> >;
  template class VAE< math::blas_real<double> >;
}
