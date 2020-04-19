/*
 * Variational Autoencoder code
 * Tomas Ukkonen <tomas.ukkonen@iki.fi> 2020
 * 
 * Implementation research paper:
 * "Auto-Encoding Variational Bayes. Diedrik Kingma and Max Welling."
 * Implementation extra math notes in hyperplane2 repository (non-free) 
 * [variational-bayes-encoder.tm]
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
  }
  
  template <typename T>
  VAE<T>::VAE(const nnetwork<T>& encoder, // x -> (z_mean, z_var)
	      const nnetwork<T>& decoder) // z -> (x_mean)
    throw(std::invalid_argument)
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
  }

  
  template <typename T>
  VAE<T>::VAE(const std::vector<unsigned int> encoderArchitecture, // x -> (z_mean, z_var)
	      const std::vector<unsigned int> decoderArchitecture) // z -> (x_mean)
    
    throw(std::invalid_argument)
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

#pragma omp for nowait schedule(dynamic)
      for(unsigned int i=0;i<xsamples.size();i++){
	encode(xsamples[i], zmean, zstdev);
	decode(zmean, xmean);
	
	auto delta = xsamples[i] - xmean;
	e += delta.norm();
      }

#pragma omp critical
      {
	error += e;
      }
    }
    
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
			       T convergence_ratio)
  {
    // implements gradient descent
    if(convergence_ratio <= T(0.0) || convergence_ratio >= T(1.0))
      return false;

    std::list<T> errors;
    T error = getError(xsamples);
    T lrate = T(0.0001);
    unsigned int counter = 0;
    
    
    while(1){
      // gradient search of better solution
      math::vertex<T> grad;
      if(calculateGradient(xsamples, grad) == false){
	std::cout << "calculateGradient() returns false!" << std::endl;
      }

      std::cout << "norm(grad) == " << grad.norm() << std::endl;
      
      // FIXME write proper line search in grad direction
      math::vertex<T> params, p;
      getParameters(params);
      std::cout << "[before update] norm(params) = " << params.norm() << std::endl;
      bool found = false;
      
      T eprev = error;
      lrate *= T(4.0);
      
      while(eprev <= error && lrate > T(10e-15)){
	p = params;
	p += lrate*grad;
	setParameters(p);

	error = getError(xsamples);
	
	if(error < eprev){
	  lrate *= T(2.0);
	  found = true;
	}
	else if(error > eprev){
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
      
      std::cout << "[after update] norm(params) = " << params.norm() << std::endl;
      std::cout << "ERROR " << counter << ": " << error << std::endl;

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
	  std::cout << "ERROR CONVERGENCE " << T(100.0)*v/m << "%" << std::endl;
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
				 math::vertex<T>& pgradient)
  {
    pgradient.resize(encoder.gradient_size() + decoder.gradient_size());
    pgradient.zero();
    
    bool failure = false;

#pragma omp parallel shared(pgradient)
    {
      whiteice::RNG<T> rng;
      math::vertex<T> pgrad;
      pgrad.resize(pgradient.size());
      pgrad.zero();

#pragma omp for nowait schedule(dynamic)
      for(unsigned int i=0;i<xsamples.size();i++)
      {
	if(failure) continue; // do nothing after first failure
	
	const auto& x = xsamples[i];
	// 1. first calculate gradient of "-D_KL" term
	
	// encoder values
	math::vertex<T> zmean, zstdev, inv_zstdev;
	math::vertex<T> output;
	
	if(encoder.calculate(x, output) == false){
	  failure = true;
	  continue;
	}
	
	zmean.resize(output.size()/2);
	zstdev.resize(output.size()/2);
	if(output.subvertex(zmean, 0, zmean.size()) == false){
	  failure = true;
	  continue;
	}
	
	if(output.subvertex(zstdev, zmean.size(), zstdev.size()) == false){
	  failure = true;
	  continue;
	}
	
	inv_zstdev.resize(zstdev.size());
	
	for(unsigned int i=0;i<inv_zstdev.size();i++){
	  zstdev[i] = abs(zstdev[i]);
	  inv_zstdev[i] = T(1.0)/(zstdev[i] + T(0.001));
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
	
	auto zsum = zstdev - inv_zstdev;
	
	auto g1 = T(-1.0)*zmean*Jmean + T(-1.0)*zsum*Jstdev;
	
	// 2. second calculate gradient
	math::vertex<T> decoder_gradient, encoder_gradient;
	decoder_gradient.resize(decoder.gradient_size());
	encoder_gradient.resize(encoder.gradient_size());
	decoder_gradient.zero();
	encoder_gradient.zero();
	
	const unsigned int N = 30;
	
	math::vertex<T> epsilon;
	epsilon.resize(zmean.size());
	
	
	// printf("CALCULATING 2ND GRADIENT\n");
	
	for(unsigned int j=0;j<N;j++){
	  // epsilon is ~ Normal(0,I)
	  rng.normal(epsilon);
	  
	  auto zi = zmean;
	  for(unsigned int k=0;k<zmean.size();k++){
	    zi[k] += zstdev[k]*epsilon[k];
	  }
	  
	  auto xmean = x;
	  decoder.calculate(zi, xmean);
	  
	  math::matrix<T> grad_meanx;
	  decoder.gradient(zi, grad_meanx);
	  
	  decoder_gradient += T(1.0/N)*(x - xmean)*grad_meanx;
	  
	  math::matrix<T> J_meanx_value;
	  decoder.gradient_value(zi, J_meanx_value);
	  auto gzx = T(1.0/N)*(x - xmean)*J_meanx_value; // first part of gradient
	  
	  auto Jstdev_epsilon = Jstdev;
	  for(unsigned int y=0;y<Jstdev.ysize();y++)
	    for(unsigned int x=0;x<Jstdev.xsize();x++)
	      Jstdev_epsilon(y,x) = epsilon[y]*Jstdev(y,x);
	  
	  // encoder_gradient += gzx * (Jmean + epsilon*Jstdev); // second part of gradient
	  encoder_gradient += gzx * (Jmean + Jstdev_epsilon); // second part of gradient
	}
	
	// printf("CALCULATING 2ND GRADIENT: SUM CALCULATED\n");
	
	// calculates gradient
	auto pg = pgradient;
	pg.zero();
	pg.write_subvertex(g1, 0);
	pgrad += pg;
	
	pg.zero();
	pg.write_subvertex(encoder_gradient, 0);
	pg.write_subvertex(decoder_gradient, encoder_gradient.size());
	pgrad += pg;
      }
      

#pragma omp critical
      if(!failure){
	pgradient += pgrad;
      }

    }

    if(failure) return false;
    
    pgradient /= T(xsamples.size());

    pgradient /= pgradient.norm(); // is this really needed?
    
    return true;
  }
  
  
  
  template class VAE< float >;
  template class VAE< double >;  
  template class VAE< math::blas_real<float> >;
  template class VAE< math::blas_real<double> >;
}
