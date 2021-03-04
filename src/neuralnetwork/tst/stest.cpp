/* 
 * superresolution test cases
 *
 * Tomas Ukkonen
 */

#include <iostream>
#include <vector>

#include "superresolution.h"
#include "dataset.h"
#include "nnetwork.h"
#include "NNGradDescent.h"
#include "RNG.h"

#undef __STRICT_ANSI__
#include <fenv.h>

// enables floating point exceptions, these are good for debugging 
// to notice BAD floating point values that come from software bugs..
#include <fenv.h>

extern "C" {

  // traps floating point exceptions..
#define _GNU_SOURCE 1
#ifdef __linux__
#include <fenv.h>
  static void __attribute__ ((constructor))
  trapfpe(){
    feenableexcept(FE_INVALID|FE_DIVBYZERO|FE_OVERFLOW);
    //feenableexcept(FE_INVALID);
  }
#endif
  
}

using namespace whiteice;

int main()
{
  std::cout << "superresolution test cases" << std::endl;

  whiteice::logging.setPrintOutput(true);
  
  // creates training dataset
  std::cout << "Creating training dataset.." << std::endl;
  
  dataset<
    math::superresolution< math::blas_complex<double>,
			   math::modular<unsigned int> > > data;

  data.createCluster("input", 4);
  data.createCluster("output", 4);
  
  std::vector< math::vertex<
    math::superresolution< math::blas_complex<double>,
			   math::modular<unsigned int> > > > inputs;

  std::vector< math::vertex<
    math::superresolution< math::blas_complex<double>,
			   math::modular<unsigned int> > > > outputs;

  RNG< math::blas_real<double> > prng;

  nnetwork< math::blas_real<double> > net;
  nnetwork< math::superresolution< math::blas_complex<double>,
				   math::modular<unsigned int> > > snet;

  math::vertex< math::blas_real<double> > weights;
  math::vertex< math::superresolution< math::blas_complex<double>,
				       math::modular<unsigned int> > > sweights;

  std::vector<unsigned int> arch;
  arch.push_back(4);
  arch.push_back(10);
  arch.push_back(10);
  arch.push_back(4);


  // pureLinear non-linearity (layers are all linear) [pureLinear]
  net.setArchitecture(arch, nnetwork< math::blas_real<double> >::rectifier);
  snet.setArchitecture(arch, nnetwork< math::superresolution<
		       math::blas_complex<double>,
		       math::modular<unsigned int> > >::rectifier);

  net.randomize();
  snet.randomize();
  net.setResidual(false);
  snet.setResidual(false);
  

  std::cout << "net weights size: " << net.gradient_size() << std::endl;
  std::cout << "snet weights size: " << snet.gradient_size() << std::endl;

  net.exportdata(weights);
  snet.exportdata(sweights);
  
  //math::convert(sweights, weights);
  
  snet.importdata(sweights);
  snet.exportdata(sweights);

  std::cout << "weights = " << weights << std::endl;
  std::cout << "sweights = " << sweights << std::endl;
  
  
  for(unsigned int i=0;i<100;i++){
    math::vertex<
      math::superresolution< math::blas_complex<double>,
			     math::modular<unsigned int> > > sx, sy;

    math::vertex< math::blas_real<double> > x, y;

    math::blas_real<double> sigma = 4.0;
    math::blas_real<double> f = 3.14157, a = 1.1132, w = 7.342, one = 1.0;

    x.resize(4);
    prng.normal(x);
    x = sigma*x; // x ~ N(0,4^2*I)

    y.resize(4);

    y[0] = math::sin((f*x[0]*x[1]*x[2]*x[3]));
    if(x[3].c[0] >= 0.0f)
      y[1] =  math::pow(a, (x[0]/(math::abs(x[2])+one)) );
    else
      y[1] = -math::pow(a, (x[0]/(math::abs(x[2])+one)) );

    y[2] = 0.0f;
    if(x[1].c[0] >= 0.0f) y[2] += one;
    else y[2] -= one;
    if(x[3].c[0] >= 0.0f) y[2] += one;
    else y[2] -= one;
    auto temp = math::cos(w*x[0]);
    if(temp.c[0] >= 0.0f) y[2] += one;
    else y[2] -= one;
    
    y[3] = x[1]/x[0] + x[2]*math::sqrt(math::abs(x[3])) + math::abs(x[3] - x[0]);

    // std::cout << "x = " << x << std::endl;
    //std::cout << "y = " << y << std::endl;

    // net.calculate(x, y); // USE nnetwork as a target function (easier)

    whiteice::math::convert(sx, x);
    whiteice::math::convert(sy, y);

    // std::cout << "sx = " << sx << std::endl;
    //std::cout << "sy = " << sy << std::endl;

    inputs.push_back(sx);
    outputs.push_back(sy);
    
    // calculates nnetwork response to y=f(sx) with net and snet which should give same results
    
    net.calculate(x, y);

    // std::cout << "y  =  f(x) = " << y << std::endl;

    snet.calculate(sx, sy);

    // std::cout << "sy = f(sx) = " << sy << std::endl;
  }

  data.add(0, inputs);
  data.add(1, outputs);
  
  data.preprocess(0);
  //data.preprocess(1);

  // next DO NNGradDescent<> && nnetwork<> to actually learn the data.

  // gradient descent code
  {
    
    math::vertex< math::superresolution<math::blas_complex<double>,
					math::modular<unsigned int> > > grad, err, weights, wk;
    math::vertex< math::superresolution<math::blas_complex<double>,
					math::modular<unsigned int> > > sumgrad;
    
    unsigned int counter = 0;
    math::superresolution<math::blas_complex<double>,
			  math::modular<unsigned int> > error(1000.0f), last_error(1000.0f);
    math::superresolution<math::blas_complex<double>,
			  math::modular<unsigned int> > lrate(0.0010f);
    
    while(abs(error)[0].real() > math::blas_real<double>(0.001f) && counter < 100000){
      error = math::superresolution<math::blas_complex<double>,
				    math::modular<unsigned int> >(0.0f);
      sumgrad.zero();
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back

      math::superresolution<math::blas_complex<double>,
			    math::modular<unsigned int> > ninv =
	math::superresolution<math::blas_complex<double>,
			      math::modular<unsigned int> >(1.0f/data.size(0));

      for(unsigned int i=0;i<data.size(0);i++){
	snet.input() = data.access(0, i);
	snet.calculate(true);
	err = snet.output() - data.access(1,i);
	
	for(unsigned int j=0;j<err.size();j++){
	  error += ninv*err[j]*math::conj(err[j]) / data.access(0,0).size();
	}
	  
#if 1
	// this works with pureLinear non-linearity
	auto delta = err; // delta = (f(z) - y)
	math::matrix< math::superresolution<math::blas_complex<double>,
					    math::modular<unsigned int> > > DF;
	snet.jacobian(data.access(0, i), DF);
	
	auto cDF = DF;
	cDF.conj();

	// grad = delta*cDF;
	
#if 1
	grad.resize(cDF.xsize());
	grad.zero();
	
	for(unsigned int j=0;j<cDF.xsize();j++){
	  for(unsigned int k=0;k<delta[i].size();k++){
	    for(unsigned int i=0;i<delta.size();i++){
	      grad[j][k] += delta[i][k]*cDF(i,j)[k];
	    }
	  }
	}
#endif
	
	// grad = delta*DF; // [THIS DOESN'T WORK]
#else
	auto delta = err;
	
	for(unsigned int n=0;n<delta.size();n++)
	  for(unsigned int l=0;l<delta[0].size();l++)
	    if(l != k) delta[n][l] = 0.0f;
	
	if(snet.mse_gradient(delta, grad) == false) // returns: delta*conj(DF)
	  std::cout << "gradient failed." << std::endl;
	
#endif
	
	if(i == 0)
	  sumgrad = ninv*grad;
	else
	  sumgrad += ninv*grad;
      }

      sumgrad.normalize(); // normalizes gradient length

      snet.importdata(weights);

      if(snet.exportdata(weights) == false)
	std::cout << "export failed." << std::endl;

      const math::superresolution<math::blas_complex<double>,
				  math::modular<unsigned int> > alpha(1e-3f);

      
      weights -= lrate * sumgrad + (alpha*weights);
      
      if(snet.importdata(weights) == false)
	std::cout << "import failed." << std::endl;

      auto abserror = abs(error);

      for(unsigned int i=1;i<abserror.size();i++)
	abserror[0] += abserror[i];
      
      std::cout << counter << " : " << abserror[0] << std::endl;
      
      counter++;
    }
    
    std::cout << counter << " : " << abs(error) << std::endl;

    math::vertex< math::superresolution<math::blas_complex<double>,
					math::modular<unsigned int> > > params;
    snet.exportdata(params);
    std::cout << "nn solution weights = " << params << std::endl;
    
  }
  
  
    
  return 0;
}
