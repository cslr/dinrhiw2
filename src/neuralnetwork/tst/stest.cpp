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

using namespace whiteice;

int main()
{
  std::cout << "superresolution test cases" << std::endl;
  
  // creates training dataset
  std::cout << "Creating training dataset.." << std::endl;
  
  dataset<
    math::superresolution< math::blas_complex<double>,
			   math::modular<unsigned int> > > data;

  data.createCluster("input", 4);
  data.createCluster("output", 4);
  
  std::vector< math::vertex<
    math::superresolution< math::blas_complex<double>,
			   math::modular<unsigned int> > > > input;

  std::vector< math::vertex<
    math::superresolution< math::blas_complex<double>,
			   math::modular<unsigned int> > > > output;

  RNG< math::blas_real<double> > prng;

  nnetwork< math::blas_real<double> > net;
  nnetwork< math::superresolution< math::blas_complex<double>,
				   math::modular<unsigned int> > > snet;

  math::vertex< math::blas_real<double> > weights;
  math::vertex< math::superresolution< math::blas_complex<double>,
				       math::modular<unsigned int> > > sweights;

  std::vector<unsigned int> arch;
  arch.push_back(4);
  arch.push_back(4);
  arch.push_back(4);

  // pureLinear non-linearity (layers are all linear)
  net.setArchitecture(arch, nnetwork< math::blas_real<double> >::rectifier);
  snet.setArchitecture(arch, nnetwork< math::superresolution<
		       math::blas_complex<double>,
		       math::modular<unsigned int> > >::rectifier);

  net.randomize();
  snet.randomize();

  std::cout << "net weights size: " << net.gradient_size() << std::endl;
  std::cout << "snet weights size: " << snet.gradient_size() << std::endl;

  net.exportdata(weights);
  snet.exportdata(sweights);

  math::convert(sweights, weights);

  snet.importdata(sweights);
  snet.exportdata(sweights);

  std::cout << "weights = " << weights << std::endl;
  std::cout << "sweights = " << sweights << std::endl;
  
  
  for(unsigned int i=0;i<10;i++){
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
      y[1] =  math::pow(a, (x[0]/x[2]) );
    else
      y[1] = -math::pow(a, (x[0]/x[2]) );

    y[2] = 0.0f;
    if(x[1].c[0] >= 0.0f) y[2] += one;
    else y[2] -= one;
    if(x[3].c[0] >= 0.0f) y[2] += one;
    else y[2] -= one;
    auto temp = math::cos(w*x[0]);
    if(temp.c[0] >= 0.0f) y[2] += one;
    else y[2] -= one;
    
    y[3] = x[1]/x[0] + x[2]*math::sqrt(x[3]) + math::abs(x[3] - x[0]);

    // std::cout << "x = " << x << std::endl;
    //std::cout << "y = " << y << std::endl;

    whiteice::math::convert(sx, x);
    whiteice::math::convert(sy, y);

    // std::cout << "sx = " << sx << std::endl;
    //std::cout << "sy = " << sy << std::endl;

    input.push_back(sx);
    output.push_back(sy);
    
    // calculates nnetwork response to y=f(sx) with net and snet which should give same results
    
    net.calculate(x, y);

    std::cout << "y  =  f(x) = " << y << std::endl;

    snet.calculate(sx, sy);

    std::cout << "sy = f(sx) = " << sy << std::endl;
  }

  data.add(0, input);
  data.add(1, output);

  data.preprocess(0);
  data.preprocess(1);

  // data preprocessed NOW

  // next DO NNGradDescent<> && nnetwork<> to actually learn the data.
    
  return 0;
}
