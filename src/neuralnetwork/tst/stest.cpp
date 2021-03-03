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

  whiteice::RNG< math::blas_complex<double> > prng;
  
  for(unsigned int i=0;i<1000;i++){
    math::vertex<
      math::superresolution< math::blas_complex<double>,
			     math::modular<unsigned int> > > sx, sy;

    math::vertex< math::blas_complex<double> > x, y;

    x.resize(4);
    prng.normal(x);    

    // input.push_back(x);

    y.resize(4);

    math::blas_complex<double> f = 3.14157, a = 1.1132, w = 7.342, one = 1.0;

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

    //std::cout << "x = " << x << std::endl;
    //std::cout << "y = " << y << std::endl;

    whiteice::math::convert(sx, x);
    whiteice::math::convert(sy, y);

    //std::cout << "sx = " << sx << std::endl;
    //std::cout << "sy = " << sy << std::endl;

    input.push_back(sx);
    output.push_back(sy);
  }

  data.add(0, input);
  data.add(1, output);

  data.preprocess(0);
  data.preprocess(1);

  // data preprocessed NOW

  // next DO NNGradDescent<> && nnetwork<> to actually learn the data.
    
  return 0;
}
