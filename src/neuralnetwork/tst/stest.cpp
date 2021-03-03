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
  
  dataset<
    math::superresolution< math::blas_complex<double>,
			   math::modular<unsigned int> > > data;

  // creates training dataset
  std::cout << "Creating training dataset.." << std::endl;

  std::vector< math::vertex<
    math::superresolution< math::blas_complex<double>,
			   math::modular<unsigned int> > > > input;

  std::vector< math::vertex<
    math::superresolution< math::blas_complex<double>,
			   math::modular<unsigned int> > > > output;

  whiteice::RNG< math::superresolution< math::blas_complex<double>,
					math::modular<unsigned int> > > prng;

  for(unsigned int i=0;i<10;i++){
    math::vertex<
      math::superresolution< math::blas_complex<double>,
			     math::modular<unsigned int> > > x, y;

    // x.resize(4);

    // prng.normal(x);

    std::cout << x << std::endl;
  }
    
    
  return 0;
}
