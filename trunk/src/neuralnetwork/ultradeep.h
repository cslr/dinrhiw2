/*
 * learns data using ultra-deep neural networks
 */

#ifndef __ultradeep_neuralnetwork__
#define __ultradeep_neuralnetwork__

#include "vertex.h"
#include "matrix.h"

#include <vector>


namespace whiteice
{
  struct ultradeep_parameters
  {
    math::vertex<> d;
    math::vertex<> b;
  };


  bool ultradeep(std::vector< math::vertex<> > input,
		 std::vector< ultradeep_parameters >& params,
		 const std::vector< math::vertex<> >& output);
  
  
};


#endif
