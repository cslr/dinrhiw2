
#include "FeedforwardNeuralNetworkLS.h"
#include <algos/algos.h>


namespace whiteice
{
  
  FeedforwardNeuralNetworkLS::FeedforwardNeuralNetworkLS()
  {
  }
  
  
  FeedforwardNeuralNetworkLS::~FeedforwardNeuralNetworkLS()
  {
  }
  
  
  CORBA::Boolean FeedforwardNeuralNetworkLS::reset(CORBA::ULong input_dimensions,
						   CORBA::ULong output_dimensions)
    throw(CORBA::SystemException)
  {
  }
  
  
  CORBA::Boolean FeedforwardNeuralNetworkLS::load(char const *resourcename)
    throw(CORBA::SystemException)
  {
  }
  
  
  CORBA::Boolean FeedforwardNeuralNetworkLS::save(char const *resourcename)
    throw(CORBA::SystemException)
  {
  }
  
  
  CORBA::Boolean FeedforwardNeuralNetworkLS::add(const ::whiteice::dataset &examples)
    throw(CORBA::SystemException)
  {
    
  }
  
  
  CORBA::Boolean FeedforwardNeuralNetworkLS::estimate(const ::whiteice::vertex &input,
						      CORBA::ULong outputsize,
						      ::whiteice::vertexlist_out output)
    throw(CORBA::SystemException)
  {
  }
  
  
};
