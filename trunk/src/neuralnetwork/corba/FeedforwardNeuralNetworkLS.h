
#ifndef FeedforwardNeuralNetworkLS_h
#define FeedforwardNeuralNetworkLS_h

#include "LearningSystem-cpp-common.h"
#include <algos/algos.h>


namespace whiteice
{
  
  class FeedforwardNeuralNetworkLS : public POA_whiteice::LearningSystem
  {
  public:
    FeedforwardNeuralNetworkLS();
    ~FeedforwardNeuralNetworkLS();
    
    CORBA::Boolean reset(CORBA::ULong input_dimensions,
			 CORBA::ULong output_dimensions)
      throw(CORBA::SystemException);
    
    CORBA::Boolean load(char const *resourcename)
      throw(CORBA::SystemException);
    
    CORBA::Boolean save(char const *resourcename)
      throw(CORBA::SystemException);
    
    CORBA::Boolean add(const ::whiteice::dataset &examples)
      throw(CORBA::SystemException);
    
    CORBA::Boolean estimate(const ::whiteice::vertex &input,
			    CORBA::ULong outputsize,
			    ::whiteice::vertexlist_out output)
      throw(CORBA::SystemException);
    
    
  private:
    
    
    
  };
  
};


#endif // FeedforwardNeuralNetworkLS_h

