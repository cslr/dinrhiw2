
#include <stdio.h>
#include <iostream>

#include "dinrhiw.h"
#include "RNN_RBM.h"


void rnn_bbrbm_test();


int main(int argc, char** argv)
{
  printf("RECURRENT NEURAL NETWORK (RBM) TESTS\n");

  rnn_bbrbm_test();
  
  return 0;
}


void rnn_bbrbm_test()
{
  const int DIMENSION = 11;
  
  whiteice::RNN_RBM<> rbm(DIMENSION, 10, 2);

  // test problem: generates 1d ping-pong timeseries where "1" bounces
  // from in 1-dimensional vector
  std::vector< std::vector< whiteice::math::vertex<> > > timeseries;
  
  {
    timeseries.resize(1);

    // ping-pongs counter mark between [0..DIMENSION[
    int counter = 0;
    int delta = 1;
    
    for(unsigned int i=0;i<1000;i++){
      whiteice::math::vertex<> v;
      v.resize(DIMENSION);
      v.zero();
      v[counter] = 1.0;

      timeseries[0].push_back(v);

      if(counter >= DIMENSION-1 && delta > 0){
	delta = -delta;
      }
      if(counter <= 0 && delta < 0){
	delta = -delta;
      }

      counter += delta;
    }
    
  }

  if(rbm.optimize(timeseries) == false)
    printf("ERROR: optimizing RNN-RBM failed.\n");
  
}
