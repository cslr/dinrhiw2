
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>

#include "dinrhiw.h"
#include "RNN_RBM.h"
#include "norms.h"


void rnn_bbrbm_save_load_test();
void rnn_bbrbm_synthesis_test();
void rnn_bbrbm_test();


int main(int argc, char** argv)
{
  srand(time(0));

  rnn_bbrbm_save_load_test();
  rnn_bbrbm_synthesis_test();
  rnn_bbrbm_test();
  
  return 0;
}


void rnn_bbrbm_synthesis_test()
{
  printf("RNN-RBM SYNTHESIZATION TEST\n");

  const unsigned int DIMENSION = 11;

  whiteice::RNN_RBM<> rbm(DIMENSION, 2*DIMENSION , 2);

  {
    rbm.synthStart();
    
    std::vector< whiteice::math::vertex<> > timeserie;
    
    for(unsigned int i=0;i<100;i++){
      whiteice::math::vertex<> vnext;
      
      if(rbm.synthNext(vnext) == false){
	printf("ERROR: RNN-RBM SYNTHESIS FAILS (%d)\n", i);
      }
      
      if(vnext.size() != rbm.getVisibleDimensions()){
	printf("ERROR: visible dimensions mismatch (%d != %d)\n",
	       vnext.size(), rbm.getVisibleDimensions());
      }
      
      timeserie.push_back(vnext);
      
      if(rbm.synthSetNext(vnext) == false){
	printf("ERROR: RNN-RBM SETNEXT FAILS (%d)\n", i);
      }
    }
  }

  
  {
    rbm.synthStart();
    
    std::vector< whiteice::math::vertex<> > timeserie;
    
    for(unsigned int i=0;i<100;i++){
      std::vector< whiteice::math::vertex<> > samples;
      whiteice::math::vertex<> vnext;

      const unsigned int R = (rand() % 10) + 3;
      
      if(rbm.synthNext(R, samples) == false){
	printf("ERROR: RNN-RBM synthesis FAILS (%d)\n", i);
      }

      if(samples.size() != R){
	printf("ERROR: RNN-RBM synthesis does not generate proper number of samples\n");
      }

      for(unsigned int k=0;k<samples.size();k++){

	if(samples[k].size() != rbm.getVisibleDimensions()){
	  printf("ERROR: visible dimensions mismatch (%d != %d)\n",
		 samples[k].size(), rbm.getVisibleDimensions());
	}
      }

      vnext = samples[rand() % samples.size()];
      
      timeserie.push_back(vnext);
      
      if(rbm.synthSetNext(vnext) == false){
	printf("ERROR: RNN-RBM SETNEXT FAILS (%d)\n", i);
      }
    }
  }
  
  
  printf("RNN-RBM SYNTHESIZATION TEST.. DONE\n");
}


void rnn_bbrbm_test()
{
  printf("RECURRENT NEURAL NETWORK (RBM) OPTIMIZATION TEST\n");
  
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

  if(rbm.startOptimize(timeseries) == false){
    printf("ERROR: starting RNN-RBM optimization\n");
  }

  unsigned int iters = 0;

  
  while(rbm.isRunning()){
    whiteice::math::blas_real<float> e;
    const unsigned int old_iters = iters;

    if(rbm.getOptimizeError(iters, e)){
      if(iters != old_iters){
	printf("ITER %d. RNN-RBM ERROR: %f\n", iters, e.c[0]);
      }
    }

    sleep(1);
  }

  rbm.stopOptimize();
}



void rnn_bbrbm_save_load_test()
{
  printf("RNN_RBM SAVE()&LOAD() TESTS\n");
  
  unsigned int dimVisible = rand() % 32 + 5;
  unsigned int dimHidden  = rand() % 32 + 5;
  unsigned int dimRecurrent = rand() % 16 + 1;
  
  whiteice::RNN_RBM<> rbm(dimVisible, dimHidden, dimRecurrent);
  whiteice::RNN_RBM<> rbm2;

  if(rbm.save("test.model") == false){
    printf("ERROR: Saving RNN_RBM FAILED.\n");
  }

  if(rbm2.load("test.model") == false){
    printf("ERROR: Loading RNN_RBM FAILED.\n");
  }

  if(rbm.getVisibleDimensions() != rbm2.getVisibleDimensions()){
    printf("ERROR: visible neurons mismatch\n");
  }

  if(rbm.getHiddenDimensions() != rbm2.getHiddenDimensions()){
    printf("ERROR: hidden neurons mismatch\n");
  }

  if(rbm.getRecurrentDimensions() != rbm2.getRecurrentDimensions()){
    printf("ERROR: recurrent dimensions mismatch\n");
  }

  {
    whiteice::nnetwork<> nn1, nn2;
    whiteice::math::vertex<> rnn_params1;
    whiteice::math::vertex<> rnn_params2;

    rbm.getRNN(nn1);
    rbm2.getRNN(nn2);

    if(nn1.exportdata(rnn_params1) == false){
      printf("ERROR: exportting RNN parameters FAILED (1)\n");
    }

    if(nn2.exportdata(rnn_params2) == false){
      printf("ERROR: exportting RNN parameters FAILED (2)\n");
    }

    whiteice::math::blas_real<float> e = (rnn_params1 - rnn_params2).norm();
    e /= (float)rnn_params1.size();

    if(e > 0.01f){
      printf("ERROR: RNN parameters mismatch\n");
    }
    else{
      printf("Good. RNN weights difference within limits (%f)\n", e.c[0]);
    }
  }

  
  {
    whiteice::BBRBM<> bbrbm1, bbrbm2;

    rbm.getRBM(bbrbm1);
    rbm2.getRBM(bbrbm2);

    auto delta =
      whiteice::math::frobenius_norm( bbrbm1.getWeights() - bbrbm2.getWeights() );

    if(delta > 0.01f){
      printf("ERROR: BBRBM weight matrix mismatch\n");
    }
    else{
      printf("Good. BBRBM weights difference within limits (%f)\n", delta.c[0]);
    }
  }
  
  
  printf("RNN_RBM SAVE()&LOAD() TESTS.. DONE.\n");
}
