

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>

#include "HMM.h"

#include <vector>



void hmm_test();


int main(int argc, char** argv)
{
  srand(time(0));
  
  hmm_test();
}


void hmm_test()
{
  printf("HIDDEN MARKOV MODEL (HMM) TESTS\n");

  {
    printf("HMM - LEARNING TEST\n");
    
    // we hand set a simple two hidden state markov model emitting 3 different symbols and test if we can learn it 
    
    whiteice::HMM hmm(3, 2);

    // transition probabilities are taken from Crazy Soft Drink Machine
    // (Statistical Natural Language Processing book, page 321)

    //std::cout << "A size: " << hmm.getA().size() << std::endl;
    //std::cout << "A[0] size: " << hmm.getA()[0].size() << std::endl;
    
    // state transition probabilties
    hmm.getA()[0][0] = 0.7;
    hmm.getA()[0][1] = 0.3;
    hmm.getA()[1][0] = 0.5;
    hmm.getA()[1][1] = 0.5;

    // symbol emitting probabilities
    hmm.getB()[0][0][0] = 0.6;
    hmm.getB()[0][0][1] = 0.1;
    hmm.getB()[0][0][2] = 0.3;
    hmm.getB()[0][1][0] = 0.6;
    hmm.getB()[0][1][1] = 0.1;
    hmm.getB()[0][1][2] = 0.3;

    hmm.getB()[1][0][0] = 0.1;
    hmm.getB()[1][0][1] = 0.7;
    hmm.getB()[1][0][2] = 0.2;
    hmm.getB()[1][1][0] = 0.1;
    hmm.getB()[1][1][1] = 0.7;
    hmm.getB()[1][1][2] = 0.2;

    // initial state
    hmm.getPI()[0] = 1.0;
    hmm.getPI()[1] = 0.0;

    
    // generates "string" of observations from hand-set HMM
    std::vector<unsigned int> data;
    
    assert(hmm.sample(10000, data) == true);

    
    whiteice::HMM hmm2(3, 2);

    hmm2.startTrain(data, 1000, true, 0.0001);

    while(hmm2.isRunning()){
      sleep(1);
    }

    hmm2.stopTrain();
    double r = hmm2.getSolutionGoodness();

    printf("MODEL FIT (logp): %f\n", r);
    fflush(stdout);

    printf("NEW PARAMETERS\n");
    {
      unsigned int i = 0;
      unsigned int j = 0;
      unsigned int k = 0;
      
      auto pi = hmm2.getPI();
      printf("PI\n");
      for(auto& p : pi){
	printf("[%d] %f ", i, p.getDouble());
	i++;
      }
      printf("\n\n");

      i = 0;
      j = 0;
      k = 0;
      
      auto A = hmm2.getA();
      
      printf("A\n");
      for(auto& ai : A){
	j = 0;
	for(auto& aij : ai){
	  printf("[%d,%d] %f ", i, j, aij.getDouble());
	  j++;
	}
	printf("\n");
	i++;
      }
      printf("\n");

      i = 0;
      j = 0;
      k = 0;

      auto B = hmm2.getB();
      
      printf("B\n");
      for(auto& bi : B){
	j = 0;
	for(auto& bij : bi){
	  k = 0;
	  for(auto& bijo : bij){
	    printf("[%d,%d,%d] %f ", i,j,k, bijo.getDouble());
	    k++;
	  }
	  printf("\n");
	  j++;
	}
	printf("\n");
	i++;
      }
      printf("\n");
      
    }

    
    printf("REFERENCE PARAMETERS\n");
        {
      unsigned int i = 0;
      unsigned int j = 0;
      unsigned int k = 0;
      
      auto pi = hmm.getPI();
      printf("PI\n");
      for(auto& p : pi){
	printf("[%d] %f ", i, p.getDouble());
	i++;
      }
      printf("\n\n");

      i = 0;
      j = 0;
      k = 0;
      
      auto A = hmm.getA();
      
      printf("A\n");
      for(auto& ai : A){
	j = 0;
	for(auto& aij : ai){
	  printf("[%d,%d] %f ", i, j, aij.getDouble());
	  j++;
	}
	printf("\n");
	i++;
      }
      printf("\n");

      i = 0;
      j = 0;
      k = 0;

      auto B = hmm.getB();
      
      printf("B\n");
      for(auto& bi : B){
	j = 0;
	for(auto& bij : bi){
	  k = 0;
	  for(auto& bijo : bij){
	    printf("[%d,%d,%d] %f ", i,j,k, bijo.getDouble());
	    k++;
	  }
	  printf("\n");
	  j++;
	}
	printf("\n");
	i++;
      }
      printf("\n");
      
    }
	
    
  }

  {
    printf("HMM - RANDOM LARGE HMM TEST\n");
    
    const unsigned int V = rand() % 64 + 50;
    const unsigned int H = rand() % 16 + 10;

    std::cout << "HMM visible values: " << V << std::endl;
    std::cout << "HMM hidden values: " << H << std::endl;
    
    whiteice::HMM hmm(V, H);
    whiteice::HMM hmm2(V, H);

    std::vector<unsigned int> data;

    assert(hmm.sample(1000, data) == true);

    hmm2.startTrain(data, 1000, true, 0.0001);

    while(hmm2.isRunning()){
      sleep(1);
    }

    hmm2.stopTrain();
    double r = hmm2.getSolutionGoodness();
    
    std::cout  << "HMM solution plog = " << r << std::endl;
  }
  

  {
    printf("HMM - SAVE() & LOAD() TEST\n");

    unsigned int v = rand() % 64 + 10;
    unsigned int h = rand() % 64 + 10;

    whiteice::HMM hmm1(v, h);
    whiteice::HMM hmm2(v + 1, h + 2);

    hmm1.randomize();

    assert(hmm1.save("hmm1.dat") == true);
    assert(hmm2.load("hmm1.dat") == true);

    // compares parameters
    {
      auto pi1 = hmm1.getPI();
      auto pi2 = hmm2.getPI();
      
      whiteice::math::realnumber error(0.0, 128);

      for(unsigned int i=0;i<pi1.size();i++)
	error += abs(pi1[i] - pi2[i]);

      error /= ((double)pi1.size());

      if(error.getDouble() > 0.001){
	printf("ERROR: pi parameter mismatch after save()&load()\n");
	return;
      }

      auto A1 = hmm1.getA();
      auto A2 = hmm2.getA();

      error = 0.0;

      for(unsigned int i=0;i<A1.size();i++){
	for(unsigned int j=0;j<A1[i].size();j++){
	  error += abs(A1[i][j] - A2[i][j]);
	}
      }

      error /= ((double)(A1.size()*A1[0].size()));
      
      if(error.getDouble() > 0.001){
	printf("ERROR: A parameter mismatch after save()&load(): %f\n",
	       error.getDouble());
	return;
      }

      
      auto B1 = hmm1.getB();
      auto B2 = hmm2.getB();

      error = 0.0;

      for(unsigned int i=0;i<B1.size();i++){
	for(unsigned int j=0;j<B1[i].size();j++){
	  for(unsigned int k=0;k<B1[i][j].size();k++){
	    error += abs(B1[i][j][k] - B2[i][j][k]);
	  }
	}
      }

      error /= ((double)(B1.size()*B1[0].size()*B1[0][0].size()));
      
      if(error.getDouble() > 0.001){
	printf("ERROR: B parameter mismatch after save()&load(): %f\n",
	       error.getDouble());
	return;
      }
      
    }
    
    printf("HMM LOAD() & SAVE() TESTS PASSED\n");
  }



  {
    printf("HMM - SAVE() & LOAD() ARBITRARY PRECISION TEST\n");

    unsigned int v = rand() % 64 + 10;
    unsigned int h = rand() % 64 + 10;

    whiteice::HMM hmm1(v, h);
    whiteice::HMM hmm2(v + 1, h + 2);

    hmm1.randomize();

    assert(hmm1.saveArbitrary("hmm2.dat") == true);
    assert(hmm2.loadArbitrary("hmm2.dat") == true);

    // compares parameters
    {
      auto pi1 = hmm1.getPI();
      auto pi2 = hmm2.getPI();
      
      whiteice::math::realnumber error(0.0, 128);

      for(unsigned int i=0;i<pi1.size();i++)
	error += abs(pi1[i] - pi2[i]);

      error /= ((double)pi1.size());

      if(error.getDouble() > 0.001){
	printf("ERROR: pi parameter mismatch after save()&load()\n");
	return;
      }

      auto A1 = hmm1.getA();
      auto A2 = hmm2.getA();

      error = 0.0;

      for(unsigned int i=0;i<A1.size();i++){
	for(unsigned int j=0;j<A1[i].size();j++){
	  error += abs(A1[i][j] - A2[i][j]);
	}
      }

      error /= ((double)(A1.size()*A1[0].size()));
      
      if(error.getDouble() > 0.001){
	printf("ERROR: A parameter mismatch after save()&load(): %f\n",
	       error.getDouble());
	return;
      }

      
      auto B1 = hmm1.getB();
      auto B2 = hmm2.getB();

      error = 0.0;

      for(unsigned int i=0;i<B1.size();i++){
	for(unsigned int j=0;j<B1[i].size();j++){
	  for(unsigned int k=0;k<B1[i][j].size();k++){
	    error += abs(B1[i][j][k] - B2[i][j][k]);
	  }
	}
      }

      error /= ((double)(B1.size()*B1[0].size()*B1[0][0].size()));
      
      if(error.getDouble() > 0.001){
	printf("ERROR: B parameter mismatch after save()&load(): %f\n",
	       error.getDouble());
	return;
      }
      
    }
    
    printf("HMM LOAD() & SAVE() ARBITRARY TESTS PASSED\n");
  }

  {
    printf("REALNUMBER RANDOM NUMBER CHECK\n");
    
    whiteice::math::realnumber r;
    for(unsigned int i=0;i<10;i++)
      std::cout << r.random() << std::endl;
    
  }
  
}
