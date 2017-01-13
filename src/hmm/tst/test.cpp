

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

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
    // we hand set a simple two hidden state markov model emitting 3 different symbols and test if we can learn it 
    
    whiteice::HMM hmm(3, 2);

    // transition probabilities are taken from Crazy Soft Drink Machine
    // (Statistical Natural Language Processing book, page 321)
    
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
    
    assert(hmm.sample(1024, data) == true);

    
    whiteice::HMM hmm2(3, 2);
    double r = hmm2.train(data);

    printf("MODEL FIT (logp): %f\n", r);
    fflush(stdout);
    
  }

  
  
}
