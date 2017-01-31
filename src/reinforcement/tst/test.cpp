
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "CartPole.h"
#include "RIFL_abstract.h"



int main(int argc, char** argv)
{
  printf("REINFORCEMENT LEARNING TESTCASES\n");
  fflush(stdout);

  srand(time(0));

  whiteice::CartPole< whiteice::math::blas_real<double> > system;

  system.start();

  while(1) sleep(1);

  system.stop();
  

  return 0;
}
