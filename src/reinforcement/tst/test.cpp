
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

#include "CartPole.h"
#include "RIFL_abstract.h"



int main(int argc, char** argv)
{
  printf("REINFORCEMENT LEARNING TESTCASES\n");
  fflush(stdout);

  srand(time(0));

  if(argc <= 1){
    whiteice::CartPole< whiteice::math::blas_real<double> > system;

    system.setEpsilon(0.33); // 33% of examples are selected accoring to model
    system.load("rifl.dat");
    
    system.start();

    while(1){
      sleep(180); // saved model file every 3 minutes
      system.save("rifl.dat");
      printf("MODEL FILE SAVED\n");
    }
    
    system.stop();
    
  }
  else if(strcmp(argv[1], "use") == 0){

    whiteice::CartPole< whiteice::math::blas_real<double> > system;

    system.setEpsilon(1.00); // 100% of examples are selected accoring to model
    if(system.load("rifl.dat") == false){
      printf("ERROR: loading model file failed.\n");
      return -1;
    }
    
    system.start();
    
    sleep(3600); // 1 hour
    
    system.stop();
  }
  

  return 0;
}
