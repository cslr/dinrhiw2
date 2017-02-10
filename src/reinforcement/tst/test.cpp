
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

    system.setEpsilon(0.50); // 50% of examples are selected accoring to model
    system.setLearningMode(true);
    
    // system.load("rifl.dat");
    
    system.start();

    sleep(1);

    unsigned int counter = 1;

    while(system.physicsIsRunning()){
      sleep(1); 
      if((counter % 180) == 0){ // saved model file every 3 minutes
	if(system.save("rifl.dat"))
	  printf("MODEL FILE SAVED\n");
      }
      
      counter++;
    }

    system.stop();
    
  }
  else if(strcmp(argv[1], "use") == 0){

    whiteice::CartPole< whiteice::math::blas_real<double> > system;

    system.setEpsilon(1.00); // 100% of examples are selected accoring to model
    system.setLearningMode(false);
    system.setHasModel(true);
    
    if(system.load("rifl.dat") == false){
      printf("ERROR: loading model file failed.\n");
      return -1;
    }
    
    system.start();

    while(system.physicsIsRunning())
      sleep(1);
    
    system.stop();
  }
  

  return 0;
}
