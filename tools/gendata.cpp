// simple program creating machine learning example data


#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <time.h>
#include <math.h>

// generates examples used on machine learning
void generate(std::vector<float>& x){

  float max = -INFINITY;

  for(unsigned int i=0;i<x.size();i++){
    x[i] = ((float)rand())/((float)RAND_MAX);
    if(i != x.size()-1) if(x[i] > max) max = x[i];
  }

  x[x.size()-1] = max;
}



int main(int argc, char** argv)
{
  if(argc != 2) return -1;
  
  const unsigned int dimension = atoi(argv[1]);

  // generates data sets
  srand(time(0));
  
  FILE* handle1 = fopen("gendata_training.csv", "wt");
  FILE* handle2 = fopen("gendata_scoring.csv", "wt");
  FILE* handle3 = fopen("gendata_scoring_correct.csv", "wt");

  for(unsigned int i=0;i<10000;i++){
    std::vector<float> example;
    example.resize(dimension);
    generate(example);

    for(unsigned int j=0;j<example.size();j++){
      fprintf(handle1, "%f ", example[j]);
    }
    fprintf(handle1, "\n");

    example.resize(dimension);
    generate(example);

    for(unsigned int j=0;j<example.size();j++){
      if(j != example.size()-1)
	fprintf(handle2, "%f ", example[j]);
      else
	fprintf(handle3, "%f ", example[j]);
    }
    fprintf(handle2, "\n");
    fprintf(handle3, "\n");
  }

  fclose(handle1);
  fclose(handle2);
  fclose(handle3);
  
  return 0;
}

