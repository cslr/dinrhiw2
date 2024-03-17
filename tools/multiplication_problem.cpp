// simple file to generate multiplication problem for the neural network
// the problem is has three inputs and one output f(x,y,z) = x*y*z
// and the task of the neural network is to learn to multiply
// three different numbers it has been given


#include <iostream>
#include <fstream>
#include <chrono>
#include <random>

int main(int argc, char** argv)
{
  if(argc != 2) return -1;

  unsigned int N = atoi(argv[1]);

  if(N == 0 || N > 1000000)
    return -1; // bad number

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 1.0);
  
  std::ofstream input, output;
  
  input.open("numbers.in");
  output.open("numbers.out");

  for(unsigned int n=0;n<N;n++){
    double a = distribution(generator);
    double b = distribution(generator);
    double c = distribution(generator);

    input  << a << " " << b << " " << c << std::endl;
    output << (a*b*c) << std::endl;
  }

  input.close();
  output.close();

  return 0;
}
