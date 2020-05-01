
#include <iostream>
#include <new>
#include <stdlib.h>
#include <time.h>

#include "PSO.h"
#include "test_function.h"

using namespace whiteice;





class gaussian_balls : public whiteice::optimized_function<float>
{
public:
  
  gaussian_balls(unsigned int N, unsigned int D){
    centers.resize(N);
    dim = D;
    
    for(unsigned int i=0;i<centers.size();i++){
      centers[i].resize(D);
      for(unsigned int j=0;j<D;j++)
	centers[i][j] = (float)rand()/((float)rand());
    }
    
    variance = 100000.0f/((float)D);
  }
  
  virtual ~gaussian_balls(){
  }
  
  unsigned int dimension() const  {
    return dim;
  }
  
  // calculates value of function
  float operator() (const whiteice::math::vertex<float>& x) const {
    return this->calculate(x);
  }
  
  // calculates value
  float calculate(const whiteice::math::vertex<float>& x) const {
    whiteice::math::vertex<float> r(dim);
    
    float result = 0.0f;
    
    for(unsigned int i=0;i<centers.size();i++){
      r = (centers[i] - x)/variance;
      result += whiteice::math::exp((-(r * r))[0]);
    }
    
    return result;
  }
  
  
  void calculate(const whiteice::math::vertex<float>&x, float& y) const {
    y = calculate(x);
  }
  
  
  // creates copy of object
  whiteice::function<whiteice::math::vertex<float>,float>* clone() const {
    gaussian_balls* gb = 
      new gaussian_balls(centers.size(), dim);
    
    for(unsigned int i=0;i<centers.size();i++)
      for(unsigned int j=0;j<centers[i].size();j++)
	gb->centers[i][j] = centers[i][j];
    
    gb->variance = variance;
    gb->dim = dim;
    
    return gb;
  }
  
  float goodvalue() const{
    return calculate(centers[0]);
  }
  
  
  //////////////////////////////////////////////////////////////////////
  
  bool hasGradient() const {
    return false;
  }
  
  // gets gradient at given point
  math::vertex<float> grad(math::vertex<float>& x) const{
    return x;
  }
  
  void grad(math::vertex<float>& x, math::vertex<float>& y) const{
    y = x;
  }
  
  
      
  bool hasHessian() const {
    return false;
  }
      
  // gets hessian at given point
  math::matrix<float> hessian(math::vertex<float>& x) const{
    return math::matrix<float>(1,1);
  }
  
  void hessian(math::vertex<float>& x, math::matrix<float>& y) const{
    return;
  }
  
  
  std::vector< whiteice::math::vertex<float> > centers;
  unsigned int dim;
  float variance;
};




int main(int argc, char **argv)
{   
  srand(time(0));
  
  optimized_function<float>* of = new gaussian_balls(10, 1000);
  
  PSO< float >* pso;
  pso = new PSO< float >(*of);
  math::vertex< float > input;
  
  pso->verbosity(true);
  
  pso->maximize(10000, 100);
  pso->getBest(input);
  std::cout << "maximization result "
	    << of->calculate(input) << std::endl;
  
  
  pso->minimize(10000, 100);
  pso->getBest(input);
  
  std::cout << "minimization result "
	    << of->calculate(input) << std::endl;
  
  
  std::cout << "swarm size " << pso->size() << std::endl;
  
  delete pso;
  
  return 0;
}






