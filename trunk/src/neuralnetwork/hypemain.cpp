
#include "hypervolume.h"
// #include "hypercircle.h" (lost?)
#include <exception>
#include <iostream>

using namespace whiteice;

void bayesian_inference_test();

int main(unsigned int argc, char **argv)
{
  bayesian_inference_test()
  
  // hypervolume test
  try{
    
    hypervolume<float>* cube;
    cube = new hypervolume<float>(3, 10);

    for(unsigned int i=0;i<10;i++){
      for(unsigned int j=0;j<10;j++){
	for(unsigned int k=0;k<10;k++){
	  if(k == i && i == j) 
	    (*cube)[i][j][k].value() = 1.0;
	  else
	    (*cube)[i][j][k].value() = 0.0;
	}
      }
    }

    std::cout << "cube(2,2,2) = " << (*cube)[2][2][2].value()
	      << " (should be 1)" << std::endl;
    
    std::cout << "cube(2,1,2) = " << (*cube)[2][1][2].value()      
	      << " (should be 0)" << std::endl;
    
    std::cout << "cube(1,1,0) = " << (*cube)[1][1][0].value()
	      << " (should be 0)" << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "unhandled exception: " << e.what() << std::endl;
    return -1;
  }


#if 0
  // hypercircle test
    
  hypercircle<float>* circle;
  circle = new hypercircle<float>(2, 5); // 2d circle with radius 5
  delete circle;
#endif

  return 0;
}




void bayesian_inference_test()
{
  // non-parameterized bayesian inference
  
  // number of divisions between probability interval
  const unsigned int N = 8; // N fuzzy and probility levels
  
  
  hypervolume<float>* probability; // of parameters M
  std::vector<float> data; // input data
  
  
  // probability of fuzzy *conditional* distribution
  // let M be model values
  // p(D|M) = p[M](D) (probability with parameters M) (*probability)[data]
  // p(M) = can be chosen at the beginning
  // 
  // from above -> p(M|D) ~ P(D|M)*P(M)

  // M = probability of (f1, p1, f2, p2, f(x1 op x2), p(x1 op x2), probability) -> 7 dims
  // D = (f1, p1, f2, p2, f(x1 op x2), p(x1 op x2))
  // N**6 = (2**3)**7 = 2**21 = 2 MB*sizeof(float), not too bad
  probability = new hypervolume<float>(6, N);
  data.resize(6);
  
  // bayesian inference
  {
    // M = z | x,y  discreted distribution values (parameters)
    // p(M | data) ~ p(data | M) * p(M)
    
    hypervoluem<float>* distribution = new hypervolume<float>(probability);
    
    std::vector<unsigned int> iter;
    iter.resize(6);
    bool quit = false;
    
    while(!quit){
      
      (*probility)[data] * (*probability)(
    }
  }
  

}


