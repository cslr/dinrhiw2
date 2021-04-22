/*
 * Testsuite 4 A
 * Tests addition neural network learning works
 *
 * Q and Policy learnign from data works (NNGradDescent)
 *
 * calculation of Gradient Grad(Q(x,policy(x))) don't work
 * and maximizing policy(x) to give max Q(x) value don't work
 * 
 */


#include "nnetwork.h"
#include "NNGradDescent.h"
#include "dataset.h"
#include "RNG.h"

int main(int argc, char** argv)
{
#if 1
  // creates Q dataset for learning
  whiteice::nnetwork<> Q;
  
  {
    std::cout << "Q learning for AdditionProblem" << std::endl;
    
    whiteice::dataset<> data;
    const unsigned int NUM_DATAPOINTS = 1000;

    whiteice::RNG<> random;

    data.createCluster("input", 6);
    data.createCluster("output", 1);

    for(unsigned int i=0;i<NUM_DATAPOINTS;i++){
      whiteice::math::vertex<> in, out;
      whiteice::math::vertex<> r1, r2;

      r1.resize(3);
      r2.resize(3);

      random.normal(r1);
      random.normal(r2);

      in.resize(6);
      assert(in.write_subvertex(r1, 0) == true);
      assert(in.write_subvertex(r2, r1.size()) == true);

      out.resize(1);
      auto delta = r1 - r2;
      whiteice::math::blas_real<float> MAX = 5.0f;
      auto value = delta.norm();
      if(value > MAX) value = 0.0f;
      else value = MAX - value;
      out[0] = value;
      
      data.add(0, in);
      data.add(1, out);
    }
    
    whiteice::nnetwork<> nn;
    std::vector<unsigned int> arch;
    whiteice::math::NNGradDescent<> grad;

    arch.push_back(6);
    arch.push_back(51);
    arch.push_back(52);
    arch.push_back(53);
    arch.push_back(54);
    arch.push_back(55);
    arch.push_back(1);

    nn.setArchitecture(arch);
    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<>::rectifier);
    nn.randomize();
    nn.setResidual(false);

    assert(grad.startOptimize(data, nn, 1, 250, false, true) == true);

    while(grad.isRunning()){
      whiteice::math::blas_real<float> error = 0.0f;
      unsigned int iterations = 0;
      
      grad.getSolution(nn, error, iterations);

      std::cout << "ITER " << iterations << ". ERROR: " << error << std::endl;

      sleep(1);
    }

    {
      grad.stopComputation();
      whiteice::math::blas_real<float> error = 0.0f;
      unsigned int iterations = 0;
      
      grad.getSolution(Q, error, iterations);
      
    }
    
  }
#endif

#if 0
  // policy learning for optimal action, policy(s) = -s, because Q(s,a) = s+a
  {
    std::cout << "policy learning for AdditionProblem" << std::endl;

    whiteice::dataset<> data;
    const unsigned int NUM_DATAPOINTS = 1000;

    whiteice::RNG<> random;

    data.createCluster("input", 3);
    data.createCluster("output", 3);

    for(unsigned int i=0;i<NUM_DATAPOINTS;i++){
      whiteice::math::vertex<> in, out;
      whiteice::math::vertex<> r1, r2;

      r1.resize(3);
      r2.resize(3);

      random.normal(r1);
      r2 = -r1;

      in = r1;
      out = r2;
      
      data.add(0, in);
      data.add(1, out);
    }
    
    whiteice::nnetwork<> nn;
    std::vector<unsigned int> arch;
    whiteice::math::NNGradDescent<> grad;

    arch.push_back(3);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(3);

    nn.setArchitecture(arch);
    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<>::tanh);
    nn.randomize();
    nn.setResidual(false);

    assert(grad.startOptimize(data, nn, 1, 250, false, true) == true);

    while(grad.isRunning()){
      whiteice::math::blas_real<float> error = 0.0f;
      unsigned int iterations = 0;
      
      grad.getSolution(nn, error, iterations);

      std::cout << "ITER " << iterations << ". ERROR: " << error << std::endl;

      sleep(1);
    }
    
  }
#endif


  // policy learning from learnt Q function max policy: Q(in, policy(in))
  {
    std::cout << "policy learning for AdditionProblem" << std::endl;

    whiteice::dataset<> data;
    const unsigned int NUM_DATAPOINTS = 1000;

    whiteice::RNG<> random;

    data.createCluster("input", 3);

    for(unsigned int i=0;i<NUM_DATAPOINTS;i++){
      whiteice::math::vertex<> in;
      whiteice::math::vertex<> r1;

      r1.resize(3);

      random.normal(r1);

      in = r1;
      data.add(0, in);
    }
    
    whiteice::nnetwork<> nn;
    std::vector<unsigned int> arch;
    whiteice::math::NNGradDescent<> grad;

    arch.push_back(3);
    arch.push_back(50);
    arch.push_back(51);
    arch.push_back(52);
    arch.push_back(53);
    arch.push_back(54);
    arch.push_back(3);

    nn.setArchitecture(arch);
    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<>::tanh);
    nn.randomize();
    nn.setResidual(false);

    unsigned int iterations = 0;

    while(iterations < 1000){

      // calculate gradient for policy in Q(x,policy(x))
      const unsigned int GRADSAMPLES = 100;
      
      whiteice::math::vertex<> sumgrad, grad, meanq;
      grad.resize(nn.exportdatasize());
      sumgrad.resize(nn.exportdatasize());
      meanq.resize(Q.output_size());
      sumgrad.zero();
      grad.zero();
      meanq.zero();
      
      for(unsigned int i=0;i<GRADSAMPLES;i++){
	const unsigned int index = random.rand() % data.size(0);
	auto state = data.access(0, index);
	
	whiteice::math::vertex<> action;
	nn.calculate(state, action);

	whiteice::math::matrix<> gradP;

	nn.jacobian(state, gradP);

	whiteice::math::vertex<> in(state.size() + action.size());

	assert(in.write_subvertex(state, 0) == true);
	assert(in.write_subvertex(action, state.size()) == true);

	whiteice::math::matrix<> gradQ, full_gradQ;
	whiteice::math::vertex<> Qvalue;

	Q.calculate(in, Qvalue);
	Q.gradient_value(in, full_gradQ);

	meanq += Qvalue;

	full_gradQ.submatrix(gradQ,
			     state.size(), 0,
			     action.size(), full_gradQ.ysize());

	//std::cout << "gradQ = " << gradQ << std::endl;

	whiteice::math::matrix<> g;

	g = gradQ * gradP;

#if 0
	whiteice::math::blas_real<float> error_sign = 1.0;

	if(Qvalue[0] > 10.0)
	  error_sign = -1.0; // was -1.0
	else
	  error_sign = +1.0; // was +1.0
#endif
	
	assert(g.xsize() == nn.exportdatasize());

	for(unsigned int j=0;j<nn.exportdatasize();j++){
	  grad[j] = g(0, j);
	  // grad[i] = error_sign*g(0, i);
	}

	sumgrad += grad;
      }

      sumgrad /= GRADSAMPLES;
      meanq /= GRADSAMPLES;

      
      // police ascend Q(x,policy(x)) value
      {
	whiteice::math::blas_real<float> lrate = 0.001; // learning rate
	
	whiteice::math::vertex<> weights;
	
	nn.exportdata(weights);
	weights += lrate*weights;
	nn.importdata(weights);
	
      }

      // mean Q value given target policy
      {
	std::cout << "ITER "
		  << iterations
		  << ". Average Q value given policy: " << meanq << std::endl;
      }
      

      iterations++;
    }
    
  }

  
}
