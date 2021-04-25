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


whiteice::math::blas_real<float> calculateError(const whiteice::nnetwork<>& policy,
						const whiteice::nnetwork<>& Q,
						const whiteice::dataset<>& data);


int main(int argc, char** argv)
{
  const bool RESIDUAL = false;
  
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
      auto delta = r1 + r2;
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

#if 0
    arch.push_back(6);
    arch.push_back(128);
    arch.push_back(400);
    arch.push_back(200);
    arch.push_back(128);
    //arch.push_back(50);
    arch.push_back(1);
#else
    arch.push_back(6);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(50);
    arch.push_back(1);
#endif
    
    nn.setArchitecture(arch);
    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<>::rectifier);
    nn.randomize();
    nn.setResidual(RESIDUAL);

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
    arch.push_back(200);
    arch.push_back(200);
    //arch.push_back(50);
    //arch.push_back(50);
    //arch.push_back(50);
    arch.push_back(3);

    nn.setArchitecture(arch);
    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<>::tanh);
    nn.randomize();
    nn.setResidual(RESIDUAL);

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
    const unsigned int NUM_DATAPOINTS = 128;

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
    arch.push_back(128);
    arch.push_back(200);
    //arch.push_back(50);
    //arch.push_back(10000); //
    //arch.push_back(50);
    //arch.push_back(10000);
    arch.push_back(3);

    nn.setArchitecture(arch);
    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<>::pureLinear);
    nn.randomize();
    nn.setResidual(RESIDUAL);

    unsigned int iterations = 0;
    bool first_time = true;
    whiteice::math::blas_real<float> start_value = 0.0f;

    //while(iterations < 1000){
    while(1){

      // calculate gradient for policy in Q(x,policy(x))
      const unsigned int GRADSAMPLES = 1000;
      
      whiteice::math::vertex<> sumgrad, grad, meanq;
      grad.resize(nn.exportdatasize());
      sumgrad.resize(nn.exportdatasize());
      meanq.resize(Q.output_size());
      sumgrad.zero();
      grad.zero();
      meanq.zero();
      
      for(unsigned int i=0;i<GRADSAMPLES;i++){
	//const unsigned int index = random.rand() % data.size(0);
	const unsigned int index = i % data.size(0);
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

#if 1
	whiteice::math::blas_real<float> error_sign = 1.0;
	
	if(Qvalue[0] > 10.0)
	  error_sign = -1.0; // was -1.0
	else
	  error_sign = +1.0; // was +1.0
#endif
	
	assert(g.xsize() == nn.exportdatasize());

	for(unsigned int j=0;j<nn.exportdatasize();j++){
	  // grad[i] = g(0, i);
	  grad[i] = error_sign*g(0, i);
	}

	sumgrad += grad;
      }

      sumgrad /= GRADSAMPLES;
      meanq /= GRADSAMPLES;

      if(first_time){
	first_time = false;
	start_value = meanq[0];
      }

      
      // police ascend Q(x,policy(x)) value
      {
	whiteice::math::blas_real<float> lrate = 1.0; // learning rate

	auto init_value = meanq[0];
	auto cur_value = meanq[0];
	whiteice::nnetwork<> nn2(nn);

	do{
	  whiteice::math::vertex<> weights;
	  
	  nn.exportdata(weights);
	  weights += lrate*sumgrad;
	  nn2.importdata(weights);

	  cur_value = calculateError(nn2, Q, data);

	  lrate *= 0.5;
	}
	while(cur_value <= init_value && lrate > 1e-20);
	
	meanq[0] = cur_value;
	nn = nn2;
      }

      // mean Q value given target policy
      {
	const unsigned int index = 0 % data.size(0);
	auto state = data.access(0, index);
	whiteice::math::vertex<> action;
	nn.calculate(state, action);
	
	std::cout << "ITER "
		  << iterations
		  << ". Average Q value given policy: " << meanq
		  << " (start: " << start_value << ")"
		  << " f(" << state << ") = " << action << " "
		  << std::endl;
      }
      

      iterations++;
    }
    
  }

  
}


whiteice::math::blas_real<float> calculateError(const whiteice::nnetwork<>& policy,
						const whiteice::nnetwork<>& Q,
						const whiteice::dataset<>& data)
{
  whiteice::math::blas_real<float> meanq = 0.0;
  
  for(unsigned int i=0;i<data.size(0);i++){
    //const unsigned int index = random.rand() % data.size(0);
    const unsigned int index = i % data.size(0);
    auto state = data.access(0, index);
	
    whiteice::math::vertex<> action;
    policy.calculate(state, action);

    whiteice::math::vertex<> in(state.size() + action.size());
    
    assert(in.write_subvertex(state, 0) == true);
    assert(in.write_subvertex(action, state.size()) == true);

    whiteice::math::vertex<> Qvalue;

    Q.calculate(in, Qvalue);

    meanq += Qvalue[0];
  }

  meanq /= data.size(0);

  return meanq;
}
