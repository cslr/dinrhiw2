/*
 * simple tests
 */

#include "HC.h"

#include "neuralnetwork.h"
#include "backpropagation.h"
#include "neuronlayer.h"
#include "neuron.h"
#include "activation_function.h"
#include "odd_sigmoid.h"

#include "nnetwork.h"
#include "GDALogic.h"

#include "dataset.h"
#include "nnPSO.h"
#include "atlas.h"

#include "bayesian_nnetwork.h"
#include "HMC.h"

#include "HMC_gaussian.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <new>


#include <assert.h>
#include <string.h>


using namespace whiteice;


void activation_test();

void neuron_test();
void neuronlayer_test();
void neuronlayer_test2();
void neuralnetwork_test();

void nnetwork_test();
void bayesian_nnetwork_test();
void backprop_test(const unsigned int size);
void neuralnetwork_saveload_test();
void neuralnetwork_pso_test();

void hmc_test();

void gda_clustering_test();

void simple_dataset_test();

void compressed_neuralnetwork_test();


int main()
{
  unsigned int seed = (unsigned int)time(0);
  printf("seed = %x\n", seed);
  srand(seed);
  

  try{
    nnetwork_test();
    bayesian_nnetwork_test();

    hmc_test();

    /*
    simple_dataset_test();
    backprop_test(500);
    
    neuron_test();
    neuronlayer_test();
    neuronlayer_test2();
    neuralnetwork_test();
    */
#if 0
    neuralnetwork_pso_test();
    // neuralnetwork_saveload_test();
    // compressed_neuralnetwork_test();
    
    activation_test();  
    
    gda_clustering_test();
#endif
    
    return 0;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
    return -1;
  }    
}


/************************************************************/


class test_exception : public std::exception
{
public:
  
  test_exception() throw()
  {
    reason = 0;
  }
  
  test_exception(const std::exception& e) throw()
  {
    reason = 0;
    
    const char* ptr = e.what();
    
    if(ptr)
      reason = (char*)malloc(sizeof(char)*(strlen(ptr) + 1));
    
    if(reason) strcpy(reason, ptr);
  }
  
  
  test_exception(const char* ptr) throw()
  {
    reason = 0;
    
    if(ptr){
      reason = (char*)malloc(sizeof(char)*(strlen(ptr) + 1));
      
      if(reason) strcpy(reason, ptr);
    }
  }
  
  
  virtual ~test_exception() throw()
  {
    if(reason) free(reason);
    reason = 0;
  }
  
  virtual const char* what() const throw()
  {
    if(reason == 0) // null terminated value
      return ((const char*)&reason);

    return reason;
  }
  
private:
  
  char* reason;
  
};


/************************************************************/

void hmc_test()
{
  std::cout << "HMC SAMPLING TEST (Normal distribution)" << std::endl;

  {
    HMC_gaussian<> sampler(2);
    
    time_t start_time = time(0);
    unsigned int counter = 0;

    sampler.startSampler();

    // samples for 5 seconds
    while(counter < 5){
      std::cout << "sampling.. samples = "
		<< sampler.getNumberOfSamples() << std::endl;
      fflush(stdout);
      
      sleep(1);
      counter = (unsigned int)(time(0) - start_time);
    }

    sampler.stopSampler();

    std::cout << sampler.getNumberOfSamples()
	      << " samples." << std::endl;

    std::cout << "mean = " << sampler.getMean() << std::endl;
    std::cout << "cov  = " << sampler.getCovariance() << std::endl;

    std::cout << "Should be zero mean and unit I variance: N(0,I)"
	      << std::endl;

    std::cout << "Saving samples to CSV-file: gaussian.out" << std::endl;

    FILE* out = fopen("gaussian.out", "wt");

    std::vector< math::vertex< math::atlas_real<float> > > samples;
    sampler.getSamples(samples);

    for(unsigned int i=0;i<samples.size();i++){
      fprintf(out, "%f, %f\n",
	      samples[i][0].c[0], samples[i][1].c[0]);
    }
    
    fclose(out);
    
  }

  std::cout << "HMC sampling test DONE." << std::endl;
  
  
}


/************************************************************/

void rechcprint_test(hcnode< GDAParams, math::atlas_real<float> >* node,
		     unsigned int depth);


void gda_clustering_test()
{
  std::cout << "GDA CLUSTERING TEST" << std::endl;
  
  {
    whiteice::HC<GDAParams, math::atlas_real<float> > hc;
    whiteice::GDALogic behaviour;
    
    std::vector< math::vertex< math::atlas_real<float> > > data;
    
    // creates test data
    {
      std::vector< math::vertex< math::atlas_real<float> > > means;
      std::vector< unsigned int > sizes; // variance of all clusters is same
      
      math::vertex< math::atlas_real<float> > v(2);
      v[0] =  3.0f; v[1] =  7.5f; means.push_back(v); sizes.push_back(20);
      v[0] =  2.0f; v[1] =  5.0f; means.push_back(v); sizes.push_back(10);
      v[0] =  3.0f; v[1] =  5.0f; means.push_back(v); sizes.push_back(10);
      v[0] =  4.0f; v[1] =  5.0f; means.push_back(v); sizes.push_back(10);
      v[0] =  1.5f; v[1] =  1.5f; means.push_back(v); sizes.push_back(20);
      v[0] =  6.0f; v[1] =  5.0f; means.push_back(v); sizes.push_back(10);
      v[0] =  8.0f; v[1] =  6.0f; means.push_back(v); sizes.push_back(20);
      v[0] =  9.0f; v[1] =  6.0f; means.push_back(v); sizes.push_back(10);
      v[0] = 10.0f; v[1] =  4.0f; means.push_back(v); sizes.push_back(10);
      
      for(unsigned int j=0;j<sizes.size();j++){
	for(unsigned int i=0;i<sizes[j];i++){
	  v = means[j];
	  
	  v[0] += ((rand()/((float)RAND_MAX)) - 0.5f)*3.0f;
	  v[1] += ((rand()/((float)RAND_MAX)) - 0.5f)*3.0f;
	  
	  data.push_back(v);
	}
      }
      
    }
    
    
    hc.setProgramLogic(&behaviour);
    hc.setData(&data);
    
    if(hc.clusterize() == false){
      std::cout << "ERROR: CLUSTERING FAILED" << std::endl;
      return;
    }
    
    if(hc.getNodes().size() != 1){
      std::cout << "ERROR: INCORRECT NUMBER OF ROOT NODES" << std::endl;
      return;
    }
    
    std::cout << "FORM OF THE RESULTS IS CORRECT." << std::endl;
    std::cout << "PRINTING CLUSTERING RESULTS." << std::endl;
    
    rechcprint_test(hc.getNodes()[0], 1);
    
    std::cout << "CLUSTERING LIST PRINTED." << std::endl;
    std::cout << std::endl;
  }
}


void rechcprint_test(hcnode< GDAParams, math::atlas_real<float> >* node,
		     unsigned int depth)
{
  std::cout << "NODE " << std::hex << node 
	    << std::dec << " AT DEPTH " << depth << std::endl;
  std::cout << node->p.n << " POINTS  MEAN: " << node->p.mean << std::endl;

  if(node->data)
    std::cout << "LEAF NODE." << std::endl;
  
  if(node->childs.size()){
    std::cout << "CHILD NODES: ";
    std::cout << std::hex;
    
    for(unsigned int i=0;i<node->childs.size();i++)
      std::cout << node->childs[i] << " ";
    
    std::cout << std::dec << std::endl;
  }
  else{
    std::cout << "NO CHILDS (BUG?)" << std::endl;
  }
  
  std::cout << std::endl;
  
  
  for(unsigned int i=0;i<node->childs.size();i++)
    rechcprint_test(node->childs[i], depth+1);
}


/************************************************************/

void compressed_neuralnetwork_test()
{
  std::cout << "COMPRESSED NEURAL NETWORK TEST" << std::endl;
  
  using namespace whiteice::math;
  
  
  // first test - compare values are same with compressed and
  // uncompressed neural network
  {
    neuralnetwork< atlas_real<float> >* nn;
    neuralnetwork< atlas_real<float> >* cnn;
    
    std::vector<unsigned int> nn_arch;
    nn_arch.push_back(2);
    nn_arch.push_back(100);
    nn_arch.push_back(100);
    nn_arch.push_back(2);
    
    nn = new neuralnetwork< atlas_real<float> >(nn_arch, false);
    nn->randomize();
    cnn = new neuralnetwork< atlas_real<float> >(*nn);
    
    if(cnn->compress() == false){
      std::cout << "neural network compression failed"
		<< std::endl;
      
      delete nn;
      delete cnn;
      return;
    }
    
    
    // tests operation
    for(unsigned int i=0;i<10;i++){
      
      for(unsigned int d=0;d<nn->input().size();d++){
	nn->input()[d] = (rand() / ((float)RAND_MAX));
	cnn->input()[d] = nn->input()[d];
      }
      
      if(nn->calculate() == false ||
	 cnn->calculate() == false){
	std::cout << "neural network activation failed"
		  << std::endl;
	delete nn;
	delete cnn;
	return;
      }
      
      for(unsigned int d=0;d<nn->output().size();d++){
	if(nn->output()[d] != cnn->output()[d]){
	  std::cout << "nn and compressed nn outputs differ"
		    << "(dimension " << d << " )"
		    << nn->output()[d] << " != " 
		    << cnn->output()[d] << std::endl;
	  
	  delete nn;
	  delete cnn;
	  return;
	}
      }
    }
    
    
    delete nn;
    delete cnn;
    
  }
  
  
  // second test - compression rate of taught neural network
  // (compression ratio of random network >= 1)
  {
    neuralnetwork< atlas_real<float> >* nn;
    neuralnetwork< atlas_real<float> >* cnn;
    
    std::vector<unsigned int> nn_arch;
    nn_arch.push_back(2);
    nn_arch.push_back(100);
    nn_arch.push_back(100);
    nn_arch.push_back(2);
    
    nn = new neuralnetwork< atlas_real<float> >(nn_arch, false);
    
    nn->randomize();
    
    //////////////////////////////////////////////////
    // teaches nn
    
    dataset< atlas_real<float> >  in(2), out(2);
    
    // creates data
    {
      const unsigned int SIZE = 1000;
      std::vector< math::vertex<math::atlas_real<float> > > input;
      std::vector< math::vertex<math::atlas_real<float> > > output;
    
      input.resize(SIZE);
      output.resize(SIZE);
      
      for(unsigned int i=0;i<SIZE;i++){
	input[i].resize(2);
	output[i].resize(2);
	
	input[i][0] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
	input[i][1] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
      
	output[i][0] = input[i][0] - 0.2f * input[i][1];
	output[i][1] = -0.12f * input[i][0] + 0.1f * input[i][1];
      }
      
      if(in.add(input) == false || out.add(output) == false){
	std::cout << "data creation failed" << std::endl;
	return;
      }
      
      in.preprocess();
      out.preprocess();
    }
    
    
    // pso learner
    {
      nnPSO< math::atlas_real<float> >* nnoptimizer;
      nnoptimizer = 
	new nnPSO<math::atlas_real<float> >(nn, &in, &out, 20);

      nnoptimizer->improve(25);
      std::cout << "learnt nn error: " << nnoptimizer->getError()
		<< std::endl;
      
      std::cout << "random sample: "
		<< nnoptimizer->sample() << std::endl;
	
      
      delete nnoptimizer;
    }
    
    //////////////////////////////////////////////////
    
    cnn = new neuralnetwork< atlas_real<float> >(*nn);
    
    if(cnn->compress() == false){
      std::cout << "neural network compression failed"
		<< std::endl;
      
      delete nn;
      delete cnn;
      return;
    }
    
    
    // shows compression ratio

    std::cout << "NN COMPRESSION RATIO:" 
	      <<  cnn->ratio() << std::endl;
    
    delete nn;
    delete cnn;
  }
  
  
}



void simple_dataset_test()
{
  try{
    std::cout << "DATASET PREPROCESS TEST" 
	      << std::endl;
    
    math::atlas_real<float> a;
    
    math::vertex< math::atlas_real<float> > v[2];
    std::vector< math::vertex<math::atlas_real<float> > > data;
    dataset< math::atlas_real<float> > set(2);
    
    data.resize(2);
    data[0].resize(2);
    data[1].resize(2);
    v[0].resize(2);
    v[1].resize(2);
    
    std::cout << "OLD A " << a << std::endl;    
    a = 1.1;
    std::cout << "    A " << a << std::endl;    
    
    std::cout << "OLD DATA1: " << data[0] << std::endl;
    std::cout << "OLD DATA2: " << data[1] << std::endl;
    std::cout << "DATA1 SIZE: " << data[0].size() << std::endl;
    std::cout << "DATA2 SIZE: " << data[1].size() << std::endl;
    
    std::cout << "OLD V1: " << v[0] << std::endl;
    std::cout << "OLD V2: " << v[1] << std::endl;
    
    v[0][0] = 1.1;
    v[0][1] = 2.1;
    v[1][0] = -1.3;
    v[1][1] = 2.7;
    
    std::cout << "V1: " << v[0] << std::endl;
    std::cout << "V2: " << v[1] << std::endl;
    
    std::cout << "v[0] =";
    for(unsigned int i=0;i<v[0].size();i++)
      std::cout << " " << v[0][i];
    std::cout << std::endl;
    
    data[0] = v[0];
    data[1] = v[1];
    
    std::cout << "DATA1: " << data[0] << std::endl;
    std::cout << "DATA2: " << data[1] << std::endl;
    std::cout << "DATA1.0: " << data[0][0] << std::endl;
    std::cout << "DATA2.1: " << data[1][1] << std::endl;
    

    set.add(data);
    set.preprocess();

    
    std::cout << "DATASET PREPROCESS TESTS SUCCESFULLY DONE"
	      << std::endl;
    
    
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception " 
	      << e.what() << std::endl;
    return;
  }
}




void neuralnetwork_pso_test()
{
  try{
    std::cout << "NEURAL NETWORK PARTICLE SWARM OPTIMIZER" << std::endl;
    
    neuralnetwork< math::atlas_real<float> >* nn;
    nnPSO< math::atlas_real<float> >* nnoptimizer;
    dataset< math::atlas_real<float> > I1(2), O1(1);
    
    const unsigned int SIZE = 10000;
    
    std::vector< math::vertex<math::atlas_real<float> > > input;
    std::vector< math::vertex<math::atlas_real<float> > > output;
    
    // creates data
    {
      input.resize(SIZE);
      output.resize(SIZE);
      
      for(unsigned int i=0;i<SIZE;i++){
	input[i].resize(2);
	output[i].resize(1);
	
	input[i][0] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
	input[i][1] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
	
	output[i][0] = input[i][0] - 0.2f*input[i][1];
	// output[i][1] = -0.12f*input[i][0] + 0.1f*input[i][1];
      }
      
      
      for(unsigned int i=0;i<SIZE;i++){
	if(I1.add(input[i]) == false){
	  std::cout << "dataset creation failed [1]\n";
	  return;
	}
	
	if(O1.add(output[i]) == false){
	  std::cout << "dataset creation failed [2]\n";
	  return;
	}
      }
    }
    
    
    I1.preprocess(dataset< math::atlas_real<float> >::dnMeanVarianceNormalization);
    O1.preprocess(dataset< math::atlas_real<float> >::dnMeanVarianceNormalization);
    
    
    std::vector<unsigned int> nn_arch;
    nn_arch.push_back(2);
    nn_arch.push_back(10);
    nn_arch.push_back(1);
    
    nn = new neuralnetwork< math::atlas_real<float> >(nn_arch);
    nn->randomize();
    
    // nn arch
    {
      std::cout << "NN-ARCH = ";
      
      for(unsigned int i=0;i<nn_arch.size();i++)
	std::cout << nn_arch[i] << " ";
      
      std::cout << std::endl;
    }
    
    const unsigned int SWARM_SIZE = 20;
    const unsigned int STEP = 10;
    
    nnoptimizer = 
      new nnPSO<math::atlas_real<float> >
      (nn, &I1, &O1, SWARM_SIZE);
    
    nnoptimizer->verbosity(true);
    
    std::cout << "PSO OPTIMIZATION START" << std::endl;
    std::cout << "SWARM SIZE: " << SWARM_SIZE << std::endl;
    
    for(unsigned int i=0;i<(1000/STEP);i++){
      nnoptimizer->improve(STEP);
      
      std::cout << (i+1)*10 << ": " 
		<< nnoptimizer->getError() << std::endl;
    }
    
    
    delete nn;
    delete nnoptimizer;
    
    return;
  }
  catch(std::exception& e){
    return;
  }
}



void neuralnetwork_saveload_test()
{
  try{
    std::cout << "NEURAL NETWORK SAVE & LOAD TEST" << std::endl;
    
    // NOTE: saving and loading tests may 'fail' due to lack of precision
    // in floating point presentation if T type uses more accurate
    // representation
    
    
    // create random networks - save & load them and check that
    // NN is same after load
    
    neuralnetwork< math::atlas_real<float> >* nn[2];
    std::vector<unsigned int> s;
    
    std::string file = "nnarch.cfg";
    
    for(unsigned int i=0;i<10;i++){
      
      // makes neuro arch
      {
	s.clear();
	unsigned int k = (rand() % 10) + 2;
	
	do{
	  s.push_back((rand() % 4) + 2);
	  k--;
	}
	while(k > 0);
      }
      
      
      nn[0] = new neuralnetwork< math::atlas_real<float> >(s);
      nn[1] = new neuralnetwork< math::atlas_real<float> >(2, 2);
      
      // set random values of neural network
      nn[0]->randomize();
      
      for(unsigned int j=0;j<nn[0]->length();j++){
	(*nn[0])[j].moment() = 
	  math::atlas_real<float>(((float)rand()) / ((float)RAND_MAX));
	
	(*nn[0])[j].learning_rate() = 
	  math::atlas_real<float>(((float)rand()) / ((float)RAND_MAX));
	
	math::vertex<math::atlas_real<float> >& bb =
	  (*nn[0])[j].bias();
	
	math::matrix<math::atlas_real<float> >& MM =
	  (*nn[0])[j].weights();
	
	for(unsigned int k=0;k<bb.size();k++)
	  bb[k] = (rand() / ((float)RAND_MAX))*100.0;
	
	for(unsigned int k=0;k<MM.ysize();k++)
	  for(unsigned int l=0;l<MM.xsize();l++)
	    MM(k,l) = (rand() / ((float)RAND_MAX))*50.0;
      }
      
      std::cout << "saving ...\n";
      
      if(!(nn[0]->save(file))){
	std::cout << "neural network save() failed."
		  << std::endl;
	return;
      }
      
      std::cout << "loading ...\n";
      
      if(!(nn[1]->load(file))){
	std::cout << "neural network load() failed."
		  << std::endl;
	return;
      }
      

      // compares values between neural networks
      // (these should be same)
      
      std::cout << "compare: ";
      
      // compares global architecture      
      if(nn[0]->input().size() != nn[1]->input().size()){
	std::cout << "loaded nn: input size differ\n";
	return;
      }
      
      if(nn[0]->output().size() != nn[1]->output().size()){
	std::cout << "loaded nn: output size differ\n";
	return;
      }
      
      if(nn[0]->length() != nn[1]->length()){
	std::cout << "loaded nn: wrong number of layers\n";
	return;
      }
      else
	std::cout << "A";
      
      
      // compares layers
      for(unsigned int j=0;j<nn[1]->length();j++){
	
	if( (*nn[0])[j].input_size() != (*nn[1])[j].input_size() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : input size is wrong" << std::endl;
	  return;
	}
	
	
	if( (*nn[0])[j].size() != (*nn[1])[j].size() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : number of neurons is wrong" << std::endl;
	  return;
	}
	
	
	
	if( typeid( (*nn[0])[j].get_activation() ) != 
	    typeid( (*nn[1])[j].get_activation() ) ){
	  std::cout << "loaded nn, layer " << j 
		    << " : wrong activation function" << std::endl;
	  return;
	}
	
	
	// compares bias
	if( (*nn[0])[j].bias() != (*nn[1])[j].bias() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : incorrect bias" << std::endl;
	  return;
	}
	
	
	// compares weights
	if( (*nn[0])[j].weights() != (*nn[1])[j].weights() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : incorrect weight matrix" << std::endl;
	  return;
	}
	
	// compares moment & learning rate
	if( (*nn[0])[j].moment() != (*nn[1])[j].moment() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : incorrect moment" << std::endl;
	  return;
	}
	
	if( (*nn[0])[j].learning_rate() != (*nn[1])[j].learning_rate() ){
	  std::cout << "loaded nn, layer " << j 
		    << " : incorrect lrate" << std::endl;
	  return;
	}
	else
	  std::cout << "L";
      }            
      
      // checks nn activation works correctly
      nn[0]->calculate();
      nn[1]->calculate();
      
      std::cout << "C" << std::endl;
      
      delete nn[0];
      delete nn[1];
    }
    

    
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
    assert(0);
  }
}


/************************************************************/


void neuronlayer_test()
{
  // checks neuronlayer is ok
  
  std::cout << "NEURONLAYER TEST" << std::endl;
  
  // neuronlayer ctor (input, output, act. func) +
  // correct values + calculations are correct test
  try{    
    
    {
      math::vertex< math::atlas_real<float> > input, output;
      neuronlayer< math::atlas_real<float> >* l;
      
      odd_sigmoid< math::atlas_real<float> > os;
      
      input.resize(10);
      output.resize(5);
      
      l = new neuronlayer< math::atlas_real<float> >(&input, &output, os);
      
      if(l->input_size() != 10)
	throw test_exception("neuronlayer input_size() returns wrong value");
      
      if(l->size() != 5)
	throw test_exception("neuronlayer size() returns wrong value");
      
      if(l->input() != &input)
	throw test_exception("neuronlayer input() returns wrong value");
      
      if(l->output() != &output)
	throw test_exception("neuronlayer output() returns wrong value");
      
      // checks calculation with trivial case W = 0, b != 0.
      
      // sets bias and weights
      
      math::matrix< math::atlas_real<float> >& W = l->weights();
      math::vertex< math::atlas_real<float> >& b = l->bias();
      
      W = math::atlas_real<float>(0.0f); // set W
      
      for(unsigned int i=0;i<b.size();i++)
	b[i] = ((float)rand())/((float)RAND_MAX);
      
      std::cout << "bias = " << b << std::endl;
      std::cout << "output 1 = " << output << std::endl;
      if(l->calculate() == false)
	std::cout << "layer->calculate failed." << std::endl;
      std::cout << "output 2 = " << output << std::endl;
      
      // calculates bias = output manually
      for(unsigned int i=0;i<b.size();i++)
	b[i] = os(b[i]);
      
      output -= b;
      
      // calculates mean squared error
      {
	math::atlas_real<float> error = 0.0f;
	
	for(unsigned int i=0;i<output.size();i++)
	  error += output[i]*output[i];
	
	if(error > 0.01f){ 
	  std::cout << "cout: errors: " << output << std::endl;
	  throw test_exception("W = 0, b = x. test failed.");
	}
      }
      
      
      delete l;
    }
    
    
    {
      math::vertex< math::atlas_real<float> > input, output;
      neuronlayer< math::atlas_real<float> >* l;
      odd_sigmoid< math::atlas_real<float> > os;
      
      input.resize(10);
      output.resize(10);
      
      l = new neuronlayer< math::atlas_real<float> >(&input, &output, os);
      
      math::matrix< math::atlas_real<float> >& W = l->weights();
      math::vertex< math::atlas_real<float> >& b = l->bias();
    
      // W = I, b = 0. test
      
      for(unsigned int i=0;i<b.size();i++)
	b[i] = 0.0f;
      
      for(unsigned int j=0;j<W.ysize();j++){
	for(unsigned int i=0;i<W.xsize();i++){
	  if(i == j)
	    W(j,i) = 1.0f;
	  else
	    W(j,i) = 0.0f;
	}
      }
      
      
      for(unsigned int i=0;i<input.size();i++)
	input[i] = ((float)rand())/((float)RAND_MAX);
      
      l->calculate(); // output = g(W*x+b) = g(x)
      
      // calculates output itself
      for(unsigned int i=0;i<input.size();i++)
	input[i] = os(input[i]);
      
      output -= input;
      
      {
	math::atlas_real<float> error = 0.0f;
	
	for(unsigned int j=0;j<output.size();j++)
	  error += output[j] * output[j];
	
	if(error > 0.01f)
	  throw test_exception("W=I, b = 0. test failed.");
      }
      
      delete l;
    }
    
    
    {
      math::vertex< math::atlas_real<float> > input, output;
      neuronlayer< math::atlas_real<float> >* l;
      odd_sigmoid< math::atlas_real<float> > os;
      
      input.resize((rand() % 13) + 1);
      output.resize((rand() % 13) + 1);
      
      l = new neuronlayer< math::atlas_real<float> >(&input, &output, os);
      
      math::matrix< math::atlas_real<float> >& W = l->weights();
      math::vertex< math::atlas_real<float> >& b = l->bias();
      
      // sets weights and biases
      
      for(unsigned int j=0;j<W.ysize();j++){
	for(unsigned int i=0;i<W.xsize();i++){
	  W(j,i) = ((float)rand())/((float)RAND_MAX);
	}
      }
      
      for(unsigned int i=0;i<b.size();i++)
	b[i] = ((float)rand())/((float)RAND_MAX);
      
      // sets input & output
      
      for(unsigned int i=0;i<input.size();i++)
	input[i] = ((float)rand())/((float)RAND_MAX);
      
      math::vertex< math::atlas_real<float> > result(input);
      
      result = W*result;
      result += b;
      
      for(unsigned int i=0;i<result.size();i++)
	result[i] = os(result[i]);
      
      l->calculate();
      
      result -= output;
      
      {
	math::atlas_real<float> error = 0.0f;
	
	for(unsigned int i=0;i<result.size();i++)
	  error += result[i] * result[i];
	
	if(error > 0.01f)
	  throw test_exception("W=rand, b = rand. test failed.");
      }
      
      delete l;
    }
    
    
  }
  catch(std::exception& e){
    std::cout << "Error - unexpected exception: " 
	      << e.what() << std::endl;
  }
  
  
  // neuronlayer ctor (isize, nsize) and
  // neuronlayer copy ctor tests
  try{

    {
      neuronlayer< math::atlas_real<float> >* l[2];
      math::vertex< math::atlas_real<float> > input;
      math::vertex< math::atlas_real<float> > output;
      
      input.resize(10);
      output.resize(1);
      
      l[0] = new neuronlayer< math::atlas_real<float> >(10, 1);
      l[0]->input() = &input;
      l[0]->output() = &output;
      
      if(l[0]->input_size() != 10 || l[0]->size() != 1)
	throw test_exception("neuronlayer (isize, nsize) test failed.");
      
      // sets up biases and weights randomly
      
      math::vertex< math::atlas_real<float> >& b = l[0]->bias();
      math::matrix< math::atlas_real<float> >& W = l[0]->weights();
      
      for(unsigned int i=0;i<b.size();i++)
	b[i] = ((float)rand())/((float)RAND_MAX);
      
      for(unsigned int j=0;j<W.ysize();j++)
	for(unsigned int i=0;i<W.xsize();i++){
	  W(j,i) = ((float)rand())/((float)RAND_MAX);
	}
      
      
      l[1] = new neuronlayer< math::atlas_real<float> >(*(l[0]));
      
      if(l[0]->input() != l[1]->input()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [input()]");
      }
	
      if(l[0]->output() != l[1]->output()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [output()]");
      }
      
      if(l[0]->moment() != l[1]->moment()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [moment()]");
      }
      
      if(l[0]->learning_rate() != l[1]->learning_rate()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [learning_rate()]");
      }
      
      if(l[0]->input_size() != l[1]->input_size()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [input_size()]");
      }
      
      if(l[0]->size() != l[1]->size()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [size()]");
      }
      
      if(l[0]->bias() != l[1]->bias()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [bias()]");
      }
      
      if(l[0]->weights() != l[1]->weights()){
	throw test_exception("neuronlayer copy ctor simple sameness tests failed [weights()]");
      }
      
      
      // tests calculation() gives same output
      
      // sets input and output randomly
      
      for(unsigned int i=0;i<input.size();i++)
	(*(l[0]->input()))[i] = ((float)rand())/((float)RAND_MAX);
      
      for(unsigned int i=0;i<output.size();i++)
	(*(l[0]->output()))[i] = ((float)rand())/((float)RAND_MAX);
      
      math::vertex< math::atlas_real<float> > alternative_output(*(l[0]->output()));
      
      l[0]->output() = &alternative_output;
      
      l[0]->calculate();
      l[1]->calculate();
      
      output -= alternative_output;
      
      {
	// calculates error
	
	math::atlas_real<float> error = 0.0f;
	
	for(unsigned int i=0;i<output.size();i++)
	  error += output[i]*output[i];
	
	if(error >= 0.01f)
	  throw test_exception("neuronlayer copy ctor test: calculation result mismatch.");
	
      }
      
      delete l[0];
      delete l[1];
    }

  }
  catch(std::exception& e){
    std::cout << "Error - unexpected exception: " 
	      << e.what() << std::endl;
  }
}



void backprop_test(const unsigned int size)
{
  {
    cout << "SIMPLE BACKPROPAGATION TEST 1" << endl;
    
    /* tests linear separation ability of nn between two classes */
    
    std::vector< math::vertex<float > > input(size);
    std::vector< math::vertex<float > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);

      input[i][0] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
      
      output[i][0] = input[i][0] - 0.2*input[i][1];
      output[i][1] = -0.12*input[i][0] + 0.1*input[i][1];
      
      
      // output[i][0] += rand()/(double)RAND_MAX;
    }
    
    dataset< float > I0(2);
    if(!I0.add(input)){
      std::cout << "dataset creation failed\n";
      return;
    }
    
    dataset< float > I1(2);
    if(!I1.add(output)){
      std::cout << "dataset creation failed\n";
      return;
    }
  
  
    I0.preprocess();
    I1.preprocess();
    
    neuralnetwork< float >* nn;
    backpropagation< float > bp;
    
    vector<unsigned int> nn_arch;
    nn_arch.push_back(2);
    nn_arch.push_back(2); // 8
    nn_arch.push_back(2);
  
    /* two outputs == bad - fix/make nn structure more dynamic,
       ignores second one */
    nn = new neuralnetwork< float >(nn_arch); 
    nn->randomize();
    
    math::vertex< float > correct_output(2);
    
    // FOR SOME REASON C++ BACKPROPAGATION CODE HAS ERRORS. FIX IT (OR GCC 3.4 HAS BUGS.., TEST WITH GCC 3.3)
    
    float sum_error;
  
  
    std::cout << "NN-ARCH = ";
    
    for(unsigned int i=0;i<nn_arch.size();i++)
      std::cout << nn_arch[i] << " ";
    
    std::cout << std::endl;
    std::cout << "====================================================" << std::endl;
    
    bp.setData(nn, &I0, &I1);
    
    for(unsigned int e=0;e<100;e++){
      
      if(bp.improve(1) == false)
	std::cout << "BP::improve() failed." << std::endl;
      
      sum_error = bp.getError();
      
      std::cout << e << "/100  MEAN SQUARED ERROR: "
		<< sum_error << std::endl;
    }
    
    
    std::cout << "EPOCH ERROR SHOULD HAVE CONVERGED CLOSE TO ZERO." << std::endl;
    
    delete nn;
  }





  
  
  {
    cout << "SIMPLE BACKPROPAGATION TEST 2" << endl;
    
    /* tests linear separation ability of nn between two classes */
    
    std::vector< math::vertex<math::atlas_real<float> > > input(size);
    std::vector< math::vertex<math::atlas_real<float> > > output(size);

    // creates data
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);

      input[i][0] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*1.10f - 0.55f; // [-0.55,+0.55]
      
      output[i][0] = input[i][0] - 0.2f*input[i][1];
      output[i][1] = -0.12f*input[i][0] + 0.1f*input[i][1];      
    }
    
    dataset< math::atlas_real<float> > I0(2);
    if(!I0.add(input)){
      std::cout << "dataset creation failed\n";
      return;
    }
    
    dataset< math::atlas_real<float> > I1(2);
    if(!I1.add(output)){
      std::cout << "dataset creation failed\n";
      return;
    }
    
    std::cout << "pre abs maxs" << std::endl;
    math::atlas_real<float> mm = 0.0, MM = 0.0;
    
    for(unsigned int i=0;i<I0.size();i++){
      if(mm < whiteice::math::abs(I0[i][0]))
	mm = whiteice::math::abs(I0[i][1]);	
      if(MM < whiteice::math::abs(I0[i][1]))
	MM = whiteice::math::abs(I0[i][1]);
    }
    std::cout << "mm = " << mm << std::endl;
    std::cout << "MM = " << MM << std::endl;
    
    I0.preprocess();
    I1.preprocess();
    
    
    std::cout << "post abs maxs" << std::endl;
    mm = 0, MM = 0;
    for(unsigned int i=0;i<I0.size();i++){
      if(mm < whiteice::math::abs(I0[i][0]))
	mm = whiteice::math::abs(I0[i][1]);	
      if(MM < whiteice::math::abs(I0[i][1]))
	MM = whiteice::math::abs(I0[i][1]);
    }
    std::cout << "mm = " << mm << std::endl;
    std::cout << "MM = " << MM << std::endl;
    
    
    neuralnetwork< math::atlas_real<float> >* nn;
    backpropagation< math::atlas_real<float> > bp;
    
    vector<unsigned int> nn_arch;
    nn_arch.push_back(2);
    nn_arch.push_back(8);
    nn_arch.push_back(2);
    
    /* two outputs == bad - fix/make nn structure more dynamic,
       ignores second one */
    nn = new neuralnetwork< math::atlas_real<float> >(nn_arch); 
    nn->randomize();
    
    math::vertex< math::atlas_real<float> >& nn_input  = nn->input();
    math::vertex< math::atlas_real<float> >& nn_output = nn->output();
    
    math::vertex< math::atlas_real<float> > correct_output(2);
    
    math::atlas_real<float> sum_error;
    
    
    std::cout << "NN-ARCH = ";
    
    for(unsigned int i=0;i<nn_arch.size();i++)
      std::cout << nn_arch[i] << " ";
    
    std::cout << std::endl;
    
  
    for(unsigned int e=0;e<1000;e++){
      
      sum_error = 0;
      
      for(unsigned int i = 0;i<I0.size();i++){
	unsigned int index = rand() % I0.size();
	
	nn_input[0] = I0[index][0];
	nn_input[1] = I0[index][1];      
	
	nn->calculate();
	
	correct_output[0] = I1[index][0];
	correct_output[1] = I1[index][1];
	
	math::atlas_real<float> sq_error = 
	  (nn_output[0] - correct_output[0])*(nn_output[0] - correct_output[0]) +
	  (nn_output[1] - correct_output[1])*(nn_output[1] - correct_output[1]);
	
	// std::cout << "neural network i/o" << std::endl;
	// std::cout << nn_output[0] << std::endl;
	// std::cout << correct_output[0] << std::endl;
	// std::cout << nn_output[1] << std::endl;
	// std::cout << correct_output[1] << std::endl;
	
	sum_error += sq_error;
	
	if(bp.calculate(*nn, correct_output) == false){
	  cout << "WARNING CALCULATION FAILED: " << i << endl;
	}
	else{
	  //cout << "SQ_ERROR: " << sq_error << endl;
	}
	
      }
      
      sum_error = sqrt(sum_error / ((math::atlas_real<float>)I0.size()) );
      
      cout << "MEAN SQUARED ERROR: " << sum_error << endl;
    }
    
    
    cout << "EPOCH ERROR SHOULD HAVE CONVERGED CLOSE TO ZERO." << endl;
    
    delete nn;
  }
  
}



void neuralnetwork_test()
{
  cout << "SIMPLE NEURAL NETWORK TEST" << endl;
  
  neuralnetwork<double>* nn;
  
  {
    nn = new neuralnetwork<double>(100,10);
    
    math::vertex<double>& input  = nn->input();
    math::vertex<double>& output = nn->output();
    
    for(unsigned int i=0;i<input.size();i++)
      input[i] = 1.0;
    
    for(unsigned int i=0;i<output.size();i++)
      output[i] = 0.0;
    
    
    nn->randomize();
    nn->calculate();
    
    std::cout << "OUTPUTS " << std::endl;
    std::cout << output << std::endl;
    
    delete nn;
  }
  
  cout << "10-1024x10-4 NEURAL NETWORK TEST" << endl;
  
  {
    vector<unsigned int> nn_arch;
    nn_arch.push_back(10);
    
    for(unsigned int c=0;c<10;c++)
      nn_arch.push_back(1024);
    
    nn_arch.push_back(4);
    
    nn = new neuralnetwork<double>(nn_arch);
    
    math::vertex<double>& input  = nn->input();
    math::vertex<double>& output = nn->output();
    
    for(unsigned int i=0;i<input.size();i++)
      input[i] = 1.0;
    
    for(unsigned int i=0;i<output.size();i++)
      output[i] = 0.0;
    
    nn->randomize();
    nn->calculate();

    nn->randomize();
    nn->calculate();
    
    std::cout << "2ND NN OUTPUTS" << std::endl;
    std::cout << output << std::endl;
    
    delete nn;
  }
  
}


void neuronlayer_test2()
{
  cout << "NEURONLAYER TEST2" << endl;
  
  neuronlayer<double>* layer;
  
  layer = new neuronlayer<double>(10, 10); /* 10 inputs, 10 outputs 1-layer nn */
  
  math::vertex<double> input(10);
  math::vertex<double> output(10);

  layer->input()  = &input;
  layer->output() = &output;

  for(unsigned int i=0;i<input.size();i++) input[i] = 0;
  
  layer->bias()[5] = 10;

  layer->calculate();

  cout << "neural network activation results:" << endl;

  for(unsigned int i=0;i<10;i++){
    cout << output[i] << " ";
  }

  cout << endl;
    
  delete layer;
}



void neuron_test()
{
  cout << "NEURON TEST" << endl;
  
  neuron<double> n(10);
  double v = 0;
  
  n.bias() = -55*0.371;
  
  vector<double> input;
  
  input.resize(10);
  
  for(int i=1;i<=10;i++){
    input[i-1] = i*0.371;
    n[i-1] = 1;
  }
  
  v = n(input);
  
  cout << "neuron(" << n.local_field() << ") = " << v << endl; // should be zero
}



template <typename T>
void calculateF(activation_function<T>& F, T value)
{
  cout << "F(" << value << ") = " << F(value) << endl;
}


void activation_test()
{
  std::cout << "SIGMOID ACTIVATION FUNCTION TEST" << std::endl;
  
  activation_function<double> *F;
  F = new odd_sigmoid<double>;
  
  
  double value = -1;
  
  while(value <= 1.0){
    calculateF<double>(*F, value);
    value += 0.25;
  }
  
  delete F;
}


/************************************************************************/

void nnetwork_test()
{
  try{
    std::cout << "NNETWORK TEST -1: GET/SET PARAMETER TESTS"
	      << std::endl;

    std::vector<unsigned int> arch;
    arch.push_back(4);
    arch.push_back(4);
    arch.push_back(4);
    arch.push_back(5);
    
    nnetwork<> nn(arch); // 4-4-4-5 network (3 layer network)

    math::vertex<> b;
    math::matrix<> W;

    nn.getBias(b, 0);
    nn.getWeights(W, 0);

    std::cout << "First layer W*x + b." << std::endl;
    std::cout << "W = " << W << std::endl;
    std::cout << "b = " << b << std::endl;

    math::vertex<> all;
    nn.exportdata(all);

    std::cout << "whole nn vector = " << all << std::endl;

    W(0,0) = 100.0f;

    if(nn.setWeights(W, 1) == false)
      std::cout << "ERROR: cannot set NN weights." << std::endl;

    b.resize(5);

    if(nn.setBias(b, 2) == false)
      std::cout << "ERROR: cannot set NN bias." << std::endl;
    
    math::vertex<> b2;

    if(nn.getBias(b2, 2) == false)
      std::cout << "ERROR: cannot get NN bias." << std::endl;

    if(b.size() != b2.size())
      std::cout << "ERROR: bias terms mismatch (size)." << std::endl;

    math::vertex<> e = b - b2;

    if(e.norm() > 0.01)
      std::cout << "ERROR: bias terms mismatch." << std::endl;
    
      
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }

  

  
  try{
    std::cout << "NNETWORK TEST 0: SAVE() AND LOAD() TEST" << std::endl;
    
    nnetwork<>* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(18);
    arch.push_back(10);
    arch.push_back(1);

    std::vector<unsigned int> arch2;
    arch2.push_back(8);
    arch2.push_back(10);
    arch2.push_back(3);
    
    nn = new nnetwork<>(arch);    
    nn->randomize();
    
    nnetwork<>* copy = new nnetwork<>(*nn);
    
    if(nn->save("nntest.cfg") == false){
      std::cout << "nnetwork::save() failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }
    
    nn->randomize();
    
    nn = new nnetwork<>(arch2);    
    nn->randomize();
    
    if(nn->load("nntest.cfg") == false){
      std::cout << "nnetwork::load() failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }
    
    math::vertex<> p1, p2;
    
    if(nn->exportdata(p1) == false || copy->exportdata(p2) == false){
      std::cout << "nnetwork exportdata failed." << std::endl;
      delete nn;
      delete copy;
      return;
    }
    
    math::vertex<> e = p1 - p2;
    
    std::cout << "p1 = " << p1 << std::endl;
    std::cout << "p2 = " << p2 << std::endl;
    std::cout << "e  = " << e << std::endl;
    
    if(e.norm() > 0.001f){
      std::cout << "ERROR: save() & load() failed in nnetwork!" << std::endl;
      delete nn;
      delete copy;
      return;
    }
       
      
    delete nn;
    nn = 0;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  


  
  
  try{
    std::cout << "NNETWORK TEST 2: SIMPLE PROBLEM + DIRECT GRADIENT DESCENT" << std::endl;
    
    nnetwork<>* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(2);
    arch.push_back(20);
    arch.push_back(2);
    
    nn = new nnetwork<>(arch);
    
    const unsigned int size = 500;
    
    
    std::vector< math::vertex< math::atlas_real<float> > > input(size);
    std::vector< math::vertex< math::atlas_real<float> > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);
      
      input[i][0] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      
      output[i][0] = input[i][0] - math::atlas_real<float>(0.2f)*input[i][1];
      output[i][1] = math::atlas_real<float>(-0.12f)*input[i][0] + math::atlas_real<float>(0.1f)*input[i][1];
      
    }
    
    
    dataset<> data;
    std::string inString, outString;
    inString = "input";
    outString = "output";
    data.createCluster(inString,  2);
    data.createCluster(outString, 2);
    
    data.add(0, input);
    data.add(1, output);
    
    data.preprocess(0);
    data.preprocess(1);
    
    math::vertex<> grad, err, weights;
    
    unsigned int counter = 0;
    math::atlas_real<float> error = math::atlas_real<float>(1000.0f);
    math::atlas_real<float> lrate = math::atlas_real<float>(0.01f);
    while(error > math::atlas_real<float>(0.001f) && counter < 10000){
      error = math::atlas_real<float>(0.0f);
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back
      
      for(unsigned int i=0;i<data.size(0);i++){
	nn->input() = data.access(0, i);
	nn->calculate(true);
	err = data.access(1,i) - nn->output();
	
	for(unsigned int i=0;i<err.size();i++)
	  error += err[i]*err[i];
	
	if(nn->gradient(err, grad) == false)
	  std::cout << "gradient failed." << std::endl;
	
	if(nn->exportdata(weights) == false)
	  std::cout << "export failed." << std::endl;
	
	weights -= lrate * grad;
	
	if(nn->importdata(weights) == false)
	  std::cout << "import failed." << std::endl;
      }
      
      error /= math::atlas_real<float>((float)data.size());
      
      std::cout << counter << " : " << error << std::endl;
      
      counter++;
    }
    
    std::cout << counter << " : " << error << std::endl;
    
    delete nn;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }


  try{
    std::cout << "NNETWORK TEST 3: SIMPLE PROBLEM + SUM OF DIRECT GRADIENT DESCENT" << std::endl;
    
    nnetwork<>* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(2);
    arch.push_back(20);
    arch.push_back(2);
    
    nn = new nnetwork<>(arch);
    
    const unsigned int size = 500;
    
    
    std::vector< math::vertex< math::atlas_real<float> > > input(size);
    std::vector< math::vertex< math::atlas_real<float> > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);
      
      input[i][0] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      
      output[i][0] = input[i][0] - math::atlas_real<float>(0.2f)*input[i][1];
      output[i][1] = math::atlas_real<float>(-0.12f)*input[i][0] + math::atlas_real<float>(0.1f)*input[i][1];
      
    }
    
    
    dataset<> data;
    std::string inString, outString;
    inString = "input";
    outString = "output";
    data.createCluster(inString,  2);
    data.createCluster(outString, 2);
    
    data.add(0, input);
    data.add(1, output);
    
    data.preprocess(0);
    data.preprocess(1);
    
    math::vertex<> grad, err, weights;
    math::vertex<> sumgrad;
    
    unsigned int counter = 0;
    math::atlas_real<float> error = math::atlas_real<float>(1000.0f);
    math::atlas_real<float> lrate = math::atlas_real<float>(0.01f);
    
    while(error > math::atlas_real<float>(0.001f) && counter < 10000){
      error = math::atlas_real<float>(0.0f);
      
      // goes through data, calculates gradient
      // exports weights, weights -= 0.01*gradient
      // imports weights back

      math::atlas_real<float> ninv =
	math::atlas_real<float>(1.0f/data.size(0));
      
      for(unsigned int i=0;i<data.size(0);i++){
	nn->input() = data.access(0, i);
	nn->calculate(true);
	err = data.access(1,i) - nn->output();
	
	for(unsigned int j=0;j<err.size();j++)
	  error += err[j]*err[j];
	
	if(nn->gradient(err, grad) == false)
	  std::cout << "gradient failed." << std::endl;

	if(i == 0)
	  sumgrad = ninv*grad;
	else
	  sumgrad += ninv*grad;
      }
      
      
      if(nn->exportdata(weights) == false)
	std::cout << "export failed." << std::endl;
      
      weights -= lrate * sumgrad;
      
      if(nn->importdata(weights) == false)
	std::cout << "import failed." << std::endl;

      
      error /= math::atlas_real<float>((float)data.size());
      
      std::cout << counter << " : " << error << std::endl;
      
      counter++;
    }
    
    std::cout << counter << " : " << error << std::endl;
    
    delete nn;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }



  try{
    std::cout << "NNETWORK TEST 4: SIMPLE PROBLEM + HMC sampler" << std::endl;
    
    nnetwork<>* nn;
    
    std::vector<unsigned int> arch;
    arch.push_back(2);
    arch.push_back(10);
    arch.push_back(2);
    
    nn = new nnetwork<>(arch);
    
    const unsigned int size = 500;
    
    
    std::vector< math::vertex< math::atlas_real<float> > > input(size);
    std::vector< math::vertex< math::atlas_real<float> > > output(size);
    
    for(unsigned int i = 0;i<size;i++){
      input[i].resize(2);
      output[i].resize(2);
      
      input[i][0] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      input[i][1] = (((float)rand())/((float)RAND_MAX))*2.0f - 0.5f; // [-1.0,+1.0]
      
      output[i][0] = input[i][0] - math::atlas_real<float>(0.2f)*input[i][1];
      output[i][1] = math::atlas_real<float>(-0.12f)*input[i][0] + math::atlas_real<float>(0.1f)*input[i][1];
      
    }
    
    
    dataset<> data;
    std::string inString, outString;
    inString = "input";
    outString = "output";
    data.createCluster(inString,  2);
    data.createCluster(outString, 2);
    
    data.add(0, input);
    data.add(1, output);
    
    data.preprocess(0);
    data.preprocess(1);


    {
      whiteice::HMC<> hmc(*nn, data);

      hmc.startSampler(20); // 20 threads (extreme testcase)

      while(hmc.getNumberOfSamples() < 200){
	if(hmc.getNumberOfSamples() > 0){
	  std::cout << "Number of samples: "
		    << hmc.getNumberOfSamples() << std::endl;
	  std::cout << "Mean error (100 latest samples): "
		    << hmc.getMeanError(100) << std::endl;
	}
	sleep(1);
      }

      hmc.stopSampler();
      nn->importdata(hmc.getMean());
    }

    math::atlas_real<float> error = math::atlas_real<float>(0.0f);
      
    for(unsigned int i=0;i<data.size(0);i++){
      nn->input() = data.access(0, i);
      nn->calculate(true);
      math::vertex<> err = data.access(1,i) - nn->output();
	
      for(unsigned int j=0;j<err.size();j++)
	error += err[j]*err[j];
    }
      
    
    error /= math::atlas_real<float>((float)data.size());
    
    std::cout << "FINAL MEAN ERROR : " << error << std::endl;
    
    delete nn;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  

  

}

/******************************************************************/

void bayesian_nnetwork_test()
{
  try{
    std::cout << "BAYES NNETWORK TEST 0: SAVE() AND LOAD() TEST"
	      << std::endl;
    
    nnetwork<>* nn;
    bayesian_nnetwork<> bnn;
    
    std::vector<unsigned int> arch;
    arch.push_back(4);
    arch.push_back(10);
    arch.push_back(1);

    nn = new nnetwork<>(arch);

    std::vector< math::vertex<> > weights;
    weights.resize(10);

    for(unsigned int i=0;i<weights.size();i++){
      nn->randomize();
      if(nn->exportdata(weights[i]) == false){
	std::cout << "ERROR: NN exportdata failed.\n";
	return;
      }
    }
    
    if(bnn.importSamples(arch, weights) == false){
      std::cout << "ERROR: BNN importSamples() failed" << std::endl;
      return;
    }

    if(bnn.save("bnn_file.conf") == false){
      std::cout << "ERROR: BNN save() failed" << std::endl;
      return;
    }

    bayesian_nnetwork<> bnn2;

    if(bnn2.load("bnn_file.conf") == false){
      std::cout << "ERROR: BNN load() failed" << std::endl;
      return;
    }

    std::vector<unsigned int> loaded_arch;
    std::vector< math::vertex<> > loaded_weights;

    if(bnn2.exportSamples(loaded_arch, loaded_weights) == false){
      std::cout << "ERROR: BNN exportSamples() failed" << std::endl;
      return;
    }

    if(loaded_arch.size() != arch.size()){
      std::cout << "ERROR: BNN save/load arch size mismatch"
		<< loaded_arch.size() << " != "
		<< arch.size() 
		<< std::endl;
      return;
    }
    
    if(loaded_weights.size() != weights.size())
      std::cout << "ERROR: BNN save/load weight size mismatch "
		<< loaded_weights.size() << " != "
		<< weights.size()
		<< std::endl;

    for(unsigned int i=0;i<arch.size();i++){
      if(loaded_arch[i] != arch[i]){
	std::cout << "ERROR: BNN arch values mismatch" << std::endl;
	return;
      }
    }

    for(unsigned int i=0;i<loaded_weights.size();i++){
      math::vertex<> e = loaded_weights[i] - weights[i];

      if(e.norm() > 0.01f){
	std::cout << "ERROR: BNN weights value mismatch" << std::endl;
	return;
      }
    }
    
    
    delete nn;
    nn = 0;


    std::cout << "BAYES NNETWORK SAVE/LOAD TEST OK." << std::endl;
    
    
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }  
}

