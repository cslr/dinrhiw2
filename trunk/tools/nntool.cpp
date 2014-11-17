/*
 * nntool (narya) - 
 * a feedforward neural network
 * optimizer command line tool.
 * 
 * (C) copyright Tomas Ukkonen 2004, 2005, 2014-
 *
 *************************************************************
 * 
 * neural networks and other machine learning 
 * models try to build generic models which can
 * describe and predict essential features of data.
 * 
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <dinrhiw/dinrhiw.h>
#include <exception>

#include <vector>
#include <string>

#include "argparser.tab.h"
#include "cpuid_threads.h"

void print_usage(bool all);

int numberOfCPUThreads();

using namespace whiteice;



int main(int argc, char** argv)
{
  try{
    std::string datafn, nnfn;
    std::string lmethod;
    std::vector<std::string> lmods;
    std::vector<unsigned int> arch;
    unsigned int cmdmode;
    std::string ior, cns;
    bool daemon, verbose;
    
    bool overtrain = false;

    unsigned int samples = 0;
    unsigned int secs = 0;
    
    parse_commandline(argc, argv, datafn, nnfn, 
		      lmethod, lmods, arch, 
		      cmdmode, secs, ior, cns,
		      daemon, verbose);
    srand(time(0));

    if(secs <= 0) // no time limit
      samples = 2000; // we take 2000 samples

    const unsigned int threads = // for multithread-enabled code
      (unsigned int)numberOfCPUThreads();
    
    if(cmdmode != 0 || daemon == true){
      printf("Daemon and 'send command' modes aren't supported yet.\n");
      return 0;
    }
    
    // tries to open data and nnfile
    
    // loads data
    dataset<> data;
    bool stdinout_io = false;
    
    if(datafn.size() > 0){ // file input
      if(!data.load(datafn)){
	fprintf(stderr, "error: couldn't open datafile: %s.\n", datafn.c_str());
	exit(-1);
      }
      
      if(data.getNumberOfClusters() < 1){
	fprintf(stderr, "error: datafile is empty.\n");
	exit(-1);
      }
      
      if(lmethod != "use"){
	if(data.getNumberOfClusters() < 2){
	  fprintf(stderr, "error: datafile doesn't contain example pairs.\n");
	  exit(-1);
	}
      }
      
      
      if(arch[0] <= 0){
	arch[0] = data.dimension(0);
      }
      else if(arch[0] != data.dimension(0)){
	fprintf(stderr, "error: bad network input layer size, input data dimension pair.\n");
	exit(-1);
      }
      
      if(arch[arch.size()-1] <= 0){
	if(data.getNumberOfClusters() >= 2){
	  arch[arch.size()-1] = data.dimension(1);
	}
	else{
	  fprintf(stderr, "error: neural network do not have proper output dimension.\n");
	  exit(-1);
	}
      }
      else{
	if(data.getNumberOfClusters() >= 2){
	  if(arch[arch.size()-1] != data.dimension(1)){
	    fprintf(stderr, "error: bad network output layer size, output data dimension pair.\n");
	    exit(-1);
	  }
	}
      }
      
      if(data.size(0) == 0 || (data.size(1) == 0 && lmethod != "use")){
	fprintf(stderr, "error: empty datasets cannot be used for training.\n");
	exit(-1);
      }
      else if(lmethod != "use" && data.size(0) != data.size(1)){
	if(data.size(0) < data.size(1)){
	  printf("warning: output dataset is larger than input dataset.\n");
	  printf("some data is discarded. pairing may be incorrect.\n");
	  
	  data.resize(1, data.size(0));
	}
	else if(data.size(0) > data.size(1)){
	  printf("warning. input dataset is larger than output dataset.\n");
	  printf("some data is discarded. pairing may be incorrect.\n");
	  
	  data.resize(0, data.size(1));
	}
      }
    }
    else{
      stdinout_io = true;
      fprintf(stderr, "stdin/stdout I/O isn't supported yet.\n");    
      exit(-1);
    }
    

    nnetwork<>* nn = new nnetwork<>(arch);
    bayesian_nnetwork<>* bnn = new bayesian_nnetwork<>();
    
    if(verbose && !stdinout_io){
      if(lmethod == "use")
	printf("Processing %d data points.\n", data.size(0));
      else
	printf("%d data points for %d -> %d mapping.\n",
	       data.size(0), data.dimension(0), data.dimension(1));
    }
    

    fflush(stdout);
    
    if(lmethod == "grad+ot"){ // overtraining is activated
      lmethod = "grad";
      overtrain = true;
    }
    
    
    // learning or activation
    
    if(lmethod == "random"){
      unsigned int threads = (unsigned int)numberOfCPUThreads();
      
      if(verbose)
	std::cout << "Starting neural network parallel random search (T=" << secs << " seconds, " << threads << " threads).."
		  << std::endl;
      
      math::NNRandomSearch<> search;
      search.startOptimize(data, arch, threads);

      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::atlas_real<float> error = 100.0f;
	unsigned int solutions = 0;
	
	
	while(error > math::atlas_real<float>(0.001f) &&
	      counter < secs) // compute max SECS seconds
	{
	  search.getSolution(*nn, error, solutions);
	  	  
#ifndef WINNT
	  struct timespec ts;
	  ts.tv_sec  = 0;
	  ts.tv_nsec = 500000000; // 500ms
	  nanosleep(&ts, 0);
#else
	  Sleep(500);
#endif
	  
	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  printf("\r%d tries: %f (%f minutes remaining)         ", solutions, error.c[0], (secs - counter)/60.0f);
	  fflush(stdout);
	}
	
	printf("\r%d tries: %f (%f minutes remaining)           \n", solutions, error.c[0], (secs - counter)/60.0f);
	fflush(stdout);

	search.stopComputation();

	// gets the final (optimum) solution
	search.getSolution(*nn, error, solutions);

	bnn->importNetwork(*nn);
      }
      

    }
    else if(lmethod == "parallelgrad"){      
      
      if(verbose)
	std::cout << "Starting neural network parallel multistart gradient descent (T=" << secs << " seconds, " << threads << " threads).."
		  << std::endl;
      
      math::NNGradDescent<> grad;
      grad.startOptimize(data, arch, threads);

      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::atlas_real<float> error = 100.0f;
	unsigned int solutions = 0;
	
	
	while(counter < secs) // compute max SECS seconds
	{
	  grad.getSolution(*nn, error, solutions);
	  	  
#ifndef WINNT
	  struct timespec ts;
	  ts.tv_sec  = 0;
	  ts.tv_nsec = 500000000; // 500ms
	  nanosleep(&ts, 0);
#else
	  Sleep(500);
#endif
	  
	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  printf("\r%d tries: %f (%f minutes remaining)         ", solutions, error.c[0], (secs - counter)/60.0f);
	  fflush(stdout);
	}
	
	printf("\r%d tries: %f (%f minutes remaining)           \n", solutions, error.c[0], (secs - counter)/60.0f);
	fflush(stdout);

	grad.stopComputation();

	// gets the final (optimum) solution
	grad.getSolution(*nn, error, solutions);
	bnn->importNetwork(*nn);
      }

      
    }
    else if(lmethod == "grad"){
      if(verbose){
	std::cout << "Starting neural network gradient descent optimizer.."
		  << std::endl;
	if(overtrain)
	  std::cout << "Overtraining gradient descent method." << std::endl;
	else
	  std::cout << "Gradient descent with early stopping (testing dataset)." << std::endl;
      }
      
      
      if(overtrain == false){
	// divide data to training and testing sets
	dataset<> dtrain, dtest;
	
	dtrain = data;
	dtest  = data;
	
	dtrain.clearData(0);
	dtrain.clearData(1);
	dtest.clearData(0);
	dtest.clearData(1);
	
	for(unsigned int i=0;i<data.size(0);i++){
	  const unsigned int r = (rand() & 1);
	  
	  if(r == 0){
	    math::vertex<> in  = data.access(0,i);
	    math::vertex<> out = data.access(1,i);
	    
	    dtrain.add(0, in,  true);
	    dtrain.add(1, out, true);
	  }
	  else{
	    math::vertex<> in  = data.access(0,i);
	    math::vertex<> out = data.access(1,i);
	    
	    dtest.add(0, in,  true);
	    dtest.add(1, out, true);	    
	  }
	}
	
	// 1. normal gradient descent optimization using dtrain dataset
	// 2. after each iteration calculate the actual error terms from dtest dataset
	{
	  math::vertex<> grad, err, weights;
	  
	  unsigned int counter = 0;
	  math::atlas_real<float> prev_error, error, ratio;
	  math::atlas_real<float> lrate =
	    math::atlas_real<float>(0.05f);
	  math::atlas_real<float> delta_error = 0.0f;
	  
	  error = math::atlas_real<float>(1000.0f);
	  prev_error = math::atlas_real<float>(1000.0f);
	  ratio = math::atlas_real<float>(1000.0f);

	  math::vertex<> prev_sumgrad;
	  
	  while(error > math::atlas_real<float>(0.001f) && 
		ratio > math::atlas_real<float>(0.00001f) && 
		counter < 10000)
	  {
	    prev_error = error;
	    error = math::atlas_real<float>(0.0f);
	    
	    // goes through data, calculates gradient
	    // exports weights, weights -= lrate*gradient
	    // imports weights back

	    math::vertex<> sumgrad;
	    math::atlas_real<float> ninv =
	      math::atlas_real<float>(1.0f/dtrain.size(0));
	    
	    for(unsigned int i=0;i<dtrain.size(0);i++){
	      nn->input() = dtrain.access(0, i);
	      nn->calculate(true);
	      err = dtrain.access(1,i) - nn->output();
	      
	      if(nn->gradient(err, grad) == false)
		std::cout << "gradient failed." << std::endl;

	      if(i == 0)
		sumgrad = ninv*grad;
	      else
		sumgrad += ninv*grad;
	    }

	    
	    if(nn->exportdata(weights) == false)
	      std::cout << "export failed." << std::endl;

	    if(prev_sumgrad.size() <= 1){
	      weights -= lrate * sumgrad;
	      prev_sumgrad = lrate * sumgrad;
	    }
	    else{
	      math::atlas_real<float> momentum =
		math::atlas_real<float>(0.8f);
	      weights -= lrate * sumgrad + momentum*prev_sumgrad;
	      prev_sumgrad = lrate * sumgrad;
	    }
	   
	    
	    if(nn->importdata(weights) == false)
	      std::cout << "import failed." << std::endl;

	    
	    // calculates error from the testing dataset
	    for(unsigned int i=0;i<dtest.size(0);i++){
	      nn->input() = dtest.access(0, i);
	      nn->calculate(false);
	      err = dtest.access(1,i) - nn->output();
	      
	      for(unsigned int i=0;i<err.size();i++)
		error += (err[i]*err[i]) / math::atlas_real<float>((float)err.size());
	    }
	    
	    error /= math::atlas_real<float>((float)dtest.size());
	    
	    delta_error = abs(error - prev_error);
	    ratio = delta_error / error;
	    
	    printf("\r%d : %f (%f)                  ", counter, error.c[0], ratio.c[0]);
	    fflush(stdout);
	    
	    counter++;
	  }
	
	  printf("\r%d : %f (%f)                  \n", counter, error.c[0], ratio.c[0]);
	  fflush(stdout);
	}

      }
      else{
	math::vertex<> grad, err, weights;
	
	unsigned int counter = 0;
	math::atlas_real<float> prev_error, error, ratio;
	math::atlas_real<float> lrate = math::atlas_real<float>(0.01f);
	math::atlas_real<float> delta_error = 0.0f;
	
	error = math::atlas_real<float>(1000.0f);
	prev_error = math::atlas_real<float>(1000.0f);
	ratio = math::atlas_real<float>(1000.0f);

	math::vertex<> sumgrad;
	math::atlas_real<float> ninv =
	  math::atlas_real<float>(1.0f/data.size(0));

	
	// we are overtraining so we ignore the ratio-parameter
	// as a stopping condition
	
	while(error > math::atlas_real<float>(0.001f) && 
	      counter < 10000)
	{
	  prev_error = error;
	  error = math::atlas_real<float>(0.0f);
	  
	  // goes through data, calculates gradient
	  // exports weights, weights -= lrate*gradient
	  // imports weights back
	  
	  for(unsigned int i=0;i<data.size(0);i++){
	    nn->input() = data.access(0, i);
	    nn->calculate(true);
	    err = data.access(1,i) - nn->output();
	    
	    for(unsigned int j=0;j<err.size();j++)
	      error += (err[j]*err[j]) / math::atlas_real<float>((float)err.size());
	    
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
	  
	  delta_error = abs(error - prev_error);
	  ratio = delta_error / error;
	  
	  printf("\r%d : %f (%f)                  ", counter, error.c[0], ratio.c[0]);
	  fflush(stdout);
	  
	  counter++;
	}
	
	printf("\r%d : %f (%f)                  \n", counter, error.c[0], ratio.c[0]);
	fflush(stdout);
     }

      bnn->importNetwork(*nn);
      
    }
    else if(lmethod == "bayes"){
      if(verbose){
	if(secs > 0){
	  std::cout << "Starting neural network bayesian inference (T=" << secs << " secs, "
		    << threads << " threads)..."
		    << std::endl;
	}
	else{
	  std::cout << "Starting neural network bayesian inference (" << threads << " threads)..."
		    << std::endl;
	}
      }

      const bool adaptive = true;
      whiteice::HMC<> hmc(*nn, data, adaptive);
      whiteice::linear_ETA<float> eta;
      
      time_t t0 = time(0);
      unsigned int counter = 0;

      if(samples > 0)
	eta.start(0.0f, (float)samples);
      
      hmc.startSampler(threads);
      
      while((hmc.getNumberOfSamples() < samples && samples > 0) || (counter < secs && secs > 0)){

	eta.update((float)hmc.getNumberOfSamples());
	
	if(hmc.getNumberOfSamples() > 0){
	  if(secs > 0)
	    printf("\r%d samples: %f (%f minutes remaining)                 ", hmc.getNumberOfSamples(), hmc.getMeanError(100).c[0], (secs - counter)/60.0);
	  else{
	    printf("\r%d/%d samples : %f (%f minutes remaining)                ",
		   hmc.getNumberOfSamples(), samples,
		   hmc.getMeanError(100).c[0], eta.estimate()/60.0);
	  }
	  fflush(stdout);
	}
	sleep(1);

	time_t t1 = time(0);
	counter = (unsigned int)(t1 - t0);
      }
      
      hmc.stopSampler();

      if(secs > 0)
	printf("\r%d samples : %f                    \n", hmc.getNumberOfSamples(), hmc.getMeanError(100).c[0]);
      else
	printf("\r%d/%d samples : %f                 \n", hmc.getNumberOfSamples(), samples, hmc.getMeanError(100).c[0]);
      
      fflush(stdout);
      
      // nn->importdata(hmc.getMean());
      delete nn;
      nn = NULL;
      
      bnn = new bayesian_nnetwork<>();
      assert(hmc.getNetwork(*bnn) == true);

      // instead of using mean weight vector
      // we now use y = E[network(x,w)] in bayesian inference
      //
      // TODO: what we really want is
      //       the largest MODE of p(w|data) distribution as 
      //       this is then the global minima (assuming samples
      //       {w_i} converge to p(w|data)).
      
    }
    else if(lmethod == "use"){
      if(verbose)
	std::cout << "Activating loaded neural network configuration.."
		  << std::endl;
      
      if(bnn->load(nnfn) == false){
	std::cout << "Loading neural network failed." << std::endl;
	delete nn;
	delete bnn;
	nn = NULL;
	return -1;
      }
      
      
      if(bnn->inputSize() != data.dimension(0)){
	std::cout << "Neural network input dimension mismatch for input dataset ("
		  << bnn->inputSize() << " != " << data.dimension(0) << ")"
		  << std::endl;
	delete nn;
	delete bnn;
	nn = NULL;
	return -1;
      }
      
      
      bool compare_clusters = false;
      
      if(data.getNumberOfClusters() == 2){
	if(data.size(0) > 0 && data.size(1) > 0 && 
	   data.size(0) == data.size(1)){
	  compare_clusters = true;
	  
	  if(bnn->outputSize() != data.dimension(1)){
	    std::cout << "Neural network output dimension mismatch for dataset ("
		      << bnn->outputSize() << " != " << data.dimension(1) << ")"
		      << std::endl;
	    delete nn;
	    return -1;	    
	  }
	}
      }
	
      
      
      if(compare_clusters == true){
	math::atlas_real<float> error = math::atlas_real<float>(0.0f);
	math::vertex<> err;
	
	for(unsigned int i=0;i<data.size(0);i++){
	  math::vertex<> out;
	  math::matrix<> cov;

	  bnn->calculate(data.access(0, i), out, cov);
	  err = data.access(1,i) - out;
	  
	  for(unsigned int i=0;i<err.size();i++)
	    error += (err[i]*err[i]) / math::atlas_real<float>((float)err.size());
	  
	}
	
	error /= math::atlas_real<float>((float)data.size());
	
	std::cout << "Average error in dataset: " << error << std::endl;
      }
      
      else{
	std::cout << "Predicting data points.." << std::endl;
	
	if(data.getNumberOfClusters() == 2 && data.size(0) > 0){
	  
	  data.clearData(1);
	
	  for(unsigned int i=0;i<data.size(0);i++){
	    math::vertex<> out;
	    math::matrix<> cov;
	    
	    bnn->calculate(data.access(0, i),  out, cov);
	    // we do NOT preprocess the output but inject it directly into dataset
	    data.add(1, out, true);
	  }
	  
	  if(data.save(datafn) == true)
	    std::cout << "Storing results to dataset file." << std::endl;
	  else
	    std::cout << "Storing results to dataset file FAILED." << std::endl;
	}
      }
    }
    
    if(lmethod != "use"){
      if(bnn){
	if(bnn->save(nnfn) == false){
	  std::cout << "Saving network data failed." << std::endl;
	  delete bnn;
	  return -1;
	}
      }
    }
    
    
    if(nn){
      delete nn;
      nn = 0;
    }
    
    if(bnn){
      delete bnn;
      bnn = 0;
    }
    
    return 0;
  }
  catch(std::exception& e){
    std::cout << "Fatal error: unexpected exception. Reason: " 
	      << e.what() << std::endl;
    return -1;
  }
  
}



void print_usage(bool all)
{
  printf("Usage: nntool [options] [data] [arch] <nnfile> [lmethod]\n");
  
  if(!all){
    printf("Try 'nntool --help' for more information.\n");
    return;
  }
  
  
  printf("Create, train and use neural network(s).\n\n");
  printf("-v             shows ETA and other details\n");
  printf("--help         shows this help\n");
  printf("--version      displays version and exits\n");
  printf("--time TIME    sets time limit for multistart optimization and bayesian inference\n");
  printf("[data]         a source file for inputs or i/o examples (binary file)\n");
  printf("               (whiteice data file format created by dstool)\n");
  printf("[arch]         the architecture of a new nn. Eg. 3-10-9 or ?-10-?\n");
  printf("<nnfile>       input/output neural networks weights file\n");
  printf("[lmethod]      method: use, grad, grad+ot, parallelgrad, random, bayes\n\n");
  
  printf("Report bugs to <dinrhiw2.sourceforge.net>.\n");
  
}


