/*
 * nntool (narya) - 
 * a feedforward neural network
 * optimizer command line tool.
 * 
 * (C) Copyright Tomas Ukkonen 2004, 2005, 2014-2016
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
#include <signal.h>
#include <assert.h>


#include <dinrhiw/dinrhiw.h>
#include <exception>

#include <vector>
#include <string>

#include "argparser.tab.h"
#include "cpuid_threads.h"

#undef __STRICT_ANSI__
#include <fenv.h>



void print_usage(bool all);

void sleepms(unsigned int ms);


volatile bool stopsignal = false; // is set true if receives CRTL-C or some other signal
void install_signal_handler();


using namespace whiteice;



int main(int argc, char** argv)
{
  try{
    std::string datafn, nnfn;
    std::string lmethod;
    std::vector<std::string> lmods;
    std::vector<unsigned int> arch;
    unsigned int cmdmode;
    bool no_init, verbose;
    bool load, help = false;
    
    bool overfit = false;
    bool adaptive = false;
    bool negfeedback = false;
    bool deep = false;
    bool pseudolinear = false;
    bool purelinear = false;

    // should we use recurent neural network or not..
    unsigned int SIMULATION_DEPTH = 1;
    
    unsigned int samples = 0; // number of samples or iterations in learning process
    unsigned int secs = 0;    // how many seconds the learning process should take

    // number of threads used in optimization
    unsigned int threads = 0;

    // number of datapoints to be used in learning (taken randomly from the dataset)
    unsigned int dataSize = 0;

    
#ifdef _GLIBCXX_DEBUG    
    // enables FPU exceptions
    feenableexcept(FE_INVALID |
		   FE_DIVBYZERO);
#endif
    
    parse_commandline(argc,
		      argv,
		      datafn,
		      nnfn, 
		      lmethod,
		      lmods,
		      arch, 
		      cmdmode,
		      secs,
		      samples,
		      threads,
		      dataSize,
		      SIMULATION_DEPTH,
		      no_init,
		      load,
		      overfit,
		      adaptive,
		      negfeedback,
		      deep,
		      pseudolinear,
		      purelinear,
		      help,
		      verbose);
    srand(time(0));

    if(secs <= 0 && samples <= 0) // no time limit
      samples = 4000; // we take 4000 samples/tries as the default

    if(help){ // prints command line usage information
      print_usage(true);
      return 0;
    }

    install_signal_handler();

    if(threads <= 0)
      threads = // for multithread-enabled code
	        // only uses half of the resources as the default
	(unsigned int)numberOfCPUThreads()/2;
    
    if(threads <= 0)
      threads = 1;
    
    
    if(cmdmode != 0){
      printf("Daemon and 'send command' modes aren't supported yet.\n");
      return 0;
    }
    
    // tries to open data and nnfile
    
    // loads data
    dataset< whiteice::math::blas_real<double> > data;
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
      
      if(lmethod != "use" && lmethod != "minimize"){
	if(data.getNumberOfClusters() < 2){
	  fprintf(stderr, "error: datafile doesn't contain example pairs.\n");
	  exit(-1);
	}
      }
      
      
      if(arch[0] <= 0){
	if(SIMULATION_DEPTH > 1){
	  if(data.getNumberOfClusters() >= 2){
	    arch[0] = data.dimension(0)+data.dimension(1);
	  }
	  else{
	    fprintf(stderr, "error: cannot compute recurrent network input layer size.\n");
	    exit(-1);
	  }
	}
	else{
	  arch[0] = data.dimension(0);
	}
      }
      else{
	if(SIMULATION_DEPTH > 1){
	  if(data.getNumberOfClusters() >= 2){
	    if(arch[0] != data.dimension(0)+data.dimension(1)){
	      fprintf(stderr, "error: bad recurrent network input layer size, input data dimension pair.\n");
	      exit(-1);
	    }
	  }
	}
	else{
	  if(arch[0] != data.dimension(0)){
	    fprintf(stderr, "error: bad network input layer size, input data dimension pair.\n");
	    exit(-1);
	  }
	}
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
      
      if(data.size(0) == 0 || (data.size(1) == 0 && (lmethod != "use" && lmethod != "minimize"))){
	fprintf(stderr, "error: empty datasets cannot be used for training.\n");
	exit(-1);
      }
      else if((lmethod != "use" && lmethod != "minimize") && data.size(0) != data.size(1)){
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

    if(pseudolinear && deep){
      fprintf(stderr,"Cannot set both deep and pseudolinear options at the same time.\n");
      exit(-1);
    }

    if(pseudolinear && deep){
      fprintf(stderr,"Cannot set both deep and purelinear options at the same time.\n");
      exit(-1);
    }

    if(pseudolinear && purelinear){
      fprintf(stderr,"Cannot set both pseudolinear and purelinear options at the same time.\n");
      exit(-1);
    }

    if(SIMULATION_DEPTH > 1){
      if(verbose){
	if(pseudolinear){
	  printf("Simple recurrent neural network (pseudolinear).\n");
	}
	else if(purelinear){
	  printf("Simple recurrent neural network (purelinear).\n");
	}
	else{
	  printf("Simple recurrent neural network (sigmoid).\n");
	}
      }
    }
    

    nnetwork< whiteice::math::blas_real<double> >* nn = NULL;
    nn = new nnetwork< whiteice::math::blas_real<double> >(arch);
    bayesian_nnetwork< whiteice::math::blas_real<double> >* bnn = new bayesian_nnetwork< whiteice::math::blas_real<double> >();

    if(pseudolinear){
      nn->setNonlinearity(whiteice::nnetwork< whiteice::math::blas_real<double> >::halfLinear);
    }
    else if(purelinear){
      nn->setNonlinearity(whiteice::nnetwork< whiteice::math::blas_real<double> >::pureLinear);
    }
    else{
      nn->setNonlinearity(whiteice::nnetwork< whiteice::math::blas_real<double> >::sigmoid);
    }

    
    if(verbose && !stdinout_io){
      math::vertex< whiteice::math::blas_real<double> > w;
      nn->exportdata(w);
      
      if(lmethod == "use"){
	printf("Processing %d data points (%d parameters in neural network).\n", data.size(0), w.size());
      }
      else{
	if(SIMULATION_DEPTH <= 1)
	  printf("%d data points for %d -> %d mapping (%d parameters in neural network).\n",
		 data.size(0), data.dimension(0), data.dimension(1),
		 w.size());
	else
	  printf("%d data points for %d+%d -> %d mapping (%d parameters in neural network).\n",
		 data.size(0), data.dimension(0), data.dimension(1), data.dimension(1),
		 w.size());
      }
    }
    
    
    fflush(stdout);

    
    if(lmethod != "use" && dataSize > 0 && dataSize < data.size(0)){
      printf("Resampling dataset down to %d datapoints.\n", dataSize);

      data.downsampleAll(dataSize);
    }

    if((lmethod != "use" && lmethod != "minimize") && deep == true){
      printf("Deep pretraining (stacked RBMs) of neural network weights (slow).\n");

      const bool binary = false; // trains Gaussian-Bernoulli RBM at the first layer
      
      if(deep_pretrain_nnetwork(nn, data, binary, verbose) == false){
	printf("ERROR: deep pretraining of nnetwork failed.\n");
	return -1;
      }

      // nn->setStochastic(false); // do not use stochastic values
    }
    /*
     * default: initializes nnetwork weight values using 
     * (simple) deep ica if possible
     */
    else if((lmethod != "use" && lmethod != "minimize") && no_init == false && load == false)
    {

      if(verbose)
	std::cout << "Heuristics: NN weights normalization initialization."
		  << std::endl;

      if(normalize_weights_to_unity(*nn, true) == false){
	std::cout << "ERROR: NN weights normalization FAILED."
		  << std::endl;
	return -1;
      }

      // also sets initial weights to be "orthogonal" against each other
      if(negfeedback){
	math::blas_real<double> alpha = 0.5f;
	negative_feedback_between_neurons(*nn, data, alpha);
      }
      
    }
    else if(load == true){
      if(verbose)
	std::cout << "Loading the previous network data from the disk." << std::endl;

      if(bnn->load(nnfn) == false){
	std::cout << "ERROR: Loading neural network failed." << std::endl;
	if(nn) delete nn;
	if(bnn) delete bnn;
	nn = NULL;
	return -1;
      }

      std::vector< math::vertex< whiteice::math::blas_real<double> > > weights;
      nnetwork< whiteice::math::blas_real<double> > nnParams;

      if(bnn->exportSamples(nnParams, weights) == false){
	std::cout << "ERROR: Loading neural network failed." << std::endl;
	if(nn) delete nn;
	if(bnn) delete bnn;
	nn = NULL;
	return -1;
      }

      std::vector<unsigned int> loadedArch;
      nnParams.getArchitecture(loadedArch);

      if(arch.size() != loadedArch.size()){
	std::cout << "ERROR: Mismatch between loaded network architecture and parameter architecture." << std::endl;
	if(nn) delete nn;
	if(bnn) delete bnn;
	nn = NULL;
	return -1;
      }

      for(unsigned int i=0;i<arch.size();i++){
	if(arch[i] != loadedArch[i]){
	  std::cout << "ERROR: Mismatch between loaded network architecture and parameter architecture." << std::endl;
	  if(nn) delete nn;
	  if(bnn) delete bnn;
	  nn = NULL;
	  return -1;
	}
      }
      

      *nn = nnParams;

      // just pick one randomly if there are multiple ones
      unsigned int index = 0;
      if(weights.size() > 1)
	index = rand() % weights.size();

      if(nn->importdata(weights[index]) == false){
	std::cout << "ERROR: Loading neural network failed (incorrect network architecture?)." << std::endl;
	if(nn) delete nn;
	if(bnn) delete bnn;
	return -1;
      }
      
    }
    

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // learning or activation
    if(lmethod == "mix"){
      // mixture of experts
      Mixture< whiteice::math::blas_real<double> > moe(2, SIMULATION_DEPTH, overfit, negfeedback); 
            

      time_t t0 = time(0);
      unsigned int counter = 0;
      unsigned int iterations = 0;
      whiteice::math::blas_real<double> error = 1000.0;
      whiteice::linear_ETA<double> eta;

      if(samples > 0)
	eta.start(0.0f, (double)samples);

      moe.minimize(*nn, data);
      
      while(error > math::blas_real<double>(0.001f) &&
	    (counter < secs || secs <= 0) && // compute max SECS seconds
	    (iterations < samples || samples <= 0) && // or max samples
	    moe.solutionConverged() == false && moe.isRunning() == true && // or until solution converged.. (or exit due error)
	    !stopsignal)
      {
	sleep(1);

	std::vector< whiteice::math::vertex< whiteice::math::blas_real<double> > > weights;
	std::vector< whiteice::math::blas_real<double> > errors;
	std::vector< whiteice::math::blas_real<double> > percent;
	unsigned int changes = 0;
	
	moe.getSolution(weights, errors, percent, iterations, changes);

	error = 0.0;
	double c = 0;

	for(unsigned int i=0;i<errors.size();i++){
	  if(isinf(errors[i]) == false){
	    error += errors[i];
	    c++;
	  }
	}

	if(c > 0.0) error /= c;
	else error = 1000.0;
	
	eta.update(iterations);
	
	time_t t1 = time(0);
	counter = (unsigned int)(t1 - t0); // time-elapsed
	
	if(secs > 0){
	  printf("\r                                                            \r");
	  printf("%d iters: %f [%.1f minutes]",
		 iterations, 
		 error.c[0], (secs - counter)/60.0f);
	}
	else{
	  printf("\r                                                            \r");
	  printf("%d/%d iters: %f [%.1f minutes]",
		 iterations, samples,  
		 error.c[0], eta.estimate()/60.0f);
	  
	}
	  
	fflush(stdout);
      }

      if(moe.isRunning())
	moe.stopComputation();

      // TODO: copy majority expert as the predicting expert (so we just work against noise here..)
      assert(0); // FIXME!
      
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "bbrbm" || lmethod == "gbrbm"){
      // greedely learns Bernoulli-Bernoulli RBM or Gaussian-Bernoulli RBM
      // where the last layer is linear predictor/regression layer

      // the idea here is that we train N-M-output neural network where M is much larger than input
      // RBM does non-linear transformation and calculates features which are then linearly combined.
      // For example, optimize 2-layer  1207-10000-1 neural network where 10000 calculates
      // 10.000 features from input space which are then linearly combined.

      printf("Starting RBM neural network optimization (ignoring parameters)..\n");

      bool binary = false; // as the default assumes the first layer is gaussian RBM

      if(lmethod == "bbrbm") binary = true;  // trains full binary RBM (last layer is linear)
      else binary = false;                   // trains gaussian-bernoully RBM (last layer is linear)
      
      if(deep_pretrain_nnetwork(nn, data, binary, verbose) == false){
	printf("ERROR: deep pretraining of nnetwork failed.\n");
	return -1;
      }

      if(bnn->importNetwork(*nn) == false){
	std::cout << "ERROR: internal error cannot import optimized RBM to data structure" << std::endl;
	return -1;
      }
      
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "lbfgs"){
      
      if(verbose){
	if(overfit == false){
	  if(secs > 0)
	    std::cout << "Starting neural network L-BFGS optimization with early stopping (T=" << secs << " seconds).."
		      << std::endl;
	  else
	    std::cout << "Starting neural network L-BFGS optimization with early stopping.."
		      << std::endl;
	}
	else{
	  if(secs > 0)
	    std::cout << "Starting neural network L-BFGS optimization (T=" << secs << " seconds threads).."
		      << std::endl;
	  else
	    std::cout << "Starting neural network L-BFGS optimization.."
		      << std::endl;
	}
      }

      if(secs <= 0 && samples <= 0){
	fprintf(stderr, "L-BFGS search requires --time or --samples command line switch.\n");
	return -1;
      }

      
      rLBFGS_nnetwork< whiteice::math::blas_real<double> > bfgs(*nn, data, SIMULATION_DEPTH, overfit, negfeedback);
      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::blas_real<double> error = 1000.0f;
	math::vertex< whiteice::math::blas_real<double> > w;
	unsigned int iterations = 0;
	whiteice::linear_ETA<double> eta;

	if(samples > 0)
	  eta.start(0.0f, (double)samples);

	// initial starting position
	nn->exportdata(w);
	
	bfgs.minimize(w);

	while(error > math::blas_real<double>(0.001f) &&
	      (counter < secs || secs <= 0) && // compute max SECS seconds
	      (iterations < samples || samples <= 0) && // or max samples
	      bfgs.solutionConverged() == false && bfgs.isRunning() == true && // or until solution converged.. (or exit due error)
	      !stopsignal) 
	{
	  sleep(1);
	  
	  bfgs.getSolution(w, error, iterations);
	  
	  eta.update(iterations);

	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  if(secs > 0){
	    printf("\r                                                            \r");
	    printf("%d iters: %f [%.1f minutes]",
		   iterations, 
		   error.c[0], (secs - counter)/60.0f);
	  }
	  else{
	    printf("\r                                                            \r");
	    printf("%d/%d iters: %f [%.1f minutes]",
		   iterations, samples,  
		   error.c[0], eta.estimate()/60.0f);

	  }
	  
	  fflush(stdout);
	}
	      
	
	if(secs > 0){
	  printf("\r                                                            \r");
	  printf("%d iters: %f [%.1f minutes]\n",
		 iterations,
		 error.c[0], (secs - counter)/60.0f);
	}
	else{
	  printf("\r                                                            \r");
	  printf("%d/%d iters: %f [%.1f minutes]\n",
		 iterations, samples,  
		 error.c[0], eta.estimate()/60.0f);
	}
	
	if(bfgs.solutionConverged()){
	  printf("Optimizer solution converged and cannot improve the result further.\n");
	}
	else if(bfgs.isRunning() == false){
	  printf("Optimizer stopped running (early stopping/overfitting).\n");
	}
	  
	fflush(stdout);

	bfgs.stopComputation();
	
	// gets the final (optimum) solution
	bfgs.getSolution(w, error, iterations);
	
	if(nn->importdata(w) == false){
	  std::cout << "ERROR: internal error" << std::endl;
	  return -1;
	}
	
	if(bnn->importNetwork(*nn) == false){
	  std::cout << "ERROR: internal error" << std::endl;
	  return -1;
	}
      }
      
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "parallelbfgs"){

      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose){
	if(overfit == false){
	  if(secs > 0)
	    std::cout << "Starting parallel neural network BFGS optimization with early stopping (T=" << secs << " seconds, " << threads << " threads).."
		      << std::endl;
	  else
	    std::cout << "Starting parallel neural network BFGS optimization with early stopping (" << threads << " threads).."
		      << std::endl;
	}
	else{
	  if(secs > 0)
	    std::cout << "Starting parallel neural network BFGS optimization (T=" << secs << " seconds, " << threads << " threads).."
		      << std::endl;
	  else
	    std::cout << "Starting parallel neural network BFGS optimization (" << threads << " threads).."
		      << std::endl;
	}
      }

      if(secs <= 0 && samples <= 0){
	fprintf(stderr, "BFGS search requires --time or --samples command line switch.\n");
	return -1;
      }
      
      pBFGS_nnetwork< whiteice::math::blas_real<double> > bfgs(*nn, data, overfit, negfeedback);
      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::blas_real<double> error = 1000.0f;
	math::vertex< whiteice::math::blas_real<double> > w;
	unsigned int iterations = 0;
	whiteice::linear_ETA<double> eta;

	if(samples > 0)
	  eta.start(0.0f, (double)samples);

	// initial starting position
	// nn->exportdata(w);
	
	bfgs.minimize(threads);

	while(error > math::blas_real<double>(0.001f) &&
	      (counter < secs || secs <= 0) && // compute max SECS seconds
	      (iterations < samples || samples <= 0) && 
	      !stopsignal)
	{
	  sleep(5);

	  bfgs.getSolution(w, error, iterations);
	  
	  error = bfgs.getError(w);
	  
	  eta.update(iterations);

	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  if(secs > 0){
	    printf("\r                                                            \r");
	    printf("%d iters: %f [%.1f minutes]",
		   iterations, 
		   error.c[0], (secs - counter)/60.0f);
	  }
	  else{
	    printf("\r                                                            \r");
	    printf("%d/%d iters: %f [%.1f minutes]",
		   iterations, samples,  
		   error.c[0], eta.estimate()/60.0f);

	  }
	  fflush(stdout);
	}
	      
	
	if(secs > 0){
	  printf("\r                                                            \r");
	  printf("%d iters: %f [%.1f minutes]\n",
		 iterations,
		 error.c[0], (secs - counter)/60.0f);
	}
	else{
	  printf("\r                                                            \r");
	  printf("%d/%d iters: %f [%.1f minutes]\n",
		 iterations, samples,  
		 error.c[0], eta.estimate()/60.0f);
	}
	fflush(stdout);

	bfgs.stopComputation();
	
	// gets the final (optimum) solution
	bfgs.getSolution(w, error, iterations);
	
	if(nn->importdata(w) == false){
	  std::cout << "ERROR: internal error" << std::endl;
	  return -1;
	}
	if(bnn->importNetwork(*nn) == false){
	  std::cout << "ERROR: internal error" << std::endl;
	  return -1;
	}
      }
      
      
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "parallellbfgs"){

      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose){
	if(overfit == false){
	  if(secs > 0)
	    std::cout << "Starting parallel neural network L-BFGS optimization with early stopping (T=" << secs << " seconds, " << threads << " threads).."
		      << std::endl;
	  else
	    std::cout << "Starting parallel neural network L-BFGS optimization with early stopping (" << threads << " threads).."
		      << std::endl;
	}
	else{
	  if(secs > 0)
	    std::cout << "Starting parallel neural network L-BFGS optimization (T=" << secs << " seconds, " << threads << " threads).."
		      << std::endl;
	  else
	    std::cout << "Starting parallel neural network L-BFGS optimization (" << threads << " threads).."
		      << std::endl;
	}
      }

      if(secs <= 0 && samples <= 0){
	fprintf(stderr, "L-BFGS search requires --time or --samples command line switch.\n");
	return -1;
      }

      // FIXME add support for recursive neural networks
      pLBFGS_nnetwork< whiteice::math::blas_real<double> > bfgs(*nn, data, overfit, negfeedback);
      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::blas_real<double> error = 1000.0f;
	math::vertex< whiteice::math::blas_real<double> > w;
	unsigned int iterations = 0;
	whiteice::linear_ETA<double> eta;

	if(samples > 0)
	  eta.start(0.0f, (double)samples);

	// initial starting position
	// nn->exportdata(w);
	
	bfgs.minimize(threads);

	while(error > math::blas_real<double>(0.001f) &&
	      (counter < secs || secs <= 0) && // compute max SECS seconds
	      (iterations < samples || samples <= 0) && 
	      !stopsignal)
	{
	  sleep(5);

	  bfgs.getSolution(w, error, iterations);
	  
	  // error = bfgs.getError(w); // we already have the error
	  
	  eta.update(iterations);

	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  if(secs > 0){
	    printf("\r                                                            \r");
	    printf("%d iters: %f [%.1f minutes]",
		   iterations, 
		   error.c[0], (secs - counter)/60.0f);
	  }
	  else{
	    printf("\r                                                            \r");
	    printf("%d/%d iters: %f [%.1f minutes]",
		   iterations, samples,  
		   error.c[0], eta.estimate()/60.0f);

	  }

	  fflush(stdout);
	}
	      
	
	if(secs > 0){
	  printf("\r                                                            \r");
	  printf("%d iters: %f [%.1f minutes]\n",
		 iterations,
		 error.c[0], (secs - counter)/60.0f);
	}
	else{
	  printf("\r                                                            \r");
	  printf("%d/%d iters: %f [%.1f minutes]\n",
		 iterations, samples,  
		 error.c[0], eta.estimate()/60.0f);
	}
	  
	fflush(stdout);
	
	bfgs.stopComputation();

	// gets the final (optimum) solution
	bfgs.getSolution(w, error, iterations);
	
	if(nn->importdata(w) == false){
	  std::cout << "ERROR: internal error" << std::endl;
	  return -1;
	}
	
	if(bnn->importNetwork(*nn) == false){
	  std::cout << "ERROR: internal error" << std::endl;
	  return -1;
	}
	
      }
      
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "random"){

      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose)
	std::cout << "Starting neural network parallel random search (T=" << secs << " seconds, " << threads << " threads).."
		  << std::endl;

      if(secs <= 0){
	fprintf(stderr, "Random search requires --time TIME command line switch.\n");
	return -1;
      }

#if 0
      // hack to test ultradeep
      // NOTE: brute-forcing does not really work..
      {
	std::vector< math::vertex< whiteice::math::blas_real<double> > > input;
	std::vector< math::vertex< whiteice::math::blas_real<double> > > output;
	
	data.getData(0, input);
	data.getData(1, output);
	
	UltraDeep ud;
	
	while(1){
	  ud.calculate(input, output);
	}
	return 0;
      }
#endif
      
      math::NNRandomSearch< whiteice::math::blas_real<double> > search;
      search.startOptimize(data, arch, threads);

      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::blas_real<double> error = 100.0f;
	unsigned int solutions = 0;
	
	
	while(error > math::blas_real<double>(0.001f) &&
	      counter < secs && // compute max SECS seconds
	      !stopsignal) 
	{
	  search.getSolution(*nn, error, solutions);

	  sleep(1);
	  
	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  printf("\r                                                            \r");
	  printf("%d tries: %f [%.1f minutes]", solutions, error.c[0], (secs - counter)/60.0f);
	  fflush(stdout);
	}

	printf("\r                                                            \r");
	printf("%d tries: %f [%.1f minutes]\n", solutions, error.c[0], (secs - counter)/60.0f);
	fflush(stdout);

	search.stopComputation();
	
	// gets the final (optimum) solution
	if(search.getSolution(*nn, error, solutions) == false){
	  std::cout << "ERROR: Cannot get result from optimizer (internal error)." << std::endl;
	}

	if(bnn->importNetwork(*nn) == false){
	  std::cout << "ERROR: Cannot transfer neural network data (internal error)." << std::endl;
	}
      }
      

    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "parallelgrad"){
      
      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose)
	std::cout << "Starting neural network parallel multistart gradient descent (T=" << secs << " seconds, " << threads << " threads).."
		  << std::endl;
      
      if(secs <= 0){
	fprintf(stderr, "Parallel gradient descent requires --time TIME command line switch.\n");
	return -1;
      }
      
      
      math::NNGradDescent< whiteice::math::blas_real<double> > grad(negfeedback);

      if(samples > 0)
	grad.startOptimize(data, *nn, threads, samples);
      else
	grad.startOptimize(data, *nn, threads);

      
      {
	time_t t0 = time(0);
	unsigned int counter = 0;
	math::blas_real<double> error = 100.0f;
	unsigned int solutions = 0;
	
	
	while(counter < secs && !stopsignal) // compute max SECS seconds
	{
	  grad.getSolution(*nn, error, solutions);

	  sleepms(5000);
	  
	  time_t t1 = time(0);
	  counter = (unsigned int)(t1 - t0); // time-elapsed

	  printf("\r                                                            \r");
	  printf("%d tries: %f [%.1f minutes]", solutions, error.c[0], (secs - counter)/60.0f);
	  fflush(stdout);
	}

	printf("\r                                                            \r");
	printf("%d tries: %f [%.1f minutes]\n", solutions, error.c[0], (secs - counter)/60.0f);
	fflush(stdout);

	grad.stopComputation();

	// gets the final (optimum) solution
	grad.getSolution(*nn, error, solutions);
	bnn->importNetwork(*nn);
      }

      
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "grad"){
      
      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose){
	std::cout << "Starting neural network gradient descent optimizer.."
		  << std::endl;
	if(overfit == false)
	  std::cout << "Early stopping (testing dataset)." << std::endl;
      }
      

      {
	// divide data to training and testing sets
	dataset< whiteice::math::blas_real<double> > dtrain, dtest;
	
	dtrain = data;
	dtest  = data;
	
	dtrain.clearData(0);
	dtrain.clearData(1);
	dtest.clearData(0);
	dtest.clearData(1);
	
	for(unsigned int i=0;i<data.size(0);i++){
	  const unsigned int r = (rand() & 1);
	  
	  if(r == 0){
	    math::vertex< whiteice::math::blas_real<double> > in  = data.access(0,i);
	    math::vertex< whiteice::math::blas_real<double> > out = data.access(1,i);
	    
	    dtrain.add(0, in,  true);
	    dtrain.add(1, out, true);
	  }
	  else{
	    math::vertex< whiteice::math::blas_real<double> > in  = data.access(0,i);
	    math::vertex< whiteice::math::blas_real<double> > out = data.access(1,i);
	    
	    dtest.add(0, in,  true);
	    dtest.add(1, out, true);	    
	  }
	}
	
	// 1. normal gradient descent optimization using dtrain dataset
	// 2. after each iteration calculate the actual error terms from dtest dataset
	{
	  math::vertex< whiteice::math::blas_real<double> > grad, err, weights;
	  
	  math::vertex< whiteice::math::blas_real<double> > best_weights;
	  
	  unsigned int counter = 0;
	  math::blas_real<double> error, mean_ratio;
	  math::blas_real<double> prev_error;
	  math::blas_real<double> lrate = math::blas_real<double>(0.05f);
	  math::blas_real<double> delta_error = 0.0f;	  

	  math::blas_real<double> minimum_error = 10000000000.0f;
	  
	  std::list< math::blas_real<double> > ratios;
	  
	  error = 1000.0f;
	  prev_error = 1000.0f;
	  mean_ratio = 1.0f;
	  

	  math::vertex< whiteice::math::blas_real<double> > prev_sumgrad;

	  whiteice::linear_ETA<double> eta;
	  if(samples > 0)
	    eta.start(0.0f, (double)samples);
	  
	  const unsigned int SAMPLE_SIZE = 500;
	  
	  while(counter < samples && !stopsignal)
	  {
	    
	    while(ratios.size() > 10)
	      ratios.pop_front();
	    
	    math::blas_real<double> inv = 1.0f/ratios.size();
	    
	    mean_ratio = 1000.0f;
	    
	    for(auto& r : ratios) // min ratio of the past 10 iters
	      if(r < mean_ratio)
		mean_ratio = r;
	    
	    // mean_ratio = math::pow(mean_ratio, inv);
	    
	    if(overfit == false){
	      if(mean_ratio > 1.30f)
		if(counter > 10) break; // do not stop immediately
	    }
	    
	    prev_error = error;
	    error = 0.0f;

	    // goes through data, calculates gradient
	    // exports weights, weights -= lrate*gradient
	    // imports weights back

	    math::vertex< whiteice::math::blas_real<double> > sumgrad;
	    math::blas_real<double> ninv = 1.0f/SAMPLE_SIZE;

	    sumgrad.resize(nn->gradient_size());
	    sumgrad.zero();

//#pragma omp parallel shared(sumgrad)
	    {
	      nnetwork< math::blas_real<double> > net(*nn);
	      math::vertex< whiteice::math::blas_real<double> > sgrad(sumgrad.size());
	      sgrad.zero();

//#pragma omp for nowait schedule(dynamic)
	      for(unsigned int i=0;i<SAMPLE_SIZE;i++){
		const unsigned index = rand() % dtrain.size(0);
		
		net.input() = dtrain.access(0, index);
		net.calculate(true);
		err = dtrain.access(1, index) - net.output();
		
		if(net.gradient(err, grad) == false)
		  std::cout << "gradient failed." << std::endl;
		
		sgrad += ninv*grad;
	      }

//#pragma omp critical
	      {
		sumgrad += sgrad;
	      }
	    }
	    
	    
	    if(nn->exportdata(weights) == false){
	      std::cout << "FATAL: export failed." << std::endl;
	      exit(-1);
	    }

	    // adds regularizer to gradient descent
#if 0
	    {
	      whiteice::math::blas_real<double>
		regularizer = 1.0;
	      
	      sumgrad += regularizer*weights;
	    }
#endif
	    
	    lrate = 0.15;
	    math::vertex< whiteice::math::blas_real<double> > w;
	    
	    do{	      
	      lrate = 0.5*lrate;
	      w = weights;
	      
	      if(prev_sumgrad.size() <= 1){
		w -= lrate * sumgrad;
	      }
	      else{
		math::blas_real<double> momentum = 0.0f; // 0.8f
		w -= lrate * sumgrad + momentum*prev_sumgrad;
	      }
	      
	      nn->importdata(w);
	      
	      if(negfeedback){
		// using negative feedback heuristic
		math::blas_real<double> alpha = 0.5f;
		negative_feedback_between_neurons(*nn, dtrain, alpha);	      
	      }
	      
	      error = 0.0;

	      
	      // calculates error from the testing dataset
#pragma omp parallel shared(error)
	      {
		math::blas_real<double> e = 0.0;
		
#pragma omp for nowait schedule(dynamic)		
		for(unsigned int i=0;i<SAMPLE_SIZE;i++){
		  const unsigned int index = rand() % dtest.size(0);
		  auto input = dtest.access(0, index);
		  auto output = dtest.access(1, index);
		  
		  nn->calculate(input, output); // thread-safe
		  err = dtest.access(1, index) - output;
		  
		  e += (err*err)[0] / math::blas_real<double>((double)err.size());
		}

#pragma omp critical
		{
		  error += e;
		}
	      }
	      
	      
	      error /= SAMPLE_SIZE;
	      error *= math::blas_real<double>(0.5f); // missing scaling constant
	      
	      delta_error = (prev_error - error); // if the error is negative (error increases) we try again
	    }
	    while(delta_error < 0.0f && lrate > 10e-20);
	    

	    prev_sumgrad = lrate * sumgrad;
	    
	    if(error < minimum_error){
	      best_weights = w;
	      minimum_error = error;
	    }
	    else if((rand() & 0xFF) == 0xFF){ // on averare every 128th iteration reset back to known best solution
	      w = best_weights;
	      error = minimum_error;
	      prev_sumgrad.zero();
	    }
	    
	    math::blas_real<double> ratio = error / minimum_error;
	    ratios.push_back(ratio);	    

	    printf("\r                                                            \r");
	    printf("%d/%d iterations: %f (%f) [%.1f minutes]",
		   counter, samples, error.c[0], mean_ratio.c[0], eta.estimate()/60.0);
	    
	    fflush(stdout);
	    
	    counter++;
	    eta.update((double)counter);
	  }


	  
	  if(best_weights.size() > 1)
	    nn->importdata(best_weights);

	  printf("\r                                                            \r");
	  printf("%d/%d : %f (%f) [%.1f minutes]\n",
		 counter, samples, error.c[0], mean_ratio.c[0], eta.estimate()/60.0);
	  fflush(stdout);
	}
	
      }
      
      bnn->importNetwork(*nn);
      
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "bayes"){
      
      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      threads = 1;

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

#if 0
      // calculates error covariance matrix using current 
      // (random or loaded neural network configuration)
      math::matrix< whiteice::math::blas_real<double> > covariance;
      {
	covariance.resize(data.dimension(1), data.dimension(1));
	covariance.zero();
	
	math::vertex< whiteice::math::blas_real<double> > mean;
	mean.resize(data.dimension(1));
	mean.zero();
	
	for(unsigned int i=0;i<data.size(0);i++){
	  math::vertex< whiteice::math::blas_real<double> > out1;
	  math::matrix< whiteice::math::blas_real<double> > cov;
	  
	  bnn->calculate(data.access(0, i), out1, cov);
	  auto err = data.access(1, i) - out1;
	  
	  mean += err;
	  covariance += err.outerproduct();
	}

	mean /= whiteice::math::blas_real<double>(data.size(0));
	covariance /= whiteice::math::blas_real<double>(data.size(0));
	covariance -= mean.outerproduct();
      }
      
      std::cout << "covariance = " << covariance << std::endl;
#endif
      
      // whiteice::HMC_convergence_check< whiteice::math::blas_real<double> > hmc(*nn, data, adaptive);
      unsigned int ptlayers =
	(unsigned int)(math::log(data.size(0))/math::log(1.25));
      
      if(ptlayers <= 10) ptlayers = 10;
      else if(ptlayers > 100) ptlayers = 100;

      // std::cout << "Parallel Tempering depth: " << ptlayers << std::endl;

      // need for speed: (we downsample
      

      whiteice::HMC< whiteice::math::blas_real<double> > hmc(*nn, data, adaptive);
      // whiteice::UHMC< whiteice::math::blas_real<double> > hmc(*nn, data, adaptive);
      
      // whiteice::PTHMC< whiteice::math::blas_real<double> > hmc(ptlayers, *nn, data, adaptive);
      whiteice::linear_ETA<double> eta;
      
      time_t t0 = time(0);
      unsigned int counter = 0;
      
      hmc.startSampler();
      
      if(samples > 0)
	eta.start(0.0f, (double)samples);
      
      while(((hmc.getNumberOfSamples() < samples && samples > 0) || (counter < secs && secs > 0)) && !stopsignal){
      // while(!hmc.hasConverged() && !stopsignal){

	eta.update((double)hmc.getNumberOfSamples());
	
	if(hmc.getNumberOfSamples() > 0){
	  if(secs > 0){
	    printf("\r                                                            \r");
	    printf("%d samples: %f [%.1f minutes]",
		   hmc.getNumberOfSamples(),
		   hmc.getMeanError(1).c[0],		   
		   (secs - counter)/60.0);
	  }
	  else{
	    printf("\r                                                            \r");
	    printf("%d/%d samples : %f [%.1f minutes]",
		   hmc.getNumberOfSamples(),
		   samples,
		   hmc.getMeanError(1).c[0],
		   eta.estimate()/60.0);
	  }
	  fflush(stdout);
	}
	
	sleep(5);

	time_t t1 = time(0);
	counter = (unsigned int)(t1 - t0);
      }
      
      hmc.stopSampler();

      if(secs > 0){
	printf("\r                                                            \r");
	printf("%d samples : %f\n",
	       hmc.getNumberOfSamples(), hmc.getMeanError(100).c[0]);
      }
      else{
	printf("\r                                                            \r");
	printf("%d/%d samples : %f\n",
	       hmc.getNumberOfSamples(), samples, hmc.getMeanError(100).c[0]);
      }
      
      fflush(stdout);

      // nn->importdata(hmc.getMean());
      delete nn;
      nn = NULL;

      // saves samples to bayesian network data structure
      // (keeps only 50% latest samples) so we ignore initial tail
      // to high probability distribution
      {
	bnn = new bayesian_nnetwork< whiteice::math::blas_real<double> >();
	
	unsigned int savedSamples = 1;
	if(hmc.getNumberOfSamples() > 1){
	  savedSamples = hmc.getNumberOfSamples()/2;
	}
	
	assert(hmc.getNetwork(*bnn, savedSamples) == true);
      }

      // instead of using mean weight vector
      // we now use y = E[network(x,w)] in bayesian inference
      //
      // TODO: what we really want is
      //       the largest MODE of p(w|data) distribution as 
      //       this is then the global minima (assuming samples
      //       {w_i} converge to p(w|data)).
      
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    else if(lmethod == "minimize"){
      if(SIMULATION_DEPTH > 1){
	printf("ERROR: recurrent nnetwork not supported\n");
	exit(-1);
      }
      
      if(verbose){
	if(secs > 0){
	  std::cout << "Finding neural network input (genetic algorithms) with minimum response (T=" 
		    << secs << " seconds)"
		    << std::endl;
	}
	else{
	  std::cout << "Finding neural network input (genetic algorithms) with minimum response.."
		    << std::endl;
	}
      }
      
            
      if(bnn->load(nnfn) == false){
	std::cout << "Loading neural network failed." << std::endl;
	delete nn;
	delete bnn;
	nn = NULL;
	return -1;
      }
      
      // loads nnetwork weights from BNN
      {
	std::vector< math::vertex< math::blas_real<double> > > weights;
	nnetwork< whiteice::math::blas_real<double> > nnParam;
	
	if(bnn->exportSamples(nnParam, weights) == false){
	  std::cout << "Loading neural network failed." << std::endl;
	  delete nn;
	  delete bnn;
	  nn = NULL;
	  return -1;
	}

	if(weights.size() == 0){
	  std::cout << "Loading neural network failed." << std::endl;
	  delete nn;
	  delete bnn;
	  nn = NULL;
	  return -1;
	}
	
	delete nn;
	nn = new nnetwork< whiteice::math::blas_real<double> >(nnParam);

	*nn = nnParam;
	nn->importdata(weights[(rand() % weights.size())]);;
      }
      
      nnetwork_function< whiteice::math::blas_real<double> > nf(*nn);
      GA3< whiteice::math::blas_real<double> > ga(&nf);

      time_t t0 = time(0);
      unsigned int counter = 0;
      
      ga.minimize();
      
      whiteice::math::vertex< whiteice::math::blas_real<double> > s;
      math::blas_real<double> r;
      
      while(((ga.getGenerations() < samples && samples > 0) || (counter < secs && secs > 0)) && !stopsignal){
	r = ga.getBestSolution(s);
	const unsigned int g = ga.getGenerations();
	
	if(secs > 0){
	  printf("\r                                                            \r");
	  printf("%d generations: %f [%.1f minutes]",
		 g, r.c[0],
		 (secs - counter)/60.0);
	}
	else{
	  printf("\r                                                            \r");
	  printf("%d/%d generations : %f",
		 g, samples, r.c[0]);
	}
	fflush(stdout);
	
	sleep(1);
	
	time_t t1 = time(0);
	counter = (unsigned int)(t1 - t0);
      }
      
      printf("\n");
      
      data.invpreprocess(0, s);
      std::cout << "Best solution found (" << r << "): " << s << std::endl;
      
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
      
      
      if(bnn->inputSize() != data.dimension(0) && SIMULATION_DEPTH == 1){
	std::cout << "Neural network input dimension mismatch for input dataset ("
		  << bnn->inputSize() << " != " << data.dimension(0) << ")"
		  << std::endl;
	delete bnn;
	delete nn;
	nn = NULL;
	return -1;
      }
      else if(bnn->inputSize() != data.dimension(0)+bnn->outputSize() && SIMULATION_DEPTH > 1){
	std::cout << "Recurrent neural network input dimension mismatch for input dataset ("
		  << bnn->inputSize() << " != " << data.dimension(0)+bnn->outputSize() << ")"
		  << std::endl;
	delete bnn;
	delete nn;
	nn = NULL;
	return -1;
      }
      
      
      
      bool compare_clusters = false;
      
      if(data.getNumberOfClusters() == 2){
	if(data.size(0) > 0 && data.size(1) > 0 && 
	   data.size(0) == data.size(1)){
	  compare_clusters = true;
	}
	  
	if(bnn->outputSize() != data.dimension(1)){
	  std::cout << "Neural network output dimension mismatch for dataset ("
		    << bnn->outputSize() << " != " << data.dimension(1) << ")"
		    << std::endl;
	  delete bnn;
	  delete nn;
	  return -1;	    
	}
      }
      else if(data.getNumberOfClusters() == 3){
	if(data.size(0) > 0 && data.size(1) > 0 && 
	   data.size(0) == data.size(1)){
	  compare_clusters = true;
	}
	
	if(bnn->outputSize() != data.dimension(1)){
	  std::cout << "Neural network output dimension mismatch for dataset ("
		    << bnn->outputSize() << " != " << data.dimension(1) << ")"
		    << std::endl;
	  delete bnn;
	  delete nn;
	  return -1;	    
	}

	if(bnn->outputSize() != data.dimension(2)){
	  std::cout << "Neural network output dimension mismatch for dataset ("
		    << bnn->outputSize() << " != " << data.dimension(2) << ")"
		    << std::endl;
	  delete bnn;
	  delete nn;
	  return -1;	    
	}
      }
      else{
	std::cout << "Unsupported number of data clusters in dataset: "
		  << data.getNumberOfClusters() << std::endl;
	delete bnn;
	delete nn;
	return -1;	    
      }
	
      
      
      if(compare_clusters == true){
	math::blas_real<double> error1 = math::blas_real<double>(0.0f);
	math::blas_real<double> error2 = math::blas_real<double>(0.0f);
	math::blas_real<double> c = math::blas_real<double>(0.5f);
	math::vertex< whiteice::math::blas_real<double> > err;
	
	whiteice::nnetwork< whiteice::math::blas_real<double> > single_nn(*nn);
	std::vector< math::vertex< whiteice::math::blas_real<double> > > weights;
	
	bnn->exportSamples(single_nn, weights);
	math::vertex< whiteice::math::blas_real<double> > w = weights[0];
	w.zero();

	for(auto& wi : weights)
		w += wi;

	w /= weights.size(); // E[w]

	
	{
	  std::vector<unsigned int> arch2;
	  single_nn.getArchitecture(arch2);

	  if(arch2.size() != arch.size()){
	    printf("ERROR: cannot import weights from bayesian nnetwork to a single network (mismatch network layout %d != %d).\n",
		   (int)arch2.size(), (int)arch.size());
	    delete bnn;
	    delete nn;
	    exit(-1);
	  }
	  
	  for(unsigned int i=0;i<arch.size();i++){
	    if(arch2[i] != arch[i]){
	      printf("ERROR: cannot import weights from bayesian nnetwork to a single network (mismatch network layout).\n");
	      for(unsigned int i=0;i<arch.size();i++)
		printf("%d ", arch[i]);
	      printf("\n");

	      for(unsigned int i=0;i<arch2.size();i++)
		printf("%d ", arch2[i]);
	      printf("\n");
	      
	      delete bnn;
	      delete nn;
	      exit(-1);
	    }
	  }
	}

	
	if(single_nn.importdata(w) == false){
	  printf("ERROR: cannot import weights from bayesian nnetwork to a single network.\n");
	  delete bnn;
	  delete nn;
	  exit(-1);
	}
	

	whiteice::linear_ETA<double> eta;
	
	if(data.size(0) > 0)
	  eta.start(0.0f, (double)data.size(0));

	unsigned int counter = 0; // number of points calculated..

	for(unsigned int i=0;i<data.size(0) && !stopsignal;i++){
	  math::vertex< whiteice::math::blas_real<double> > out1;
	  math::matrix< whiteice::math::blas_real<double> > cov;

	  bnn->calculate(data.access(0, i), out1, cov, SIMULATION_DEPTH);
	  err = data.access(1,i) - out1;
	  
	  for(unsigned int i=0;i<err.size();i++)
	    error1 += c*(err[i]*err[i]) / math::blas_real<double>((double)err.size());

	  single_nn.input().zero();
	  single_nn.output().zero();
	  single_nn.input().write_subvertex(data.access(0, i), 0);	  
	  
	  for(unsigned int d=0;d<SIMULATION_DEPTH;d++){
	    if(SIMULATION_DEPTH > 1)
	      single_nn.input().write_subvertex(single_nn.output(), data.access(0, i).size());
	    single_nn.calculate(false, false);
	  }
	  
	  err = data.access(1, i) - single_nn.output();

	  for(unsigned int i=0;i<err.size();i++)
	    error2 += c*(err[i]*err[i]) / math::blas_real<double>((double)err.size());

	  eta.update((double)(i+1));

	  double percent = 100.0f*((double)i)/((double)data.size(0));
	  double etamin  = eta.estimate()/60.0f;

	  printf("\r                                                            \r");
	  printf("%d/%d (%.1f%%) [ETA: %.1f minutes]", i, data.size(0), percent, etamin);
	  fflush(stdout);

	  counter++;
	}

	printf("\n"); fflush(stdout);

	if(counter > 0){
	  error1 /= math::blas_real<double>((double)counter);
	  error2 /= math::blas_real<double>((double)counter);
	}
	
	std::cout << "Average error in dataset (E[f(x|w)]): " << error1 << std::endl;
	std::cout << "Average error in dataset (f(x|E[w])): " << error2 << std::endl;
      }
      
      else{
	std::cout << "Predicting data points.." << std::endl;
	
	if(data.getNumberOfClusters() == 2 && data.size(0) > 0){
	  
	  data.clearData(1);
	  
	  data.setName(0, "input");
	  data.setName(1, "output");

	  whiteice::linear_ETA<double> eta;
	  
	  if(data.size(0) > 0)
	    eta.start(0.0f, (double)data.size(0));
	
	  for(unsigned int i=0;i<data.size(0) && !stopsignal;i++){
	    math::vertex< whiteice::math::blas_real<double> > out;
	    math::vertex< whiteice::math::blas_real<double> > var;
	    math::matrix< whiteice::math::blas_real<double> > cov;

	    eta.update((double)i);

	    double percent = 100.0 * ((double)(i+1))/((double)data.size(0));
	    double etamin  = eta.estimate()/60.0f;

	    printf("\r                                                            \r");
	    printf("%d/%d (%.1f%%) [ETA %.1f minutes]", i+1, data.size(0), percent, etamin);
	    fflush(stdout);
	    
	    bnn->calculate(data.access(0, i),  out, cov, SIMULATION_DEPTH);
	    
	    // we do NOT preprocess the output but inject it directly into dataset
	    data.add(1, out, true);
	  }

	  printf("\n");
	  fflush(stdout);	  
	}
	else if(data.getNumberOfClusters() == 3 && data.size(0) > 0){
	  
	  data.clearData(1);
	  data.clearData(2);
	  
	  data.setName(0, "input");
	  data.setName(1, "output");
	  data.setName(2, "output_stddev");
	  
	  for(unsigned int i=0;i<data.size(0) && !stopsignal;i++){
	    math::vertex< whiteice::math::blas_real<double> > out;
	    math::vertex< whiteice::math::blas_real<double> > var;
	    math::matrix< whiteice::math::blas_real<double> > cov;
	    
	    bnn->calculate(data.access(0, i), out, cov, SIMULATION_DEPTH);
	    
	    // we do NOT preprocess the output but inject it directly into dataset
	    data.add(1, out, true);

	    var.resize(cov.xsize());	    
	    for(unsigned int j=0;j<cov.xsize();j++)
	      var[j] = math::sqrt(cov(j,j)); // st.dev.
	    
	    data.add(2, var, true);
	  }
	}

	if(stopsignal){
	  if(bnn) delete bnn;
	  if(nn)  delete nn;
	  
	  exit(-1);
	}
	
	if(data.save(datafn) == true)
	  std::cout << "Storing results to dataset file: " 
		    << datafn << std::endl;
	else
	  std::cout << "Storing results to dataset file FAILED." << std::endl;
      }
    }
    
        
    if(lmethod != "use" && lmethod != "minimize"){
      if(bnn){
	if(bnn->save(nnfn) == false){
	  std::cout << "Saving neural network data failed." << std::endl;
	  delete bnn;
	  return -1;
	}
	else{
	  if(verbose)
	    std::cout << "Saving neural network data: " << nnfn << std::endl;
	}
      }
    }
    
    
    if(bnn){ delete bnn; bnn = 0; }
    if(nn){ delete nn; nn = 0; }
    
    
    return 0;
  }
  catch(std::exception& e){
    std::cout << "Fatal error: unexpected exception. Reason: " 
	      << e.what() << std::endl;
    return -1;
  }
  
}


void sigint_signal_handler(int s)
{
  stopsignal = true;
}


void install_signal_handler()
{
#ifndef WINOS
  struct sigaction sih;
  
  sih.sa_handler = sigint_signal_handler;
  sigemptyset(&sih.sa_mask);
  sih.sa_flags = 0;

  sigaction(SIGINT, &sih, NULL);
#endif
}


void sleepms(unsigned int ms)
{
#ifndef WINNT
  struct timespec ts;
  ts.tv_sec  = 0;
  ts.tv_nsec = ms*1000000;
  nanosleep(&ts, 0);
#else
  Sleep(ms);
#endif 
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
  printf("--no-init      do not use heuristics when initializing nn weights\n");
  printf("--overfit      do not use early stopping (bfgs,lbfgs)\n");
  printf("--deep         pretrains feedforward neural network weights using stacked RBMs (slow)\n");
  printf("--pseudolinear sets nonlinears to be 50%% linear (--purelinear is 100%% linear)\n");
  printf("--recurrent N  simple recurrent network (simulates N steps) (lbfgs, use)\n");
  printf("--adaptive     use adaptive step length in bayesian hamiltonian monte carlo (bayes)\n");
  printf("--negfb        use negative feedback between neurons (grad,parallelgrad,bfgs,lbfgs)\n");
  printf("--load         use previously computed network weights as the starting point (grad,bfgs,lbfgs,bayes)\n");
  printf("--time TIME    sets time limit for multistart optimization and bayesian inference\n");
  printf("--samples N    samples N samples or defines max iterations (eg. 2500) to be used in optimization/sampling\n");
  printf("--threads N    uses N parallel threads when looking for solution\n");
  printf("--data N       takes randomly N samples of data for the learning process (N/2 used in training)\n");
  printf("[data]         a source file for inputs or i/o examples (binary file)\n");
  printf("               (whiteice data file format created by dstool)\n");
  printf("[arch]         the architecture of a new nn. Eg. 3-10-9 or ?-10-?\n");
  printf("<nnfile>       input/output neural networks weights file\n");
  printf("[lmethod]      method: use, random, grad, parallelgrad, bayes, lbfgs, parallelbfgs, parallellbfgs, gbrbm, bbrbm, mix\n");
  printf("               parallel methods use random location multistart/restart parallel search\n");
  printf("               until timeout or the number of samples has been reached\n");
  printf("               additionally: minimize method finds input that minimizes the neural network output\n");
  printf("               gradient descent algorithms can use negative feedback heuristic\n");
  printf("               mix uses mixture of neural networks\n");
  printf("\n");
  printf("               Ctrl-C shutdowns the program.\n");
  printf("\n");
  printf("This program is distributed under GPL license <tomas.ukkonen@iki.fi> (commercial license available).\n");
  
}


