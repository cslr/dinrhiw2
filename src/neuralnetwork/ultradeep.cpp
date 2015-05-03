

#include "ultradeep.h"

#include "eig.h"
#include "correlation.h"
#include "linear_algebra.h"
#include "linear_equations.h"
#include "ica.h"

#include <map>

namespace whiteice
{
  
  UltraDeep::UltraDeep(){
    
  }
  
  
  // tries to find better solutions
  float UltraDeep::calculate(const std::vector< math::vertex<> >& input,
			     const std::vector< math::vertex<> >& output)
  {
    math::blas_real<float> best_error = 10e9;
    math::blas_real<float> best_combined = 10e9;
    unsigned int deepness = 0;
    
    params.clear();

    // checks what is the error if we compute linear 
    // fitting output = A*x + b at this time step
    {
      math::matrix<> A;
      math::vertex<> b;
      
      if(calculate_linear_fit(input, output, A, b) == false)
	return false;
      
      math::blas_real<float> error = 0.0f;
      math::vertex<> err;
      
      for(unsigned int i=0;i<input.size();i++){
	err = output[i] - A*input[i] - b;
	error += 0.5f*(err*err)[0];
      }
      
      error /= input.size();
      
      std::cout << "deepness: " << deepness << " error: " << error << std::endl;
      
      if(error < best_error)
	best_error = error;
      
      math::vertex<> kurtosis = calculateKurtosis(input);
      
      math::blas_real<float> kurt = 0.0f;
      for(unsigned int i=0;i<kurtosis.size();i++)
	kurt += kurtosis[i]/kurtosis.size();
      
      best_combined = best_error;
    }
    
    
    math::blas_real<float> orig_error    = best_error;
    math::blas_real<float> try_error     = 10e9;
    math::blas_real<float> iter_combined = 10e9;
    unsigned int iters = 0;
    

    // number of dimensions to use in the network
    const unsigned int DIMENSIONS = input[0].size();
    

    math::vertex<> d;
    math::vertex<> b;
    math::vertex<> best_d;
    math::vertex<> best_b;
    
    d.resize(DIMENSIONS);
    b.resize(DIMENSIONS);
    best_d.resize(DIMENSIONS);
    best_b.resize(DIMENSIONS);
    
    std::vector< math::vertex<> > data = input;
    std::vector< math::vertex<> > pca_data;    
    
    bool hasPCA = false;
    
    unsigned int no_improvement = 0;
    
    std::multimap<math::blas_real<float>, math::vertex<> > samples;

    
    while(deepness < 100){ // no_improvement < 1000
      auto selected = goodness.end();

#if 0      
      // prints the best result
      {
	std::cout << "d = " << best_d << std::endl;
	std::cout << "b = " << best_b << std::endl;
      }
#endif
      
      if((rand()&7) != 0 && goodness.size() > 0){
	selected = geneticAlgorithmSelect(b, d);
      }
      else{
	for(unsigned int i=0;i<d.size();i++){
	  d[i] = 4.0f*((((float)rand())/RAND_MAX) - 2.0f);
	  b[i] = 4.0f*((((float)rand())/RAND_MAX) - 2.0f);
	}
      }
      
      if(hasPCA == false){
	pca_data.clear();
	if(calculatePCA(data, pca_data, DIMENSIONS) == false)
	  return -1.0f;
	
	hasPCA = true;
      }

      // we first try with a small sample whether this combination looks good
      std::vector< math::vertex<> > processed_data;
#if 0
      {
	const unsigned int SAMPLE_SIZE = 100;
	
	for(unsigned int i=0;i<SAMPLE_SIZE;i++){
	  const unsigned int index = rand() % pca_data.size();
	  processed_data.push_back(pca_data[index]);
	}
	
	processNNStep(processed_data, b, d);
	      
      
	math::blas_real<float> error = 0.0f;
	error = calculateModelError(processed_data, output);
	
	// finds out the worst acceptable value
	if(goodness.size() >= 100){
	  auto i = goodness.rbegin().base();
	  i--;
	  
	  if(error > i->first){
	    std::cout << "EARLY ABORT" << std::endl;
	    std::cout << "error = " << error << " worst case: " << i->first << std::endl;
	    std::cout << "goodness.size = " << goodness.size() << std::endl;
	    continue; // do not look like a good solution [try another one]
	  }
	}
      }
#endif
      
      
      processed_data = pca_data;
      processNNStep(processed_data, b, d);
      
      
      math::blas_real<float> error = 0.0f;
      error = calculateModelError(processed_data, output);
      
      math::vertex<> kurtosis = calculateKurtosis(processed_data);
      math::blas_real<float> kurt = 0.0f;
      for(unsigned int i=0;i<kurtosis.size();i++)
	kurt += kurtosis[i]/kurtosis.size();
      
      auto combined = error;
      

      {
	math::vertex<> p;
	std::pair< math::blas_real<float>, math::vertex<> > tuple;
	
	p.resize(b.size() + d.size());
	
	p.importData(&(b[0]), b.size(), 0);
	p.importData(&(d[0]), d.size(), b.size());
	
	tuple.first  = combined;
	tuple.second = p;
	
	samples.insert(tuple);
	
	// stores error values
	if(selected == goodness.end()){ // randomly generated
	  goodness.insert(tuple);
	}
	else{ // previous solution
	  if(selected->first > error){ // error became smaller (error)
	    goodness.erase(selected);
	    goodness.insert(tuple);
	  }
	}


	
	std::cout << "deepness: " << deepness 
		  << " error: " << error 
		  << " kurtosis: " << kurt 
		  << " ratio1: " << best_error/orig_error 
		  << " ratio2: " << try_error/best_error
		  << " iters: " << iters << std::endl;
	
	// best b and d found in this layer so far
	if(error < try_error){
	  try_error = error;
	}
	else{

	}
	
	if(combined < iter_combined){
	  iter_combined = combined;
	  
	  best_d = d;
	  best_b = b;
	  no_improvement = 0;
	}
	else{
	  no_improvement++;
	}
	
	
	if(combined < 0.90f*best_combined){ // at least 10% improvement to the previous result
	  best_combined = combined;
	  best_error = error;
	}
	else{
	  iters++;
	  continue; 
	}	
      }
      
      goodness.clear();
      samples.clear();
      
      
      struct UltraDeep::ultradeep_parameters q;
      q.d = d;
      q.b = b;
      
      params.push_back(q);
      
      
      hasPCA = false;
      try_error = 10e9;
      no_improvement = 0;
      data = processed_data;
      
      deepness++;
      iters++;
    }
    
    return true;

  }
  
  
  bool UltraDeep::calculatePCA(const std::vector< math::vertex<> >& data,
			       std::vector< math::vertex<> >& ica_data,
			       const unsigned int DIMENSIONS)
  {
    math::vertex<> m;
    math::matrix<> Cxx;
    math::matrix<> V;
    
    if(math::mean_covariance_estimate(m, Cxx, data) == false)
      return false;
    
    if(math::symmetric_eig<>(Cxx, V) == false)
      return false;
    
    math::matrix<>& D = Cxx;
    
    for(unsigned int i=0;i<D.ysize();i++){
      if(D(i,i) < 0.0f) D(i,i) = math::abs(D(i,i));
      if(D(i,i) < 10e-8) D(i,i) = 0.0f;
      else{
	// sets diagonal variances
	math::blas_real<float> d = D(i,i);
	math::blas_real<float> s = 1.0f;
	
	D(i,i) =s/sqrt(d);
      }
    }
    
    // we keep only top max(10, ouputdims) dimensions in order to ultradeep efficient
    
    D.resize(DIMENSIONS,DIMENSIONS);
    V.transpose();
    V.resize_y(DIMENSIONS);


    math::matrix<>& PCA = D;
    PCA *= V;
    
    std::vector< math::vertex<> >& pca_data = ica_data;
    
    for(unsigned int i=0;i<data.size();i++){
      math::vertex<> u = PCA*(data[i] - m);
      pca_data.push_back(u);
    }
    
#if 0
    // calculates ICA too
    math::matrix<> ICA;
    whiteice::math::ica(pca_data, ICA);
    
    for(unsigned int i=0;i<data.size();i++){
      math::vertex<> u = ICA*data[i];
      ica_data.push_back(u);
    }
#endif
    
    return true;
  }
  
  
  std::multimap< math::blas_real<float>, math::vertex<> >::iterator 
  UltraDeep::geneticAlgorithmSelect(math::vertex<>& b, math::vertex<>& d)
  {
    auto selected = goodness.end();
    
  
    // alters one of the best solutions found
    {
      // keeps only the 100 best results
	
      if(goodness.size() > 200){
	while(goodness.size() > 100){
	  auto i = goodness.rbegin().base();
	  i--;
	  goodness.erase(i); // removes worst cases
	  }
      }
      
      bool mutation = rand() & 1;
      
      if(mutation){
	// takes random best element (only top 5 mutate)
	unsigned int index = rand() % (goodness.size() > 5 ? 5 : goodness.size());
	auto i = goodness.begin();
	
	while(index > 0){
	  i++;
	  index--;
	}
	
	selected = i;
	auto p = i->second;
	
	p.exportData(&(b[0]), b.size(), 0);
	p.exportData(&(d[0]), d.size(), b.size());
	
#if 0
	// moves each dimension -5% .. +5%
	
	for(unsigned int i=0;i<d.size();i++){
	  auto r = ((float)rand())/RAND_MAX;
	  r = 0.1f*r + 0.95; // [0.95,0.05];
	  
	  b[i] *= r;
	  
	  r = ((float)rand())/RAND_MAX;
	  r = 0.1f*r + 0.95; // [0.95,0.05];
	  
	  d[i] *= r;
	}
#endif
	// changes just one parameter
	
	if(rand()&1){
	  const unsigned int index = rand() % b.size();
	  auto r = ((float)rand())/RAND_MAX;
	  r = 0.1f*r + 0.95; // [0.95,0.05];
	  b[index] *= r;
	}
	else{
	  const unsigned int index = rand() % d.size();
	  auto r = ((float)rand())/RAND_MAX;
	  r = 0.1f*r + 0.95; // [0.95,0.05];
	  d[index] *= r;
	}
	
      }
      else{ // crossover
	
	// takes random best elements, the another one belongs always to top 5
	unsigned int index1 = rand() % (goodness.size() > 5 ? 5 : goodness.size());
	unsigned int index2 = rand() % goodness.size();
	auto i = goodness.begin();
	auto j = goodness.begin();
	
	while(index1 > 0){ i++; index1--; }
	while(index2 > 0){ j++; index2--; }
	
	// selects the worst one
	if(i->first > j->first)
	  selected = i;
	else
	  selected = j;
	
	auto p1 = i->second;
	auto p2 = j->second;
	
	math::vertex<> p;
	p.resize(p1.size());
	
	if((rand()&1) == 0){ // blending
	  math::blas_real<float> r = ((float)rand())/RAND_MAX;
	  math::blas_real<float> q = math::blas_real<float>(1.0f) - r;
	  
	  p = r*p1 + q*p2;
	}
	else{ // cut
	  for(unsigned int i=0;i<p.size();i++){
	    auto r = rand()&1;
	    if(r & 1) p[i] = p1[i];
	    else p[i] = p2[i];
	  }
	}
	
	
	p.exportData(&(b[0]), b.size(), 0);
	p.exportData(&(d[0]), d.size(), b.size());
      }
    }
    
    return selected;
  }
    

  
  
  bool UltraDeep::calculate_linear_fit(const std::vector< math::vertex<> > input,
				       const std::vector< math::vertex<> > output,
				       math::matrix<>& A,
				       math::vertex<>& b)
  {
    // NOTE: do not handle singular values properly
    
    try{
      math::matrix<> Cxx, Cxy;
      math::vertex<> mx, my;
      
      const unsigned int N = input.size();
      math::blas_real<float> inv = 1.0f/(float)N;
      
      Cxx.resize(input[0].size(),input[0].size());
      Cxy.resize(input[0].size(),output[0].size());
      mx.resize(input[0].size());
      my.resize(output[0].size());
      
      Cxx.zero();
      Cxy.zero();
      mx.zero();
      my.zero();
      
      for(unsigned int i=0;i<N;i++){
	Cxx += input[i].outerproduct();
	Cxy += input[i].outerproduct(output[i]);
	mx  += input[i];
	my  += output[i];
      }
      
      Cxx *= inv;
      Cxy *= inv;
      mx  *= inv;
      my  *= inv;
      
      Cxx -= mx.outerproduct();
      Cxy -= mx.outerproduct(my);
      
      math::matrix<> INV;
      math::blas_real<float> l = 10e-3;
      
      do{
	INV = Cxx;
	
	math::blas_real<float> trace = 0.0f;
	
	for(unsigned int i=0;i<Cxx.xsize();i++){
	  trace += Cxx(i,i);
	  INV(i,i) += l; // regularizes Cxx (if needed)
	}
	
	trace /= Cxx.xsize();
	
	l = trace + 2.0f*l;
      }
      while(symmetric_inverse(INV) == false);

      
      A = Cxy.transpose() * INV;
      b = my - A*mx;
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception: " << e.what() << std::endl;
      return false;
    }
    
    return true;
  }    
  

  void UltraDeep::processNNStep(std::vector< math::vertex<> >& data,
				const math::vertex<>& b, const math::vertex<>& d)
  {
    for(unsigned int i=0;i<data.size();i++){
      for(unsigned int j=0;j<data[i].size();j++)
	data[i][j] *= d[j];
      
      data[i] += b;
      
      for(unsigned int j=0;j<data[i].size();j++)
	// data[i][j] = cbrt(data[i][j].c[0]);
	data[i][j] = math::asinh(data[i][j]);
    }
  }

  
  math::blas_real<float> UltraDeep::calculateModelError(const std::vector< math::vertex<> >& data,
							const std::vector< math::vertex<> >& output)
  {
    // checks what is the error if we compute linear 
    // fitting output = A*x + b after this layer
    
    math::matrix<> AA;
    math::vertex<> bb;

    math::blas_real<float> error = 0.0f;
    
    if(calculate_linear_fit(data, output, AA, bb) == false){
      error = -1.0f;
      return error;
    }
    
    math::vertex<> err;
    
    for(unsigned int i=0;i<data.size();i++){
      err = output[i] - AA*data[i] - bb;
      error += 0.5f*(err*err)[0];
    }
    
    error /= data.size();
    
    return error;
  }
  
  
  math::vertex<> UltraDeep::calculateKurtosis(const std::vector< math::vertex<> >& data)
  {
    math::vertex<> kurtosis;
    math::vertex<> mean;
    math::vertex<> var;
    math::vertex<> moment4;
    
    if(data.size() > 0){
      kurtosis.resize(data[0].size());
      mean.resize(data[0].size());
      moment4.resize(data[0].size());
      var.resize(data[0].size());
      var.zero();
      moment4.zero();
      mean.zero();
      kurtosis.zero();
    }
    
    for(unsigned int j=0;j<data.size();j++)
      mean += data[j];
    
    mean /= data.size();

    for(unsigned int j=0;j<data.size();j++){
      auto v = data[j] - mean;
      
      for(unsigned int i=0;i<mean.size();i++)
	v[i] = v[i]*v[i];
      
      var += v;
      
      for(unsigned int i=0;i<mean.size();i++)
	v[i] = v[i]*v[i];
      
      moment4 += v;
    }
    
    var /= data.size();
    moment4 /= data.size();
    
    // absolute value from 3 for measuring general non-gaussianity
    for(unsigned int i=0;i<mean.size();i++){
      if(var[i] != 0.0f)
	kurtosis[i] = 
	  whiteice::math::abs(moment4[i]/(var[i]*var[i])- math::blas_real<float>(3.0f));
    }
    
    return kurtosis;
  }

}
