

#include "ultradeep.h"

#include "eig.h"
#include "correlation.h"
#include "linear_algebra.h"
#include "linear_equations.h"
#include "ica.h"

#include <map>

namespace whiteice
{
  
  
  bool calculate_linear_fit(const std::vector< math::vertex<> > input,
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
  
  


  bool ultradeep(const std::vector< math::vertex<> >& input,
		 std::vector< ultradeep_parameters >& params,
		 const std::vector< math::vertex<> >& output)
  {
    math::blas_real<float> best_error = 10e9;
    unsigned int deepness = 0;

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
    }
    
    
    math::blas_real<float> orig_error = best_error;
    math::blas_real<float> try_error  = 10e9;
    unsigned int iters = 0;
    

    std::multimap< math::blas_real<float>, math::vertex<> > goodness;
    
    
    // number of dimensions to use in the network
    unsigned int DIMENSIONS = input[0].size();
    
#if 0
    if(input[0].size() > 50)
      DIMENSIONS = input[0].size()/2;
    
    if(DIMENSIONS < output[0].size())
      DIMENSIONS = output[0].size();
#endif
    

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
    bool crossover = false;
    
    while(deepness < 100){ // no_improvement < 1000
      auto selected = goodness.end();
      
      // alters one of the best solutions found
      if((rand()&7) != 0 && goodness.size() > 0){
	crossover = false;
	
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
	  
	  // moves each dimension -5% .. +5%
	  
	  for(unsigned int i=0;i<d.size();i++){
	    auto r = ((float)rand())/RAND_MAX;
	    r = 0.1f*r + 0.95; // [0.95,0.05];
	    
	    b[i] *= r;
	    
	    r = ((float)rand())/RAND_MAX;
	    r = 0.1f*r + 0.95; // [0.95,0.05];
	    
	    d[i] *= r;
	  }
	}
	else{ // crossover
	  crossover = true;
	  
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
	  p.resize(DIMENSIONS);
	  
	  if((rand()&1) == 0){ // blending
	    math::blas_real<float> r = ((float)rand())/RAND_MAX;
	    math::blas_real<float> q = math::blas_real<float>(1.0f) - r;
	    
	    p = r*p1 + q*p2;
	  }
	  else{ // cut
	    p = p1;
	    const unsigned cut = rand()%p.size();
	    
	    for(unsigned int i=0;i<p.size();i++){
	      if(i < cut) p[i] = p1[i];
	      else p[i] = p2[i];
	    }
	  }
	  
	  
	  p.exportData(&(b[0]), b.size(), 0);
	  p.exportData(&(d[0]), d.size(), b.size());
	  
	  // moves each dimension -5% .. +5%
	  
	  for(unsigned int i=0;i<d.size();i++){
	    auto r = ((float)rand())/RAND_MAX;
	    r = 0.1f*r + 0.95; // [0.95,0.05];
	    
	    b[i] *= r;
	    
	    r = ((float)rand())/RAND_MAX;
	    r = 0.1f*r + 0.95; // [0.95,0.05];
	    
	    d[i] *= r;
	  }
	}
	  
      }
      else{
	for(unsigned int i=0;i<d.size();i++){
	  d[i] = 2.0f*((((float)rand())/RAND_MAX) - 0.5f);
	  b[i] = 2.0f*((((float)rand())/RAND_MAX) - 0.5f);
	}
      }
      
      if(hasPCA == false){
	pca_data.clear();
	
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
	    math::blas_real<float> s = 2.0f;
	    
	    D(i,i) =s/sqrt(d);
	  }
	}
	
	// we keep only top max(10, ouputdims) dimensions in order to ultradeep efficient
	
	D.resize(DIMENSIONS,DIMENSIONS);
	V.transpose();
	V.resize_y(DIMENSIONS);
	
	math::matrix<>& PCA = D;
	PCA *= V;
	
	
	for(unsigned int i=0;i<data.size();i++){
	  math::vertex<> u = PCA*(data[i] - m);
	  pca_data.push_back(u);
	}
	
	hasPCA = true;
      }
      
      std::vector< math::vertex<> > test_data = pca_data;
      
      
      {
	for(unsigned int i=0;i<test_data.size();i++){
	  for(unsigned int j=0;j<test_data[i].size();j++)
	    test_data[i][j] *= d[j];
	  
	  
	  test_data[i] += b;
	  
	  for(unsigned int j=0;j<test_data[i].size();j++)
	    test_data[i][j] = math::asinh(test_data[i][j]);
	}
      }
	
      
      // checks what is the error if we compute linear 
      // fitting output = A*x + b after this layer
      {
	math::matrix<> AA;
	math::vertex<> bb;
	
	if(calculate_linear_fit(test_data, output, AA, bb) == false){
	  iters++;
	  continue;
	}
	
	math::blas_real<float> error = 0.0f;
	math::vertex<> err;
	
	for(unsigned int i=0;i<test_data.size();i++){
	  err = output[i] - AA*test_data[i] - bb;
	  error += 0.5f*(err*err)[0];
	}
	
	error /= data.size();
	

	// stores error values
	if(selected == goodness.end()) // randomly generated
	{
	  math::vertex<> p;
	  std::pair< math::blas_real<float>, math::vertex<> > tuple;
	  
	  p.resize(b.size() + d.size());
	  
	  p.importData(&(b[0]), b.size(), 0);
	  p.importData(&(d[0]), d.size(), b.size());
	  
	  tuple.first  = error;
	  tuple.second = p;
	  
	  goodness.insert(tuple);
	}
	else{ // previous solution
	  if(selected->first > error){ // error became smaller
	    goodness.erase(selected);
	    
	    math::vertex<> p;
	    std::pair< math::blas_real<float>, math::vertex<> > tuple;
	    
	    p.resize(b.size() + d.size());
	    
	    p.importData(&(b[0]), b.size(), 0);
	    p.importData(&(d[0]), d.size(), b.size());
	    
	    tuple.first  = error;
	    tuple.second = p;
	    
	    goodness.insert(tuple);
	    
	    if(crossover) 
	      std::cout << "CROSSOVER!" << std::endl;
	  }
	}


	
	std::cout << "deepness: " << deepness 
		  << " error: " << error 
		  << " ratio1: " << best_error/orig_error 
		  << " ratio2: " << try_error/best_error
		  << " iters: " << iters << std::endl;
	
	// best b and d found in this layer so far
	if(error < try_error){
	  try_error = error;
	  best_d = d;
	  best_b = b;
	  
	  no_improvement = 0;
	}
	else{
	  no_improvement++;
	}
	
	if(error < best_error){
	  best_error = error;
	}
	else{
	  iters++;
	  continue; // try again with different parameters
	}
      }
      
      goodness.clear();
      
      ultradeep_parameters q;
      q.d = d;
      q.b = b;
      
      params.push_back(q);
      
      
      hasPCA = false;
      try_error = 10e9;
      data = test_data;
      no_improvement = 0;
      
      deepness++;
      iters++;
    }
    
    return true;
  }

}
