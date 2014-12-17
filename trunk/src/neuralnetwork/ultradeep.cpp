

#include "ultradeep.h"

#include "eig.h"
#include "correlation.h"
#include "linear_algebra.h"
#include "ica.h"


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
      
      Cxx.inv();
      
      A = Cxy.transpose() * Cxx;
      b = my - A*mx;
    }
    catch(std::exception& e){
      return false;
    }
    
    return true;
  }    
  
  


  bool ultradeep(std::vector< math::vertex<> > input,
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
    unsigned int iters = 0;
    
    std::vector< math::vertex<> > goodness_p;
    std::vector< math::vertex<> > goodness_g;

    math::vertex<> d;
    math::vertex<> b;
    
    d.resize(input[0].size());
    b.resize(input[0].size());
    
    
    while(deepness < 100){
      std::vector< math::vertex<> > data = input;
      
      if((goodness_p.size() > b.size() + d.size()) && (rand()&1) == 0){
	math::matrix<> AA;
	math::vertex<> bb;
	
	if(calculate_linear_fit(goodness_p, goodness_g, AA, bb) == false)
	  return false;
	

	
	// gradients
	math::vertex<> db;
	math::vertex<> dd;
	
	db.resize(b.size());
	dd.resize(d.size());
	
	AA.rowcopyto(db, 0, 0, db.size() - 1);
	AA.rowcopyto(dd, 0, db.size(), db.size()+dd.size() - 1);

	const unsigned int index = rand() % goodness_p.size();
	math::vertex<>& p = goodness_p[index];
	
	p.exportData(&(b[0]), b.size(), 0);
	p.exportData(&(d[0]), d.size(), b.size());
	
	math::blas_real<float> rate = ((float)rand())/RAND_MAX;
	
	// now we go to the gradient direction
	b -= rate * db;
	d -= rate * dd;
	
      }
      else{
	for(unsigned int i=0;i<d.size();i++){
	  d[i] = 2.0f*((((float)rand())/RAND_MAX) - 0.5f);
	  b[i] = 2.0f*((((float)rand())/RAND_MAX) - 0.5f);
	}
      }
      
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
      
      math::matrix<>& PCA = D;
      PCA *= V.transpose();    
      

      for(unsigned int i=0;i<data.size();i++){
	data[i] = PCA*(data[i] - m);

	for(unsigned int j=0;j<data[i].size();j++)
	  data[i][j] *= d[j];

	
	data[i] += b;
	
	for(unsigned int j=0;j<data[i].size();j++)
	  data[i][j] = math::asinh(data[i][j]);
      }
      
      // checks what is the error if we compute linear 
      // fitting output = A*x + b at this time step
      {
	math::matrix<> AA;
	math::vertex<> bb;
	
	if(calculate_linear_fit(data, output, AA, bb) == false)
	  return false;
	
	math::blas_real<float> error = 0.0f;
	math::vertex<> err;
	
	for(unsigned int i=0;i<data.size();i++){
	  err = output[i] - AA*data[i] - bb;
	  error += 0.5f*(err*err)[0];
	}
	
	error /= data.size();
	
	// stores error values
	{
	  math::vertex<> p;
	  math::vertex<> g;
	  
	  p.resize(b.size() + d.size());
	  g.resize(1);
	  
	  p.importData(&(b[0]), b.size(), 0);
	  p.importData(&(d[0]), d.size(), b.size());
	  
	  g[0] = error;
	  
	  goodness_p.push_back(p);
	  goodness_g.push_back(g);
	}
	
	std::cout << "deepness: " << deepness 
		  << " error: " << error 
		  << " ratio: " << best_error/orig_error 
		  << " iters: " << iters << std::endl;
	
	if(error < best_error){
	  best_error = error;
	  input = data;
	}
	else{
	  iters++;
	  continue; // try again with different parameters
	}
      }
      
      goodness_p.clear();
      goodness_g.clear();
      
      ultradeep_parameters q;
      q.d = d;
      q.b = b;
      
      params.push_back(q);
      
      deepness++;
      iters++;
    }
    
    return true;
  }

}
