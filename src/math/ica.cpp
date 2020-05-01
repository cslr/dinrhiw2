/*
 * independent component analysis
 */

#include "ica.h"
#include "matrix.h"
#include "vertex.h"
#include "eig.h"
#include "correlation.h"

#include <stdlib.h>


namespace whiteice
{
  namespace math
  {
    
    template bool ica< blas_real<float> >
      (const matrix< blas_real<float> >& D, matrix< blas_real<float> >& W, bool verbose) ;
    template bool ica< blas_real<double> >
      (const matrix< blas_real<double> >& D, matrix< blas_real<double> >& W, bool verbose) ;
    template bool ica< float >
      (const matrix<float>& D, matrix<float>& W, bool verbose) ;
    template bool ica< double >
      (const matrix<double>& D, matrix<double>& W, bool verbose) ;

    template bool ica< blas_real<float> >
      (const std::vector< math::vertex< blas_real<float> > >& data, matrix< blas_real<float> >& W, bool verbose) ;
    template bool ica< blas_real<double> >
      (const std::vector< math::vertex< blas_real<double> > >& data, matrix< blas_real<double> >& W, bool verbose) ;
    template bool ica< float >
      (const std::vector< math::vertex<float> >& data, matrix<float>& W, bool verbose) ;
    template bool ica< double >
      (const std::vector< math::vertex<double> >& data, matrix<double>& W, bool verbose) ;    
    
    template <typename T>
    void __ica_project(vertex<T>& w, const unsigned int n, const matrix<T>& W);

    template void __ica_project< blas_real<float> >
      (vertex< blas_real<float> >& w, const unsigned int n, const matrix< blas_real<float> >& W);
    template void __ica_project< blas_real<double> >
      (vertex< blas_real<double> >& w, const unsigned int n, const matrix< blas_real<double> >& W);
    template void __ica_project<float>
      (vertex<float>& w, const unsigned int n, const matrix<float>& W);
    template void __ica_project<double>
      (vertex<double>& w, const unsigned int n, const matrix<double>& W);
    
    
    template <typename T>
    bool ica(const std::vector< math::vertex<T> >& data, matrix<T>& W, bool verbose) 
    {
      // this is interface class that simply makes dataset<> calls work, does an costly data format transformation here
      // as the data types are not compatible (TODO: write code that uses DIRECTLY std::vector<> datasets

      if(data.size() <= 0) return false;

      matrix<T> DATA;

      DATA.resize(data.size(), data[0].size());

      for(unsigned int j=0;j<DATA.ysize();j++)
	for(unsigned int i=0;i<DATA.xsize();i++)
	  DATA(j, i) = data[j][i];

      return ica(DATA, W, verbose);
    }

    
    template <typename T>
    bool ica(const matrix<T>& D, matrix<T>& W, bool verbose) 
    {
      try{
	
	const unsigned int num = D.ysize();
	const unsigned int dim = D.xsize();
	
	// data MUST be already white (PCA preprocessed)

	const matrix<T>& X = D; // X = (V * D')'
	
	/*
	matrix<T> V;
	
	{
	  matrix<T> Rxx;
	  if(autocorrelation(Rxx, D) == false)
	    return false;
	  
	  if(symmetric_eig(Rxx, V) == false)
	    return false;
	  
	  // calculates V = D**(-0.5) * V.transpose()
	  V.transpose();
	  
	  for(unsigned int j=0;j<dim;j++){
	    T scaling = T(1.0) / whiteice::math::sqrt(Rxx(j,j));
	    for(unsigned int i=0;i<dim;i++)
	      V(j,i) *= scaling;
	  }
	}
	
	V.transpose();
	matrix<T> X = D * V; // X = (V * D')'
	V.transpose();
	*/
	
	const T TOLERANCE = T(0.0001);
	const unsigned int MAXITERS = 200;
	
	// matrix<T> W;
	W.resize(dim,dim);
	
	for(unsigned int j=0;j<dim;j++)
	  for(unsigned int i=0;i<dim;i++)
	    W(j,i) = T((float)rand()) / T((float)RAND_MAX);
	
	vertex<T> w;
	w.resize(dim);            
	
	// solves each IC separatedly (deflate method)
	for(unsigned int n=0;n<dim;n++){
	  
	  // initialization
	  W.rowcopyto(w, n);
	  
	  w.normalize();
	  __ica_project(w, n, W);
	  
	  bool convergence = 0;
	  unsigned int iter = 0;
	  vertex<T> w_old, y;
	  T scaling = T(1.0) / T(num);
	  
	  vertex<T> x; x.resize(dim);
	  vertex<T> xgy; xgy.resize(dim); // E[x * g(w * x)]
	  T dgy; // E[g'(w * x)]
	  
	  while(convergence == false){
	    
	    // updates w vector
	    w_old = w;
	    
	    y = X * w;
	    
	    if((iter % 2) == 0){ // tanh non-linearity
	      for(unsigned int i=0;i<dim;i++) xgy[i] = T(0.0);
	      dgy = 0;
	      
	      for(unsigned int i=0;i<num;i++){
		X.rowcopyto(x, i);
		T temp = whiteice::math::exp(-(y[i]*y[i])/T(2.0));
		
		xgy += scaling * ((y[i] * temp) * x);
		dgy += scaling * (temp - (y[i]*y[i])*temp);
	      }
	    }
	    else{ // u*exp(-u**2/2) non-linearity
	      
	      for(unsigned int i=0;i<dim;i++) xgy[i] = T(0.0);
	      dgy = 0;
	      
	      for(unsigned int i=0;i<num;i++){
		X.rowcopyto(x, i);
		T temp = whiteice::math::exp(-(y[i]*y[i])/T(2.0));
		
		xgy += scaling * ((y[i] * temp) * x);
		dgy += scaling * (temp - (y[i]*y[i])*temp);
	      }
	    }
	    
	    w = xgy - dgy*w;
	    w.normalize();
	    __ica_project(w, n, W); // projection
	    
	    // checks for convergence / stopping critearias
	    w_old = w_old * w;	  
	    if((T(1.0) - whiteice::math::abs(w_old[0])) < TOLERANCE)
	      convergence = true;
	    else if(iter >= MAXITERS)
	      break;
	    
	    iter++;
	  }
	  
	  W.rowcopyfrom(w, n);
	  
	  if(convergence == 0){
	    if(verbose)
	      std::cout << "Warning: IC " << (n+1) << " didn't converge"
			<< std::endl;
	  }
	  else{
	    if(verbose)
	      std::cout << "IC " << (n+1) << " converged after " 
			<< iter << " iterations." << std::endl;
	  }
	}
	
	return true;
      }
      catch(std::exception& e){
	if(verbose)
	  std::cout << "uncaught exception: " << e.what() << std::endl;
	return false;
      }
    }
    
    
    // projects w to the remaining free subspace
    template <typename T>
    void __ica_project(vertex<T>& w, const unsigned int n, const matrix<T>& W)
    {
      vertex<T> s(w.size()); // s = (initialized to be) zero vector
      vertex<T> t(w.size());
      
      for(unsigned int i=0;i<n;i++){
	W.rowcopyto(t, i); // t = W(i,:)
	s += (t * w) * t;  // amount t basis vector w has
      }
      
      w -= s;
      w.normalize();
    }
    
    
  };
};
