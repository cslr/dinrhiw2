/*
 * compiles test / bad lossy data compression
 */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "gha.h"

using namespace std;

bool generate(vector< vector<double> >& d, double p1, double p2,
	      int dimension);

bool calc(vector< vector<double> >& d1,
	  vector< vector<double> >& d2,
	  vector<double>& mean, vector<double>& var);


int main(int argc, char* argv[])
{
  PCA<double> *pca = new GHA<double>(1.0);

  const int DATA_DIMENSION  = 20;
  const int CODED_DIMENSION = 15;
  const int DATA_SIZE = 2*1000;
  const int FACTOR = 50;
  
  vector< vector<double> > D1;
  vector< vector<double> > D2;
  vector< vector<double> > CODED;
  vector< double > error;

  if(!pca->reset(DATA_DIMENSION,
		 CODED_DIMENSION))
    cout << "PCA INIT FAILED\n";

  D1.resize(DATA_SIZE/2);
  D2.resize(DATA_SIZE/2);
  CODED.resize(DATA_SIZE);
  error.resize(DATA_SIZE);

  cout << "GENERATING DATA.." << endl;

  /* generates data */
  generate(D1, 0.5,  1.25, DATA_DIMENSION);
  generate(D2, -0.2, 0.43, DATA_DIMENSION);

  cout << "TRAINING GHA.." << endl;

  /* train */
  for(unsigned int i=0;i<DATA_SIZE*FACTOR;i++)
  {
    if(i % DATA_SIZE == 0){
      printf("\r%d/%d    convergence: %f       \n",
	     i/DATA_SIZE, FACTOR,
	     dynamic_cast< GHA<double>* >(pca)->estimate_convergence());

      fflush(stdout);
    }
    
    
    unsigned int index = (unsigned int)random_number.real(0, DATA_SIZE-1);

    if(index < D1.size()){
      if(pca->train(D1[index]) == false)
	cout << "TRAINING FAILED.\n";
    }
    else
      if(pca->train(D2[index - D1.size()]) == false)
	cout << "TRAINING FAILED.\n";
  }

  printf("\n");

  cout << "CODING/ENCODING.." << endl;

  vector<double> temp;
  
  /* code */
  for(unsigned int i=0;i<(unsigned)DATA_SIZE;i++){
    if(i < D1.size()){
      pca->code(D1[i], CODED[i]);
      
      pca->encode(CODED[i], temp);
      error[i] = 0.0;

      for(unsigned int j=0;j<temp.size();j++)
	error[i] += (temp[j] - D1[i][j])*(temp[j] - D1[i][j]);

      error[i] = sqrt(error[i]);
      
    }
    else{
      pca->code(D2[i - D1.size()], CODED[i]);
      pca->encode(CODED[i], temp);

      for(unsigned int j=0;j<temp.size();j++)
	error[i] += (temp[j] - D2[i - D1.size()][j]) *
	  (temp[j] - D2[i - D1.size()][j]);

      error[i] = sqrt(error[i]);
    }
  }

  cout << "CONV: ";
  cout << dynamic_cast< GHA<double>* >(pca)->estimate_convergence();
  cout << endl;
    
    

  delete pca;

  double total_error = 0.0;

  for(unsigned int i=0;i<error.size();i++){
    double e = error[i];
    
    total_error += e;
  }

  vector<double> mean, var;
  calc(D1, D2, mean, var);

  total_error /= error.size();

  double m = 0, v = 0;
  
  for(unsigned int i=0;i<mean.size();i++){
    m += mean[i];
    v = var[i];    
  }

  m = m / (double)mean.size();
  v = v / (double)var.size();

  cout << "MEAN: " << m << " SQRT(VAR): " << v << endl;
  cout << "MEAN ERROR: " << total_error << endl;

  cout << 1.0 - CODED_DIMENSION/(double)DATA_DIMENSION << " DIM REDUCTION" << endl;

  return 0;
}



/*
 * not so very good test data
 */
bool generate(vector< vector<double> >& d, double p1, double p2,
	      int dimension)
{
  /*
   * p1 = mean
   * p2 = +- range
   * + random error
   * + range(dim) = p1*cur_dim*1.5/total_dim;
   */


  vector<double> r;
  r.resize(dimension);

  for(unsigned int i=0;i<d.size();i++){

    for(unsigned int j=0;j<(unsigned)dimension;j++){
      double range = p2;
	      
      r[j] = random_number.real(p1 - range, p1 + range);

      if(2*(j/2) == j){
	double t = random_number.real(-p1, p1);
	double u = random_number.real(-p2, p2);
	
	r[j] = random_number.real(t - u, t + u);
      }
      
    }
      
    d[i].resize(dimension);
    d[i] = r;    
  }

  return true;
}


bool calc(vector< vector<double> >& d1,
	  vector< vector<double> >& d2,
	  vector<double>& mean, vector<double>& var)
{
  mean.resize(d1[0].size());
  var.resize(d1[0].size());

  for(unsigned int j=0;j<d1[0].size();j++){

    for(unsigned int i=0;i<d1.size();i++)
      mean[j] += d1[i][j];
    
    for(unsigned int i=0;i<d2.size();i++)
      mean[j] += d2[i][j];
    
    mean[j] = mean[j] / (double)(d1.size() + d2.size());
    
    var[j] = 0;
    
    for(unsigned int i=0;i<d1.size();i++)
      var[j] += (d1[i][j] - mean[j])*(d1[i][j] - mean[j]);
    
    for(unsigned int i=0;i<d2.size();i++)
      var[j] += (d2[i][j] - mean[j])*(d2[i][j] - mean[j]);
    
    var[j] = var[j] / (double)(d1.size() + d2.size()-1); // not variance really
    
    var[j] = sqrt(var[j]);
  }

  return true;
}
