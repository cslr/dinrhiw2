
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "SOM2D.h"
#include "KMeans.h"
#include "dataset.h"


using namespace whiteice::math;
using namespace whiteice;




template <typename T>
void init_data(std::vector< vertex<T> >& data,
	       unsigned int dimension) throw();


class test_datasource :
  public data_source < vertex<float> >
{
public:
  test_datasource(){ datasource = 0; }
  test_datasource(std::vector< vertex<float> >& data){ datasource = &data; }
  ~test_datasource(){ }
  
  vertex<float>& operator[](unsigned int index) throw(std::out_of_range)
  {
    return (*datasource)[index];
  }
  
  
  const vertex<float>& operator[](unsigned int index) const throw(std::out_of_range)
  {
    return (*datasource)[index];
  }
  
  unsigned int size() const throw()
  {
    return datasource->size();
  }
  
  bool good() const throw()
  {
    return true;
  }
  
  void flush() const { }
  
  void setsource(std::vector< vertex<float> >& data)
  {
    datasource = &data;
  }
  
private:
  
  std::vector< vertex<float> >* datasource;
};




int main()
{
  using namespace whiteice;
  srand(time(0));
  
  std::vector< vertex< whiteice::math::blas_real<float> > > data;
  std::vector< vertex< whiteice::math::blas_real<float> > > pdata;
  data.resize(5000);
  init_data< whiteice::math::blas_real<float> >(data, 10);
  
  dataset<> ds;
  ds.createCluster("input", 10);
  ds.add(0, data);
  ds.preprocess(0, dataset<>::dnMeanVarianceNormalization);
  ds.getData(0, pdata);
  
  
  {
    SOM2D* som = new SOM2D(100, 100, 10);

    hierarchicalTraining(som, pdata);
    som->show(true);

    
    // sleep(5);
    
    std::string filename("somstate.dat");
    if(!som->save(filename))
      std::cout << "saving som failed\n";
    
    SOM2D* som2 = new SOM2D(8,8,10);
    if(!som2->load(filename))
      std::cout << "loading som failed\n";
    else{
      filename = "somstate2.dat";
      if(!som2->save(filename))
	std::cout << "saving second som state failed\n";
    }
    
    delete som2;
    delete som;
  }
  
#if 0
  // K-MEANS TEST CODE
  {
    KMeans<float>* kmeans;  
    kmeans = new KMeans<float>();
    delete kmeans;
    
    kmeans = new KMeans<float>();  
    kmeans->learn(10, data);
    
    std::string filename("kmeans.dat");
    if(!kmeans->save(filename))
      std::cout << "saving som failed\n";
    
    KMeans<float>* kmeans2 = new KMeans<float>();
    if(!kmeans2->load(filename))
      std::cout << "loading som failed\n";
    else{
      filename = "kmeans2.dat";
      if(!kmeans2->save(filename))
	std::cout << "saving second som state failed\n";
    }
    
    delete kmeans2;
    delete kmeans;  
  }
#endif
  
  return 0;
}




// initializes data with white noise
template <typename T>
void init_data(std::vector< vertex<T> >& data,
	       unsigned int dimension) throw()
{
  // creates test data which has three clusters
  std::vector<T> mean[3];
  std::vector<T> spread[3]; // (equal distribution spread around mean)
  
  for(unsigned int i=0;i<3;i++){
    mean[i].resize(dimension);
    for(unsigned int j=0;j<dimension;j++)
      mean[i][j] = T( (rand()/((float)RAND_MAX)) - 0.5f );
  }
  
  for(unsigned int i=0;i<3;i++){
    spread[i].resize(dimension);
    for(unsigned int j=0;j<dimension;j++)
      spread[i][j] = T( (rand()/((float)RAND_MAX)) );
  }
  
  
  for(unsigned int i=0;i<data.size();i++){
    // selects distribution randomly (even probabilities)
    unsigned int c = rand() % 3;
    
    data[i].resize(dimension);
    for(unsigned int j=0;j<data[i].size();j++){
      data[i][j] = mean[c][j] + T(rand()/((float)RAND_MAX) - 0.5f)*spread[c][j];
    }
  }
  
  
}









