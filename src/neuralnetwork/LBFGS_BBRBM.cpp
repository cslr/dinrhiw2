
#include "LBFGS_BBRBM.h"
#include "deep_ica_network_priming.h"

namespace whiteice
{

  template <typename T>
  LBFGS_BBRBM<T>::LBFGS_BBRBM(const whiteice::BBRBM<T>& nn, const dataset<T>& d, bool overfit) :
    whiteice::math::LBFGS<T>(overfit), net(nn), data(d)
  {
    // IMPORTANT incoming net must have setUData() set to proper samples!
    
    /*
    if(data.getNumberOfClusters() > 0){
      std::vector< math::vertex<T> > data;
      data.getData(0, data);
      net.setUData(data);
    }
    */
  }

  
  template <typename T>
  LBFGS_BBRBM<T>::~LBFGS_BBRBM()
  {
  }


  template <typename T>
  T LBFGS_BBRBM<T>::getError(const math::vertex<T>& x) const
  {
    // return -this->net.U(x);
    
    T e = T(0.0f);
    
#pragma omp parallel shared(e)
    {
      whiteice::BBRBM<T> nnet(this->net);
      nnet.setParametersQ(x);
      
      math::vertex<T> err;
      T esum = T(0.0f);
      
      
      // E = SUM 0.5*e(i)^2
      #pragma omp for nowait schedule(dynamic)
      for(unsigned int i=0;i<1000;i++){
	// std::cout << "data in  = " << dtest.access(0, i) << std::endl;
	// std::cout << "data out = " << dtest.access(1, i) << std::endl;
	unsigned int index = rand()%data.size(0);

	auto v = data.access(0, index);

	nnet.setVisible(v);
	nnet.reconstructData(2);
	nnet.getVisible(v);
	err = data.access(0, index) - v;
	
	err = (err*err);
	esum += T(0.5f)*err[0];
      }

#pragma omp critical
      {
	e += esum;
      }

    }
    
    e /= T( (double)1000 ); // per N
    
    return e;
  }
  

  template <typename T>
  T LBFGS_BBRBM<T>::U(const math::vertex<T>& x) const
  {
    return this->getError(x);
    // return -this->net.U(x);
  }

  
  template <typename T>
  math::vertex<T> LBFGS_BBRBM<T>::Ugrad(const math::vertex<T>& x) const
  {
    whiteice::BBRBM<T> gbrbm(this->net);
    
    return -gbrbm.Ugrad(x);
  }
  
  
  template <typename T>
  bool LBFGS_BBRBM<T>::heuristics(math::vertex<T>& x) const
  {
    return true; // does nothing..
    
  }
  
  
  template class LBFGS_BBRBM< float >;
  template class LBFGS_BBRBM< double >;
  template class LBFGS_BBRBM< math::blas_real<float> >;
  template class LBFGS_BBRBM< math::blas_real<double> >;

  
};
