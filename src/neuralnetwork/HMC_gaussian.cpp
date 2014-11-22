
#include "HMC_gaussian.h"


namespace whiteice
{

  template <typename T>
  HMC_gaussian<T>::HMC_gaussian(unsigned int dimension)
  {    
    this->dimension = dimension;

    this->adaptive = true; // we test gaussian adaptive criteria here
  }

  template <typename T>
  HMC_gaussian<T>::~HMC_gaussian()
  {
  }
  
  // probability functions for hamiltonian MC sampling of
  // P ~ exp(-U(q)) distribution
  template <typename T>
  T HMC_gaussian<T>::U(const math::vertex<T>& q) const
  {
    return T(0.5)*((q * q)[0]);
  }


  template <typename T>
  math::vertex<T> HMC_gaussian<T>::Ugrad(const math::vertex<T>& q)
  {
    return q;
  }

  
  // a starting point q for the sampler (may not be random)
  template <typename T>
  void HMC_gaussian<T>::starting_position(math::vertex<T>& q) const
  {
    // start from p(0) [max position] so we are already converged
    q.resize(dimension);
    q.zero();
  }
    
};



namespace whiteice
{
  template class HMC_gaussian< float >;
  template class HMC_gaussian< double >;
  template class HMC_gaussian< math::blas_real<float> >;
  template class HMC_gaussian< math::blas_real<double> >;  
  
};
