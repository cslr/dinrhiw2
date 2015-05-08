/*
 * Continuous Restricted Boltzmann Machine
 * 
 * data is stored between range of [-1, 1]
 */

#ifndef CRBM_H
#define CRBM_H

#include "dinrhiw_blas.h"
#include "vertex.h"
#include "matrix.h"
#include "conffile.h"
#include <exception>
#include <stdexcept>
#include <typeinfo>
#include <vector>


namespace whiteice
{
  template <typename T = math::blas_real<float> >
    class CRBM
    {
    public:
    
    // creates 1x1 network, used to load some useful network
    CRBM();
    CRBM(const CRBM<T>& rbm);
    
    // creates 2-layer: V * H network
    CRBM(unsigned int visible, unsigned int hidden) throw(std::invalid_argument);
    
    virtual ~CRBM();
    
    CRBM<T>& operator=(const CRBM<T>& rbm);
    
    ////////////////////////////////////////////////////////////
    
    math::vertex<T> getVisible() const;
    bool setVisible(const math::vertex<T>& v);
    
    math::vertex<T> getHidden() const;
    bool setHidden(const math::vertex<T>& h);
    
    // number of iterations to daydream, 
    // 2 = single step from visible to hidden and back
    // from hidden to visible (CD-1)
    bool reconstructData(unsigned int iters = 2);
    
    math::matrix<T> getWeights() const;
    
    bool initializeWeights(); // initialize weights to small values
    
    // calculates single epoch for updating weights using CD-10 and returns |dW|
    // (keep calculating until returned value is close enough to zero) or 
    //  the number of epochs is reached)
    T learnWeights(const std::vector< math::vertex<T> >& samples);
    
    ////////////////////////////////////////////////////////////
    
    // load & saves CRBM data from/to file
    
    bool load(const std::string& filename) throw();
    bool save(const std::string& filename) const throw();
    
    ////////////////////////////////////////////////////////////
    
    private:
    
    T randomValue(); // random values for initialization of data structures
    
    // visible units [+ one fixed to 1 bias unit]
    math::vertex<T> v;
    
    // W(size(h),size(v)) connection matrix W between visible and hidden neurons/units
    math::matrix<T> W; 
    
    // hidden units [+ one fixed to 1 bias unit]
    math::vertex<T> h;
      
    };

  extern template class CRBM< float >;
  extern template class CRBM< double >;  
  extern template class CRBM< math::blas_real<float> >;
  extern template class CRBM< math::blas_real<double> >;
  
};



#endif
