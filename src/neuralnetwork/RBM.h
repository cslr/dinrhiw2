/*
 * Restricted Boltzmann Machine
 * 
 * data is stored as in binary form 0 or 1
 */

#ifndef RBM_H
#define RBM_H

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
    class RBM
    {
    public:
    
    // creates 1x1 network, used to load some useful network
    RBM();
    RBM(const RBM<T>& rbm);
    
    // creates 2-layer: V * H network
    RBM(unsigned int visible, unsigned int hidden) throw(std::invalid_argument);
    
    virtual ~RBM();
    
    RBM<T>& operator=(const RBM<T>& rbm);
    
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
    
    // calculates single epoch for updating weights using CD-1 and returns |dW|
    // (keep calculating until returned value is close enough to zero) or 
    //  the number of epochs is reached)
    T learnWeights(const std::vector< math::vertex<T> >& samples);
    
    ////////////////////////////////////////////////////////////
    
    // load & saves RBM data from/to file
    
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

  extern template class RBM< float >;
  extern template class RBM< double >;  
  extern template class RBM< math::blas_real<float> >;
  extern template class RBM< math::blas_real<double> >;
  
};



#endif
