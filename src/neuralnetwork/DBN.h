/*
 * Constructs deep belief network using stacked binary RBMs.
 * 
 */

#ifndef DBN_h
#define DBN_h

#include <vector>
#include "RBM.h"
#include "vertex.h"


namespace whiteice
{
  template <typename T = math::blas_real<float> >
    class DBN
    {
    public:
    
    DBN();
    
    // constructs stacked RBM netwrok with the given architecture
    DBN(std::vector<unsigned int>& arch);
    
    DBN(const DBN<T>& dbn);
    
    bool resize(std::vector<unsigned int>& arch);
    
    ////////////////////////////////////////////////////////////
    
    // visible neurons/layer of the first RBM
    math::vertex<T> getVisible() const;
    bool setVisible(const math::vertex<T>& v);
    
    // hidden neurons/layer of the last RBM
    math::vertex<T> getHidden() const;
    bool setHidden(const math::vertex<T>& h);
    
    // number of iterations to simulate the system 
    bool reconstructData(unsigned int iters = 2);
    
    bool initializeWeights(); // initialize weights to small values
    
    // learns stacked RBM layer by layer, each RBM is trained one by one
    // until deltaW < dW and then algorithm moves to the next layer
    bool learnWeights(const std::vector< math::vertex<T> >& samples, const T& dW);
    
    private:
    
    // stacked RBMs from the first to the last one
    std::vector< whiteice::RBM<T> > machines;
    
    };
  

  extern template class DBN< float >;
  extern template class DBN< double >;  
  extern template class DBN< math::blas_real<float> >;
  extern template class DBN< math::blas_real<double> >;

};


#endif

