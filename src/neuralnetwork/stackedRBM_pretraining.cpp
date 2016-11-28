
#include "stackedRBM_pretraining.h"

#include "DBN.h"
#include "GBRBM.h"
#include "BBRBM.h"
#include "nnetwork.h"
#include "dataset.h"

namespace whiteice
{
  template <typename T>
  bool deep_pretrain_nnetwork(whiteice::nnetwork<T>*& nn,
			      const whiteice::dataset<T>& data,
			      const bool binary,
			      const bool verbose)
  {
    if(nn == NULL) return false;
    if(data.getNumberOfClusters() < 2) return false;
    if(data.size(0) != data.size(1)) return false;
    if(data.access(0,0).size() != nn->input_size()) return false;
    if(data.access(1,0).size() != nn->output_size()) return false;

    std::vector<unsigned int> arch;

    nn->getArchitecture(arch);

    if(arch.size() <= 2){
      // does nothing because there is only single layer
      // to optimize and traditional optimizers should work rather well
      return true; 
    }
    // creates DBN network for training
    arch.pop_back(); // removes output layer from DBN
    
    whiteice::DBN<T> dbn(arch, binary);

    std::vector< math::vertex<T> > samples;
    data.getData(0, samples);

    T minimumError = T(0.001); // error requirements..

    // trains deep belief network DBN
    if(dbn.learnWeights(samples, minimumError, verbose) == false)
      return false;

    arch.clear();
    nn->getArchitecture(arch);

    // .. and converts it to nnetwork (adds final linear layer)
    if(dbn.convertToNNetwork(data, nn) == false)
      return false;

    return true;
  }


  template bool deep_pretrain_nnetwork<float>
    (whiteice::nnetwork<float>*& nn,
     const whiteice::dataset<float>& data,
     const bool binary,
     const bool verbose);

  template bool deep_pretrain_nnetwork<double>
    (whiteice::nnetwork<double>*& nn,
     const whiteice::dataset<double>& data,
     const bool binary,
     const bool verbose);

  template bool deep_pretrain_nnetwork< math::blas_real<float> >
    (whiteice::nnetwork< math::blas_real<float> >*& nn,
     const whiteice::dataset< math::blas_real<float> >& data,
     const bool binary,
     const bool verbose);
  
  template bool deep_pretrain_nnetwork< math::blas_real<double> >
    (whiteice::nnetwork< math::blas_real<double> >*& nn,
     const whiteice::dataset< math::blas_real<double> >& data,
     const bool binary,
     const bool verbose);

  
}


