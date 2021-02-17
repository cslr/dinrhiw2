
#include "stackedRBM_pretraining.h"

#include "DBN.h"
#include "GBRBM.h"
#include "BBRBM.h"
#include "nnetwork.h"
#include "dataset.h"
#include "Log.h"

namespace whiteice
{
  
  template <typename T>
  bool deep_pretrain_nnetwork(whiteice::nnetwork<T>*& nn,
			      const whiteice::dataset<T>& data,
			      const bool binary,
			      const int verbose,
			      const bool* running)
  {
    if(verbose == 1){
      printf("deep_pretrain_nnetwork() started\n");
      fflush(stdout);
    }
    else if(verbose == 2)
      whiteice::logging.info("deep_pretrain_nnetwork() started");
    
    if(nn == NULL) return false;
    if(data.getNumberOfClusters() < 2) return false;
    if(data.size(0) != data.size(1)) return false;
    if(data.size(0) <= 0) return false;
    if(data.access(0,0).size() != nn->input_size()) return false;
    if(data.access(1,0).size() != nn->output_size()) return false;

    if(running) if(*running == false) return false; // stops execution

    if(verbose >= 1)
      data.diagnostics();

    std::vector<unsigned int> arch;

    nn->getArchitecture(arch);

    if(arch.size() <= 2){
      // does nothing because there is only single layer
      // to optimize and traditional optimizers should work rather well
      if(verbose) printf("deep_ptretrain_nnetwork(): A\n");
      return true; 
    }

    // creates DBN network for training
    arch.pop_back(); // removes output layer from DBN

    // checks nnetwork has proper non-linearities..
    {
      for(unsigned int l=0;l<(nn->getLayers()-1);l++)
	if(nn->getNonlinearity(l) != whiteice::nnetwork<T>::sigmoid){
	  if(verbose) printf("deep_ptretrain_nnetwork(): B\n");
	  return false;
	}
      
      if(nn->getNonlinearity(nn->getLayers()-1) != 
	 whiteice::nnetwork<T>::pureLinear){
	if(verbose) printf("deep_ptretrain_nnetwork(): C\n");
	return false;
      }
    }
    
    whiteice::DBN<T> dbn(arch, binary);

    if(verbose >= 1)
      dbn.diagnostics();

    std::vector< math::vertex<T> > samples;
    data.getData(0, samples);

    T minimumError = T(0.001); // error requirements..
    
    // trains deep belief network DBN
    if(dbn.learnWeights(samples, minimumError, verbose, running) == false){
      if(verbose) printf("deep_ptretrain_nnetwork(): D\n");
      return false;
    }

    if(running) if(*running == false) return false; // stops running

    arch.clear();
    nn->getArchitecture(arch);

    auto old_nn = nn;

    // .. and converts it to nnetwork (adds final linear layer)
    if(dbn.convertToNNetwork(data, nn) == false){
      if(nn != nullptr && nn != old_nn) delete nn;
      nn = old_nn;

      if(verbose) printf("deep_ptretrain_nnetwork(): E\n");
      
      return false;
    }

    if(old_nn) delete old_nn; // deletes old NN

    return true;
  }


  template <typename T>
  bool deep_pretrain_nnetwork_full_sigmoid(whiteice::nnetwork<T>*& nn,
					   const whiteice::dataset<T>& data,
					   const bool binary,
					   const int verbose,
					   const bool* running)
  {
    if(verbose == 1){
      printf("deep_pretrain_nnetwork_full_sigmoid() started\n");
      fflush(stdout);
    }
    else if(verbose == 2)
      whiteice::logging.info("deep_pretrain_nnetwork_full_sigmoid() started");
    
    if(nn == NULL) return false;
    if(data.getNumberOfClusters() < 1) return false;
    if(data.size(0) <= 0) return false;
    if(data.access(0,0).size() != nn->input_size()) return false;

    if(running) if(*running == false) return false; // stops execution

    if(verbose == 2)
      data.diagnostics();

    std::vector<unsigned int> arch;

    nn->getArchitecture(arch);

    if(arch.size() <= 2){
      // does nothing because there is only single layer
      // to optimize and traditional optimizers should work rather well
      return true; 
    }

    // creates DBN network for training
    // arch.pop_back(); // removes output layer from DBN

    // checks nnetwork has proper non-linearities..
    {
      for(unsigned int l=0;l<nn->getLayers();l++)
	if(nn->getNonlinearity(l) != whiteice::nnetwork<T>::sigmoid)
	  return false;      
    }
    
    whiteice::DBN<T> dbn(arch, binary);

    if(verbose == 2)
      dbn.diagnostics();

    std::vector< math::vertex<T> > samples;
    data.getData(0, samples);

    T minimumError = T(0.001); // error requirements..
    
    // trains deep belief network DBN
    if(dbn.learnWeights(samples, minimumError, verbose, running) == false)
      return false;

    if(running) if(*running == false) return false; // stops running

    arch.clear();
    nn->getArchitecture(arch);

    auto old_nn = nn;

    // .. and converts it to nnetwork (adds final linear layer)
    if(dbn.convertToNNetwork(nn) == false){
        if(nn != nullptr && nn != old_nn) delete nn;
        nn = old_nn;
        return false;
    }

    if(old_nn) delete old_nn; // deletes old NN

    return true;
  }
  
  
  //////////////////////////////////////////////////////////////////////

  
  template bool deep_pretrain_nnetwork< math::blas_real<float> >
    (whiteice::nnetwork< math::blas_real<float> >*& nn,
     const whiteice::dataset< math::blas_real<float> >& data,
     const bool binary,
     const int verbose,
     const bool* running);
  
  template bool deep_pretrain_nnetwork< math::blas_real<double> >
    (whiteice::nnetwork< math::blas_real<double> >*& nn,
     const whiteice::dataset< math::blas_real<double> >& data,
     const bool binary,
     const int verbose,
     const bool* running);

  
  template bool deep_pretrain_nnetwork_full_sigmoid< math::blas_real<float> >
  (whiteice::nnetwork< math::blas_real<float> >*& nn,
   const whiteice::dataset< math::blas_real<float> >& data,
   const bool binary,
   const int verbose,
   const bool* running);
  
  template bool deep_pretrain_nnetwork_full_sigmoid< math::blas_real<double> >
  (whiteice::nnetwork< math::blas_real<double> >*& nn,
   const whiteice::dataset< math::blas_real<double> >& data,
   const bool binary,
   const int verbose,
   const bool* running);
  
}


