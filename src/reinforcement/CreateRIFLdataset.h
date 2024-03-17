// helper class for RIFL_abstract to generate
// (in the background) dataset for training reinforcement model

#ifndef whiteice_CreateRIFLdataset_h
#define whiteice_CreateRIFLdataset_h

#include <thread>
#include <mutex>
#include <vector>

#include "dataset.h"
#include "dinrhiw_blas.h"
#include "RIFL_abstract.h"

namespace whiteice
{

  template <typename T = math::blas_real<float> >
  class CreateRIFLdataset
  {
  public:
    
    // calculates reinforcement learning training dataset from database
    // using database_lock
    CreateRIFLdataset(const RIFL_abstract<T> & rifl, 
		      const std::vector< rifl_datapoint<T> > & database,
		      const std::vector< std::vector< rifl_datapoint<T> > >& episodes,
		      std::mutex & database_mutex,
		      const unsigned int & epoch, 
		      whiteice::dataset<T>& data);
    
    virtual ~CreateRIFLdataset();
    
    // starts thread that creates NUMDATAPOINTS samples to dataset
    bool start(const unsigned int NUMDATAPOINTS, const bool useEpisodes = false);

    // returns true when computation is completed
    bool isCompleted() const;

    // returns true if computation is running
    bool isRunning() const;
    
    bool stop();

    // returns reference to dataset
    // (warning: if calculations are running then dataset can change during use)
    whiteice::dataset<T> const & getDataset() const;

  private:
    
    RIFL_abstract<T> const & rifl;
    
    const std::vector< rifl_datapoint<T> >& database;
    const std::vector< std::vector< rifl_datapoint<T> > >& episodes;
    std::mutex & database_mutex;

    bool useEpisodes = false;

    unsigned int const& epoch;

    unsigned int NUMDATA; // number of datapoints to create
    whiteice::dataset<T>& data;
    bool completed;

    std::thread* worker_thread;
    std::mutex   thread_mutex;
    bool running;

    // worker thread loop
    void loop();
    
      
    };


  extern template class CreateRIFLdataset< math::blas_real<float> >;
  extern template class CreateRIFLdataset< math::blas_real<double> >;
};

#endif
