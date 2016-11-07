/*
 * data set
 * for learning data
 * has data preprocessing code
 *
 * dataset (should) keeps order of data
 */

#ifndef dataset_h
#define dataset_h

#include <vector>
#include "vertex.h"
#include "dinrhiw_blas.h"
#include "ica.h"

#include <stdexcept>
#include <exception>
#include <string>
#include <map>



namespace whiteice
{
  
  template <typename T = math::blas_real<float> >
    class dataset
    {
      public:
      
      // creates dataset with a given number of dimensions
      // data is a set of vectors
      dataset();
      dataset(unsigned int dimension) throw(std::out_of_range);
      dataset(const dataset<T>& d);
      ~dataset() throw();
      
      
      bool createCluster(const std::string& name, const unsigned int dimension);
      
      bool getClusterNames(std::vector<std::string>& names) const;
      
      unsigned int getCluster(const std::string& name) const;
      
      std::string getName(unsigned int index = 0) const;
      bool setName(const unsigned int index, const std::string& name);
      
      unsigned int getNumberOfClusters() const;
      
      bool removeCluster(std::string& name);
      bool removeCluster(unsigned int index);
      
      
      // adds data examples
      bool add(const math::vertex<T>& input, bool nopreprocess = false) throw();
      bool add(const std::vector<math::vertex<T> >& inputs, bool nopreprocess = false) throw();
      
      bool add(const std::string& input, bool nopreprocess = false) throw();    
      bool add(const std::vector<std::string>& inputs, bool nopreprocess = false) throw();

      // adds data to clusters
      bool add(unsigned int index, const math::vertex<T>& input, 
	       bool nopreprocess = false) throw();
      bool add(unsigned int index, const std::vector<math::vertex<T> >& inputs, 
	       bool nopreprocess = false) throw();

      bool add(unsigned int index, const std::vector<T>& input,
	       bool nopreprocess = false) throw();
      
      bool add(unsigned int index, const std::string& input, 
	       bool nopreprocess = false) throw();
      bool add(unsigned int index, const std::vector<std::string>& inputs, 
	       bool nopreprocess = false) throw();
      
      // creates empty dataset
      bool clear();
      
      // removes all data from dataset but no preprocessing
      // information
      bool clearData(unsigned int index = 0);
      
      // clears data and preprocessing information
      bool clearAll(unsigned int index = 0);

      // reduces data by taking sample or 'samples' samples from
      // each cluster (which must have equal size) and keeps
      // order of samples between different clusters the same
      // so ith element of cluster A match to ith element of cluster B
      bool downsampleAll(unsigned int samples) throw();
      
      // removes data elements from dataset which have Infinite or NaN values
      // maintains pairing between cluster A and cluster B, 
      // if i:th element of A is bad, removes also i:th element of B
      bool removeBadData();
      
      // returns data in cluster "index"
      bool getData(unsigned int index, std::vector< math::vertex<T> >& data) const throw(std::out_of_range);
      
      /* defines dataset<T>::iterator */
      typedef typename std::vector< math::vertex<T> >::iterator iterator;
      typedef typename std::vector< math::vertex<T> >::const_iterator const_iterator;
      
      // iterators for dataset
      iterator begin(unsigned int index = 0) throw(std::out_of_range);
      iterator end(unsigned int index = 0) throw(std::out_of_range);
      const_iterator begin(unsigned int index = 0) const throw(std::out_of_range);
      const_iterator end(unsigned int index = 0) const throw(std::out_of_range);
      
      
      /*
       * saves and loads dataset(s) to/from disk.
       * Datasets are saved in own binary format
       * documented in dataset::load() (dataset.cpp)
       * 
       */
      bool load(const std::string& filename) throw();
      bool save(const std::string& filename) const throw();
      
      /*
       * exports dataset values as ascii data without preprocessing
       */
      bool exportAscii(const std::string& filename) const throw();
      
      // accesses data from cluster zero
      const math::vertex<T>& operator[](unsigned int index) const throw(std::out_of_range);
      
      const math::vertex<T>& access(unsigned int cluster, unsigned int data) const throw(std::out_of_range);
      const math::vertex<T>& accessName(const std::string& clusterName, unsigned int dataElem) throw(std::out_of_range);
      
      // accesses random element from specified cluster
      const math::vertex<T>& random_access(unsigned int index = 0) const throw(std::out_of_range);
      
      unsigned int size(unsigned int index) const throw(); // dataset size  
      bool clear(unsigned int index) throw(); // data set clear  
      bool resize(unsigned int index, unsigned int nsize) throw(); // reduces size of data
      unsigned int dimension(unsigned int index) const throw(); // dimension of data vectors
      
      
      // data preprocessing methods
      enum data_normalization { 
	dnMeanVarianceNormalization, // zero mean, unit variance
	dnSoftMax, // forces data to small interval
	dnCorrelationRemoval, // PCA
	dnLinearICA
      }; 
      
      
      bool getPreprocessings(unsigned int cluster,
			     std::vector<data_normalization>& preprocessings) const  throw();
      
      // is data normalized with given operation
      bool hasPreprocess(unsigned int cluster, enum data_normalization norm) const throw(){
    	  if(cluster >= clusters.size()) return false;
    	  else return is_normalized(cluster, norm);
      }

      // data preprocessing
      bool preprocess(unsigned int index,
		      enum data_normalization norm = dnCorrelationRemoval) throw();
      // index = 0
      bool preprocess(enum data_normalization norm = dnCorrelationRemoval) throw();
      
      // inverse preprocess everything, calculates new preprocessing parameters
      // and preprocesses everything with parameter data from the whole dataset
      // (dataset may grow after preprocessing)
      bool repreprocess(unsigned int index = 0) throw();
      
      // converts data with same preprocessing as with dataset vectors
      bool preprocess(unsigned int index,
		      math::vertex<T>& vec) const throw();
      
      bool preprocess(unsigned int index,
		      std::vector< math::vertex<T> >& group) const throw();
      // index = 0
      
      bool preprocess(math::vertex<T>& vec) const throw();
      
      bool preprocess(std::vector< math::vertex<T> >& group) const throw();
  
      // inverse preprocess given data vector
      
      bool invpreprocess(unsigned int index,
			 math::vertex<T>& vec) const throw();
      
      bool invpreprocess(unsigned int index,
			 std::vector< math::vertex<T> >& group) const throw();
      // index = 0
      bool invpreprocess(math::vertex<T>& vec) const throw();
      bool invpreprocess(std::vector< math::vertex<T> >& group) const throw();
      
      // changes preprocessing to given a given list of preprocessings
      bool convert(unsigned int index = 0) throw(); // removes all preprocessings from data
      bool convert(unsigned int index,
		   std::vector<enum data_normalization> plist);
      
    private:
      // is data normalized with given operation?
      bool is_normalized(unsigned int index,
			 enum data_normalization norm) const throw();
      
      // preprocessing functions and inverse preprocessing functions
      void mean_variance_removal(unsigned int index,
				 math::vertex<T>& vec) const;
      void inv_mean_variance_removal(unsigned int index,
				     math::vertex<T>& vec) const;
      void soft_max(unsigned int index, math::vertex<T>& vec) const;
      void inv_soft_max(unsigned int index, math::vertex<T>& vec) const;
      
      void whiten(unsigned int index, math::vertex<T>& vec) const;
      void inv_whiten(unsigned int index, math::vertex<T>& vec) const;

      void ica(unsigned int index, math::vertex<T>& vec) const;
      void inv_ica(unsigned int index, math::vertex<T>& vec) const;      
      
      
      ////////////////////////////////////////////////////////////
      // individually labelled datasets classes
      
      struct cluster
      {
	std::string cname; // data class name;
	unsigned int cindex; // cluster index
	
	unsigned int data_dimension; // data dimension
	std::vector< math::vertex<T> > data; // vectors
	
	std::vector<enum data_normalization> preprocessings; // done to data
	
	// mean & variance removal parameters
	math::vertex<T> mean;
	math::vertex<T> variance;
	
	T softmax_parameter;
	
	math::matrix<T> Rxx; // correlation matrix
	math::matrix<T> Wxx, invWxx; // whitening matrix (calculated from R)
	math::matrix<T> ICA; // ICA solution;
	math::matrix<T> invICA;
      };
      
      
      std::vector<cluster> clusters;
      std::map<std::string, unsigned int> namemapping;
      
      
      
      // fileformat tag
      static const char* FILEID_STRING;
      
    };
  
  
  //////////////////////////////////////////////////////////////////////
  
  extern template class dataset< whiteice::math::blas_real<float> >;
  extern template class dataset< whiteice::math::blas_real<double> >;
  extern template class dataset< float >;
  extern template class dataset< double >;
  
}




#endif

