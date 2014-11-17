/*
 * general PCA interface class
 * for pca data 
 */

#ifndef pca_h
#define pca_h

#include <vector>
#include <stdexcept>
#include <exception>

template <typename T=double>
class PCA
{
  public:
  
  virtual bool reset(unsigned int input_size,
		     unsigned int output_size=0) = 0;
  
  /* use this to feed data incrementally */
  virtual bool train(const std::vector<T>& input) throw(std::out_of_range) = 0;

  /* for a big data batch */
  virtual bool batch_train(const std::vector< const std::vector<T> >& input) throw(std::out_of_range) = 0;
  
  virtual bool code(const std::vector<T>& input,
		    std::vector<T>& coded) throw(std::out_of_range) = 0;
  
  virtual bool encode(const std::vector<T>& coded,
		      std::vector<T>& encoded) throw(std::out_of_range) = 0;

  /* returns input vector size */
  virtual unsigned int size() = 0;
  
  /* returns output size */
  virtual unsigned int output_size() = 0;
  
};




#endif

