/*
 * basic GHA PCA
 */

#ifndef GHA_h
#define GHA_h

#include "pca.h"

template <typename T>
class GHA : public PCA<T>
{
 public:

  GHA(const T& learning_rate=0.80);
  virtual ~GHA();
  
  bool reset(unsigned int input_size,
	     unsigned int output_size=0) throw();
  
  bool train(const std::vector<T>& input)
    throw(std::out_of_range);
  bool batch_train(const std::vector< const std::vector<T> >& input)
    throw(std::out_of_range);
  
   bool code(const std::vector<T>& input,
	     std::vector<T>& coded)
     throw(std::out_of_range);
   
   bool encode(const std::vector<T>& coded,
	       std::vector<T>& encoded)
    throw(std::out_of_range);

   /* returns input vector size */
   unsigned int size() throw();
   
  /* returns output size */
   unsigned int output_size() throw();
   
   std::vector<T> eigenvector(unsigned int index)
     throw(std::out_of_range);
   
   // T eigenvalue(unsigned int index) throw(std:out_of_range);

   /* 
    * returns 1 when PCA vectors are orthonormal
    */
   T estimate_convergence();
   
 private:

   bool normalize_weights(T factor = 1.0);
   
   std::vector< std::vector<T> > weights;   
   std::vector< T > output;

   std::vector< std::vector<T> > delta; // for training (delta weights)
   
   T learning_rate;
   
   unsigned int iterations;
};

#include "gha.cpp"

#endif

