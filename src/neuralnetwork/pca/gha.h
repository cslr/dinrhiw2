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
	     unsigned int output_size=0) ;
  
  bool train(const std::vector<T>& input)
    ;
  bool batch_train(const std::vector< const std::vector<T> >& input)
    ;
  
   bool code(const std::vector<T>& input,
	     std::vector<T>& coded)
     ;
   
   bool encode(const std::vector<T>& coded,
	       std::vector<T>& encoded)
    ;

   /* returns input vector size */
   unsigned int size() ;
   
  /* returns output size */
   unsigned int output_size() ;
   
   std::vector<T> eigenvector(unsigned int index)
     ;
   
   // T eigenvalue(unsigned int index) ;

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

