
#ifndef GHA_cpp
#define GHA_cpp

#include "gha.h"
#include <stdlib.h>

using namespace std;


template <typename T>
GHA<T>::GHA(const T& learning_rate)
{
  this->learning_rate = learning_rate;
  iterations = 0;
  delta.resize(0);
  weights.resize(0);
  output.resize(0);  
}


template <typename T>
GHA<T>::~GHA(){ }


/*
 * resets GHA structure for new data
 *
 * input_size  - dimension of input  vectors
 * output_size - dimension of output vectors
 */
template <typename T>
bool GHA<T>::reset(unsigned int input_size,
		   unsigned int output_size) throw()
{
  if(input_size == 0 || output_size == 0)
    return false;

  if(output_size > input_size)
    return false;

  try{

    delta.resize(output_size);
    weights.resize(output_size);
    output.resize(output_size);

    for(unsigned a = 0;a<output_size;a++)
      output[a] = static_cast<T>(0.0);

    iterations = 0;

    typename vector< std::vector<T> >::iterator i = weights.begin();
    typename vector< std::vector<T> >::iterator k = delta.begin();
    
    for(;i!=weights.end();i++,k++){
      (*i).resize(input_size);
      (*k).resize(input_size);
      
      typename vector<T>::iterator j = (*i).begin();
      typename vector<T>::iterator l = (*k).begin();
      
      for(;j!=(*i).end();j++,l++){
	*j = static_cast<T>(2.0*((float)rand())/((float)RAND_MAX)- 1.0);
	*l = static_cast<T>(0.0);
      }
    }

    
    normalize_weights(0.50);
    
    return true;

  }
  catch(exception& e){
    return false;
  }
  
}


/*
 * basic simple GHA implementation
 */
template <typename T>
bool GHA<T>::train(const std::vector<T>& input) throw(std::out_of_range)
{
  if(weights.size() <= 0) return false;
  if(input.size() != weights[0].size()) return false;

  iterations++;

  T rate = static_cast<T>(1.0/((float)iterations));

  
  /* calculates output */
  for(unsigned int j=0;j<output.size();j++){

    output[j] = static_cast<T>(0.0);
        
    for(unsigned int i=0;i<weights[j].size();i++){
      output[j] += weights[j][i]*input[i];
    }
  }

  
  /* updates weights */
  for(unsigned int j=0;j<output.size();j++)
  {
    T sum;     

    for(unsigned int i=0;i<weights[j].size();i++){

      sum = static_cast<T>(0.0);

      for(unsigned int k=0;k<=j;k++)
	sum += weights[k][i]*output[k];

      delta[j][i] = learning_rate * rate * output[j] * ( input[i] - sum );
	
      // cout << "(" << i << " , " << j << ") " << delta[j][i] << endl;
    }

  }

  for(unsigned int j=0;j<weights.size();j++){
    for(unsigned int i=0;i<weights[j].size();i++){
      weights[j][i] += delta[j][i]; // update
    }
  }

  // normalize_weights(1.00);

  return true;
}


template <typename T>
bool GHA<T>::normalize_weights(T factor)
{
  /* renormalize weights / algoritm isn't numerically stable in practice? */

  for(unsigned int j=0;j<weights.size();j++){
    T len = static_cast<T>(0.0);

    for(unsigned int i=0;i<weights[j].size();i++)
      len += weights[j][i]*weights[j][i];

    len = sqrt(len)/factor;

    if(len != 0){
    
      for(unsigned int i=0;i<weights[j].size();i++)
	weights[j][i] /= len;
    }
  }
  
  return true;
}


/*
 * NEW
 * calculates ||dW|| kins of Froberious(?)-norm
 *  - from above to zero.
 *
 * OLD
 * calculates squared cos(angle) and
 * then takes square root between
 * vectors. in case of total converge results
 * should be 1. (convergers from above)
 */
template <typename T>
T GHA<T>::estimate_convergence()
{
  T temp = 0;
  
  for(unsigned int j=0;j<delta.size();j++){
    for(unsigned int i=0;i<delta[j].size();i++){
      temp += delta[j][i]*delta[j][i];
    }
  }
  
  return temp;

#if 0
  T temp2 = 0;

  for(unsigned int j=0;j<weights.size();j++){
    

    for(unsigned int i=0;i<weights.size();i++){
      T temp = 0;
      for(unsigned int k=0;k<weights[i].size();k++){
	temp += weights[j][k]*weights[i][k];
      }

      temp2 += temp*temp;      
    }
  }

  temp2 /= (T)weights.size();
  temp2 = sqrt(temp2);
  
  return temp2;
#endif
}


template <typename T>
bool GHA<T>::batch_train(const std::vector< const std::vector<T> >& input)
  throw(std::out_of_range)
{
  typename vector< const vector<T> >::const_iterator i = input.begin();
  bool result = true;

  for(;i!=input.end();i++){
    if(!train(*i))
      result = false;
  }

  return result;
}


template <typename T>
bool GHA<T>::code(const std::vector<T>& input,
			  std::vector<T>& coded) throw(std::out_of_range)
{
  if(weights.size() <= 0) return false;
  if(input.size() != weights[0].size()) return false;


  coded.resize(output.size());

  for(unsigned int i=0;i<coded.size();i++){
    coded[i] = static_cast<T>(0.0);

    for(unsigned int j=0;j<weights[i].size();j++){
      coded[i] += static_cast<T>
	(weights[i][j] * input[j]);
    }
  }

  return true;
}


template <typename T>
bool GHA<T>::encode(const std::vector<T>& coded,
		    std::vector<T>& encoded) throw(std::out_of_range)
{
  if(weights.size() <= 0) return false;

  encoded.resize(weights[0].size());

  for(unsigned int i=0;i<weights[0].size();i++){
    encoded[i] = static_cast<T>(0.0);
    
    for(unsigned int j=0;j<weights.size();j++){
      encoded[i] += static_cast<T>
	(weights[j][i] * coded[j]);
    }
  }

  return true;
  
}


/* returns input vector size */
template <typename T>
unsigned int GHA<T>::size() throw()
{
  if(weights.size() <= 0) return 0;

  return weights[0].size();
}


/* returns output size */
template <typename T>
unsigned int GHA<T>::output_size() throw()
{
  return output.size();
}


template <typename T>
std::vector<T> GHA<T>::eigenvector(unsigned int index) 
  throw(std::out_of_range)
{
  if(weights.size() <= 0)
    throw std::out_of_range("no weights");
  if(index >= weights[0].size())
    throw std::out_of_range("eigenvector index too big");

  return weights[index];
  
}


//template <typename T>
//T GHA<T>::eigenvalue(unsigned int index) throw(std::out_of_range)
//{
//
//}


#endif

