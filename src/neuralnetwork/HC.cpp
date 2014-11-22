
#include "HC.h"
#include "KMeans.h"

namespace whiteice
{
  
  template < typename Parameters, typename T>
  HC<Parameters, T>::HC(){ data = 0; }
  
  template < typename Parameters, typename T>
  HC<Parameters, T>::~HC(){ }
  
  
  template < typename Parameters, typename T>
  bool HC<Parameters, T>::clusterize()
  {
    // 0. Prelimilinary checks
    
    if(data == 0 || logic == 0)
      return false;
    
    if(data->size() < 3)
      return false;
    
    for(unsigned int i=0;i<nodes.size();i++)
      if(!nodes[i]) delete[] nodes[i];
    
    nodes.resize(0);
    
    // 1. Creates initial clusters
    //    Each cluster will have approximatedly 100 points assigned to it
    
    {
      KMeans<T> KM;
      
      if(data->size() > 1000){
	if(KM.learn((data->size())/100 + 1, *data) == false)
	  return false;
      
	nodes.resize((data->size())/100 + 1);
      }
      else{
	if(KM.learn((data->size())/2 + 1, *data) == false)
	  return false;
      
	nodes.resize((data->size())/2 + 1);
      }
      
      for(unsigned int i=0;i<nodes.size();i++){
	nodes[i] = new whiteice::hcnode<Parameters, T>;
	nodes[i]->data = new std::vector< whiteice::math::vertex<T> >;
      }
      
      // assigns data to initial clusters
      
      for(unsigned int i=0;i<data->size();i++){
	T closest = distance(KM[0], (*data)[i]);
	unsigned int index = 0;
	
	for(unsigned int j=0;j<nodes.size();j++){
	  T tmp = distance(KM[j], (*data)[i]);
	  if(tmp < closest){
	    closest = tmp;
	    index = j;
	  }
	}
	
	nodes[index]->data->push_back((*data)[i]);
      }
      
      // calculates parameters of initial clusters
      
      for(unsigned int i=0;i<nodes.size();i++)
	if(logic->initialize(nodes[i]) == false)
	  return false;
    }
    
    
    // 2. Calculates initial similarities
    
    {
      s.resize((nodes.size() * (nodes.size() - 1))/2);
      unsigned int i0 = 0, j0 = 1;
      
      for(unsigned int i=0;i<s.size();i++){
	s[i] = logic->similarity(nodes, i0, j0);
	
	i0++;
	if(i0 >= j0){
	  i0 = 0;
	  j0++;
	}
      }
    }
	  
    
    // 3. (Loop) Merges the most similar clusters and updates similarity matrix/vector
    
    while(nodes.size() > 1){
      // finds the most similar clusters
      
      T biggest = s[0];
      unsigned int ci = 0, cj = 1;
      unsigned int i0 = 0;
      unsigned int j0 = 1;
      
      for(unsigned int i=0;i<s.size();i++){
	if(s[i] > biggest){
	  biggest = s[i];
	  ci = i0;
	  cj = j0;
	}
	
	i0++;
	
	if(i0 >= j0){
	  i0 = 0;
	  j0++;
	}
      }
      
      // merges clusters
      
      whiteice::hcnode<Parameters, T>* node = new whiteice::hcnode<Parameters, T>();
      
      if(!logic->merge(nodes[ci], nodes[cj], *node))
	return false;
      
      nodes[ci] = node; // nodes[ci] = new
      nodes[cj] = nodes[nodes.size()-1];      
      
      // updates similarity vector s
      
      s.resize(s.size() - (nodes.size() - 1));
      nodes.resize(nodes.size()-1); // nodes[cj] = nodes[last]
      
      // recalculates half-rows ci and cj
      
      for(unsigned int i=0;i<ci;i++)
	s[i + (ci*(ci - 1))/2] = logic->similarity(nodes, i, ci);
      
      for(unsigned int i=0;i<cj;i++)
	s[i + (cj*(cj - 1))/2] = logic->similarity(nodes, i, cj);
      
      // recalculates half columns ci and cj
      
      if(ci + 1 < nodes.size())
	for(unsigned int j=ci+1;j<nodes.size();j++)
	  s[ci + (j*(j - 1))/2] = logic->similarity(nodes, ci, j);
      
      if(cj + 1 < nodes.size())
	for(unsigned int j=cj+1;j<nodes.size();j++)
	  s[cj + (j*(j - 1))/2] = logic->similarity(nodes, cj, j);
    }
    
    
    return true;
  }
  
  
  
  template < typename Parameters, typename T>
  T HC<Parameters, T>::distance(const  std::vector<T>& v0, const  std::vector<T>& v1) const throw()
  {
    const unsigned int SIZE = v0.size();
    T result = T(0.0f);
    
    for(unsigned int i=0;i<SIZE;i++)
      result += v0[i]*v1[i];
    
    return whiteice::math::sqrt(result);
  }

  
  template < typename Parameters, typename T>
  T HC<Parameters, T>::distance(const math::vertex<T>& v0, const math::vertex<T>& v1) const throw()
  {
    const unsigned int SIZE = v0.size();
    T result = T(0.0f);
    
    for(unsigned int i=0;i<SIZE;i++)
      result += v0[i]*v1[i];
    
    return whiteice::math::sqrt(result);
  }
  
  
  template < typename Parameters, typename T>
  T HC<Parameters, T>::distance(const  std::vector<T>& v0, const math::vertex<T>& v1) const throw()
  {
    const unsigned int SIZE = v0.size();
    T result = T(0.0f);
    
    for(unsigned int i=0;i<SIZE;i++)
      result += v0[i]*v1[i];
    
    return whiteice::math::sqrt(result);
  }
  
  
  template < typename Parameters, typename T>
  T HC<Parameters, T>::distance(const math::vertex<T>& v0, const  std::vector<T>& v1) const throw()
  {
    const unsigned int SIZE = v0.size();
    T result = T(0.0f);
    
    for(unsigned int i=0;i<SIZE;i++)
      result += v0[i]*v1[i];
    
    return whiteice::math::sqrt(result);
  }
  
  
  
  
  ////////////////////////////////////////////////////////////
  // explicit template instantations
  
  template class HC<GDAParams, math::blas_real<float> >;
  
};
