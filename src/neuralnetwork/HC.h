/*
 * Hierarchical bottom-up clustering
 * 
 * Tomas Ukkonen <tomas.ukkonen@iki.fi> 2004
 *
 * generic hierarchical clustering class with user definable
 * similarity metric and cluster parameters 
 * (through abstract class)
 * 
 */

#ifndef HC_h
#define HC_h

#include "dinrhiw.h"
#include "HCLogic.h"

#include "hctree.h"

#include <vector>


namespace whiteice
{
  
  template < 
    typename Parameters,
    typename T = whiteice::math::blas_real<float>
    >
    class HC
    {
      public:
      
      HC();
      ~HC();
      
      bool clusterize();
      
      inline void setData(std::vector< math::vertex<T> >* data) 
      { this->data = data; }
      
      inline void setProgramLogic(whiteice::HCLogic<Parameters, T>* logic)
      { this->logic = logic; }
      
      inline const std::vector< whiteice::hcnode<Parameters, T>* >& getNodes() 
      { return nodes; }
      
      private:
      
      T distance(const  std::vector<T>& v0, const  std::vector<T>& v1) const ;
      T distance(const math::vertex<T>& v0, const math::vertex<T>& v1) const ;
      T distance(const  std::vector<T>& v0, const math::vertex<T>& v1) const ;
      T distance(const math::vertex<T>& v0, const  std::vector<T>& v1) const ;
      
      std::vector< math::vertex<T> >* data;
      std::vector< hcnode<Parameters, T>* > nodes;
      HCLogic<Parameters, T>* logic;
      
      // similarity vertex s where entry s[index(i,j)] tells similarity
      // between nodes i and j (j != i and i < j so (i,j) = (0,0) is not possible)
      // index(i,j) = { if(i > j) swap(i,j); index(i,j) = i + j*(j - 1)/2; }
      whiteice::math::vertex<T> s;
    };
};


#include "GDALogic.h"

namespace whiteice
{
  
  extern template class HC<GDAParams, math::blas_real<float> >;
  
};


#endif

