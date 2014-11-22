
#ifndef hctree_h
#define hctree_h

#include <vector>
#include "dinrhiw.h"

namespace whiteice
{
  template <typename Parameters, typename T = whiteice::math::blas_real<float> >
    class hcnode
    {
      public:
      
      hcnode(){ data = 0; }
      ~hcnode(){ if(data) delete data; }
      
      // cluster parameters
      Parameters p;
      
      // data values (in leaf nodes)
      std::vector< whiteice::math::vertex<T> >* data;
      
      // child nodes
      std::vector< whiteice::hcnode<Parameters, T>* > childs;
    };
  
};



#endif

