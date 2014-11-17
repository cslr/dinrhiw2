/*
 * radial basis function (RBF)
 * 
 * - not fully implemented yet
 */

#ifndef rbf_h
#define rbf_h

#include "activation_function.h"

namespace whiteice
{

  template <typename T>
    class RBF
    {
    public:
      // creates RBF with size number of radial basis (sp?)
      RBF(unsigned int size);
      RBF(const activation_function< std::vector<T> >& F);
      ~RBF();
      
      bool set_activation(const activation_function< std::vector<T> >& F);
      activation_function< std::vector<T> > get_actication();
      
    private:
      std::vector< std::vector<T> > points;
      activation_function< std::vector<T> >* F;
  
    };
}

#include "rbf.cpp"

#endif
