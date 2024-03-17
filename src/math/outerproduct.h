
#ifndef __whiteice_outerproduct_h
#define __whiteice_outerproduct_h

#include "dinrhiw_blas.h"


namespace whiteice
{
  namespace math
  {

    template <typename T> class matrix;
    template <typename T> class vertex;

    // optimized outerproduct: A += scalar*a*b^T
    template <typename T>
      bool addouterproduct(matrix<T>& A,
			   const T& scalar,
			   const vertex<T>& a,
			   const vertex<T>& b);


    
    extern template bool addouterproduct<float>(matrix<float>& A,
						const float& scalar,
						const vertex<float>& a,
						const vertex<float>& b);

    extern template bool addouterproduct<double>(matrix<double>& A,
						 const double& scalar,
						 const vertex<double>& a,
						 const vertex<double>& b);
    
    extern template bool addouterproduct< blas_real<float> >(matrix< blas_real<float> >& A,
							     const blas_real<float>& scalar,
							     const vertex< blas_real<float> >& a,
							     const vertex< blas_real<float> >& b);
    
    extern template bool addouterproduct< blas_real<double> >(matrix< blas_real<double> >& A,
							      const blas_real<double>& scalar,
							      const vertex< blas_real<double> >& a,
							      const vertex< blas_real<double> >& b);
    
  };
};


#include "matrix.h"
#include "vertex.h"


#endif
