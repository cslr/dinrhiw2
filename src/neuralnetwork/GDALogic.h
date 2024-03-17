/*
 * Clustering logic for GDA clustering with mutual overlap loss based
 * similarity criterion.
 * 
 */

#ifndef GDALogic_h
#define GDALogic_h

#include "dinrhiw.h"
#include "HCLogic.h"


namespace whiteice
{
  struct GDAParams
  {
    unsigned int n; // number of data points
    
    whiteice::math::vertex< whiteice::math::blas_real<float> > mean;
    whiteice::math::matrix< whiteice::math::blas_real<float> > cov;
  };
  
  
  
  class GDALogic : public HCLogic< GDAParams, whiteice::math::blas_real<float> >
  {
  public:
    
    virtual ~GDALogic(){ }
    
    bool initialize(whiteice::hcnode<GDAParams, whiteice::math::blas_real<float> >* p) ;
    
    whiteice::math::blas_real<float> similarity
      (std::vector< hcnode<GDAParams, whiteice::math::blas_real<float> >* > clusters,
       unsigned int i, unsigned int j) ;
    
    bool merge(whiteice::hcnode<GDAParams, whiteice::math::blas_real<float> >* p1,
	       whiteice::hcnode<GDAParams, whiteice::math::blas_real<float> >* p2,
	       whiteice::hcnode<GDAParams, whiteice::math::blas_real<float> >& result) const ;
    
    whiteice::math::blas_real<float> overlap(const GDAParams& c1, const GDAParams& c2) const ;
    
  };
  
};

#endif

