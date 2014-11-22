
#include "GDALogic.h"

namespace whiteice
{
  
  
  bool GDALogic::initialize(whiteice::hcnode<GDAParams, whiteice::math::blas_real<float> >* node) throw()
  {
    if(node->data == 0)
      return false;
    
    if(node->data->size() <= 0)
      return false;
    
    node->childs.clear();
    
    const unsigned int DIM = (*(node->data))[0].size();
    
    node->p.mean.resize(DIM);
    node->p.cov.resize(DIM, DIM);
    
    if(!whiteice::math::mean_covariance_estimate(node->p.mean, node->p.cov, *(node->data)))
      return false;
    
    return true;
  }
  
  
  whiteice::math::blas_real<float>
  GDALogic::similarity(std::vector< hcnode<GDAParams, whiteice::math::blas_real<float> >* > clusters,
		       unsigned int i, unsigned int j) throw()
  {
    // calculates mutual overlap loss [as described in docs/GDA.lyx]
    // after merge of clusters (i and j).
    
    // calculates merged cluster: i U j
    whiteice::hcnode<GDAParams, whiteice::math::blas_real<float> > merged;
    merge(clusters[i], clusters[j], merged);
    
    whiteice::math::blas_real<float> result = 0.0f; 
    
    if(j < i) std::swap(i,j);
    
    for(unsigned int a=0;a<i;a++)
      result += overlap(clusters[a]->p, clusters[i]->p) + 
	        overlap(clusters[a]->p, clusters[j]->p) - 
	        overlap(clusters[a]->p, merged.p);
    
    // returns negative value because *minimum* mutual
    // overlap loss should be selected
    return (-result);
  }
  
  
  
  bool GDALogic::merge(whiteice::hcnode<GDAParams, whiteice::math::blas_real<float> >* p1,
		       whiteice::hcnode<GDAParams, whiteice::math::blas_real<float> >* p2,
		       whiteice::hcnode<GDAParams, whiteice::math::blas_real<float> >& result) const throw()
  {
    // if(p1 == 0 || p2 == 0) return false; // (no safety checks)
    
    // calculates new parameters [as described in docs/GDA.lyx]
    
    result.p.n    = p1->p.n + p2->p.n;
    whiteice::math::blas_real<float> c1, c2, c3;
    
    c1 = ((float)p1->p.n)/((float)result.p.n);
    c2 = ((float)p2->p.n)/((float)result.p.n);
    c3 = (((float)(p1->p.n * p2->p.n))/(float)(result.p.n*result.p.n)); 
    
    whiteice::math::vertex< whiteice::math::blas_real<float> > delta(p1->p.mean - p2->p.mean);
    
    result.p.mean = c1 * (p1->p.mean) + c2 * ( p2->p.mean );
    result.p.cov  = c1*(p1->p.cov) + c2*(p2->p.cov) + c3 * delta.outerproduct();
    
    return true;
  }
  
  
  whiteice::math::blas_real<float> GDALogic::overlap(const GDAParams& c1, const GDAParams& c2) const throw()
  {
    // calculates pseudooverlap (logarithm of overlap between clusters)
    
    whiteice::math::matrix< whiteice::math::blas_real<float> > A(c1.cov), B(c2.cov), C;
    A.inv(); B.inv();
    C = A + B;
    
    whiteice::math::blas_real<float> result = 0.0f;
    
    result += (c1.mean.size()/-2.0f) * 
      log(whiteice::math::blas_real<float>(2.0f * M_PI) * c1.cov.det() * c2.cov.det() * C.det());
    
    result -= 0.50f * ( c1.mean * (A * c1.mean) )[0];
    result -= 0.50f * ( c2.mean * (B * c2.mean) )[0];
    
    C.inv();
    
    whiteice::math::vertex< whiteice::math::blas_real<float> > m;
    
    m = A*c1.mean + B*c2.mean;
    
    result += 0.50f * ( m * ( C * m ) )[0];
    
    return result;
  }
  
};
