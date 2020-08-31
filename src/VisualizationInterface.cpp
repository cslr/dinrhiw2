
#include "VisualizationInterface.h"
#include "correlation.h"
#include "blade_math.h"


namespace whiteice
{

  // REAL FUNCTION: plots data points using virtual functions
  bool VisualizationInterface::adaptiveScatterPlot
  (const std::vector< math::vertex< math::blas_real<float> > >& points)
  {
    // calculates mean and covariance of points
    whiteice::math::vertex< math::blas_real<float> > mx;    
    whiteice::math::matrix< math::blas_real<float> > Cxx;

    if(whiteice::math::mean_covariance_estimate(mx, Cxx, points) == false)
      return false;

    const unsigned int XMAX = this->getScreenX();
    const unsigned int YMAX = this->getScreenY();

    // calculates variance removal scaling
    whiteice::math::vertex< math::blas_real<float> > scale;
    const math::blas_real<float> epsilon(1e-12);
    scale.resize(mx.size());
    
    for(unsigned int k=0;k<mx.size();k++){
      scale[k] = whiteice::math::blas_real<float>(1.0f)/whiteice::math::sqrt(whiteice::math::abs(Cxx(k,k))+epsilon); 
    }

    // plots points by removing mean and variance
    for(unsigned int i=0;i<points.size();i++)
    {
      // removes mean
      auto z = points[i] - mx;

      // divides by sqrt(var)
      for(unsigned int k=0;k<z.size();k++)
	z[k] *= scale[k];

      // data is now N(0,I), keeps [-2,+2] range within window so
      // (z+2.0)/4.0 is in [0,1] range and we clip smaller/larger values

      for(unsigned int k=0;k<z.size();k++){
	z[k] = (z[k]+math::blas_real<float>(2.0f))/math::blas_real<float>(4.0f);

	if(z[k] < math::blas_real<float>(0.0f))
	  z[k] = math::blas_real<float>(0.0f);
	else if(z[k] > math::blas_real<float>(1.0f))
	  z[k] = math::blas_real<float>(1.0f);
      }

      float x0 = XMAX/2.0f;
      float y0 = YMAX/2.0f;

      if(z.size() >= 1){
	whiteice::math::convert(x0, z[0]);
	x0 = x0*(XMAX - 1);
      }
      if(z.size() >= 2){
	whiteice::math::convert(y0, z[1]);
	y0 = y0*(YMAX - 1);
      }

      const unsigned int xi = (unsigned int)whiteice::math::round(x0);
      const unsigned int yi = (unsigned int)whiteice::math::round(y0);

      if(this->plot(xi, yi) == false) return false;
    }
    
    return true;
  }

  
}
