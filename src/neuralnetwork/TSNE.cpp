/*
 * t-SNE dimension reduction algorithm.
 * Mathmatical implementation notes/documentation in docs/TSNE_notes.tm (texmacs)
 * Tomas Ukkonen. 2020.
 * 
 */

#include "TSNE.h"


namespace whiteice
{


  template <typename T>
  TSNE<T>::TSNE()
  {
    
  }

  template <typename T>
  TSNE<T>::TSNE(const TSNE<T>& tsne)
  {
    
  }
  
  // dimension reduces samples to DIM dimensional vectors using t-SNE algorithm
  template <typename T>
  bool TSNE<T>::calculate(const std::vector< math::vertex<T> >& samples,
			  const unsigned int DIM,
			  std::vector< math::vertex<T> >& results)
  {
    if(DIM <= 0) return false;
    if(samples.size() <= 0) return false;
    
    // perplexity of 30 seem to give reasonable good results
    const T PERPLEXITY = T(30.0f);

    whiteice::RNG<T> rng;

    // calculate p-values
    std::vector< std::vector<T> > pij;
    
    if(calculate_pvalues(samples, PERPLEXITY, pij) == false)
      return false;
    
    // initializes y values as gaussian unit variance around zero
    std::vector< math::vertex<T> > yvalues;
    
    yvalues.resize(samples.size());
    
    for(auto& y : yvalues){
      y.resize(DIM);
      rng.normal(y);
    }

    
    // gradient descent search of yvalues (TODO: NOT IMPLEMENTED YET)
    {
      std::vector< std::vector<T> > qij;
      T qsum;

      // calculates qij values
      if(calculate_qvalues(yvalues, qij, qsum) == false)
	return false;
      
      // does gradient descent
      assert(0); // IMPLEMENT ME
    }

    return true;
  }

  // calculates p values for pj|i where i = index and sigma2 for index:th vector is given
  template <typename T>
  bool TSNE<T>::calculate_pvalue_given_sigma(const std::vector< math::vertex<T> >& x,
					     const unsigned int index, // to x vector
					     const T sigma2,
					     std::vector<T>& pj)
  {
    if(index >= x.size()) return false;
    
    pj.resize(x.size());
    
    math::vertex<T> delta;
    delta.resize(x[0].size());
    delta.zero();
    
    T rsum = T(0.0f); // calculates rsum for this index

    for(unsigned int k=0;k<x.size();k++){
      if(index == k) continue;
      delta = x[k] - x[index];
      
      T v = -(delta*delta)[0]/sigma2;

      rsum += whiteice::math::exp(v);
    }
      
    for(unsigned int j=0;j<x.size();j++){
      if(index == j) continue;

      delta = x[j] - x[index];
      T v = -(delta*delta)[0]/sigma2;

      pj[j] = whiteice::math::exp(v)/rsum;
    }

    pj[index] = T(0.0f);

    return true;
  }


  // calculates distribution's perplexity
  template <typename T>
  T TSNE<T>::calculate_perplexity(const std::vector<T>& pj)
  {
    if(pj.size() == 0) return T(1.0f);

    T H = T(0.0f); // entropy

    for(const auto& p : pj){
      if(p > T(0.0f))
	H += -p*math::log(p)/math::log(T(2.0f));
    }

    const T perplexity = math::pow(T(2.0f),H);

    return perplexity;
  }


  // calculate x samples probability distribution p values
  // estimates sigma^2 variance parameter using distribution perplexity values
  template <typename T>
  bool TSNE<T>::calculate_pvalues(const std::vector< math::vertex<T> >& x,
				  const T perplexity,
				  std::vector< std::vector<T> >& pij)
  {
    if(x.size() == 0) return false;
    

    // calculates p_j|i values with given perplexity
    // perplexity is used to estimate sigma_i parameter of rji/p_j|i
    std::vector< std::vector<T> > rji;
    rji.resize(x.size());

    T total_var = T(0.0f);
    
    // estimates maximum variance: used to search for optimal sigma value
    {
      math::vertex<T> mx;
      math::matrix<T> Cxx;

      if(math::mean_covariance_estimate(mx, Cxx, x) == false)
	return false;

      total_var = T(0.0f);

      for(unsigned int i=0;i<Cxx.xsize();i++)
        total_var += Cxx(i,i);

      total_var /= T(x[0].size());
    }

    const T TOTAL_SIGMA2 = total_var;  // total variance in data
    const T MIN_SIGMA2   = TOTAL_SIGMA2/T(1000000000.0f);
    
    
    for(unsigned int j=0;j<x.size();j++){
      
      // searches for sigma2 variance value for x[j] centered distribution
      
      rji[j].resize(x.size());
      
      T sigma2_min = MIN_SIGMA2;
      T sigma2_max = TOTAL_SIGMA2; // maximum sigma value
      T perp_min, perp_max;

      std::vector<T> pj;
      
      {
	// searches for minimum sigma2 value
	calculate_pvalue_given_sigma(x, j, sigma2_min, pj);
	perp_min = calculate_perplexity(pj);
	
	while(perp_min > perplexity && sigma2_min > T(0.0f)){
	  sigma2_min /= T(2.0f);
	  calculate_pvalue_given_sigma(x, j, sigma2_min, pj);
	  perp_min = calculate_perplexity(pj);	  
	}

	if(perp_min > perplexity)
	  return false; // could not find minimum value

	// searches for maximum sigma2 value
	calculate_pvalue_given_sigma(x, j, sigma2_max, pj);
	perp_max = calculate_perplexity(pj);
	
	while(perp_max < perplexity && sigma2_max < T(10e9)){
	  sigma2_max *= T(2.0f);
	  calculate_pvalue_given_sigma(x, j, sigma2_max, pj);
	  perp_max = calculate_perplexity(pj);
	}

	if(perp_max < perplexity)
	  return false; // could not find maximum value
      }

      // no we have minimum and maximum value for sigma2 variance term
      // search for sigma2 term with target perplexity
      {
	T sigma2_next = (sigma2_min + sigma2_max)/T(2.0f);
	
	calculate_pvalue_given_sigma(x, j, sigma2_next, pj);
	T perp_next = calculate_perplexity(pj);

	while(math::abs(perplexity - perp_next) > T(0.10f) &&
	      (sigma2_max - sigma2_min) > T(10e-9)){
	  if(perplexity < perp_next){
	    sigma2_max = sigma2_next;
	  }
	  else{ // perplexity > perp_next
	    sigma2_min = sigma2_next;
	  }

	  sigma2_next = (sigma2_min + sigma2_max)/T(2.0f);
	
	  calculate_pvalue_given_sigma(x, j, sigma2_next, pj);
	  perp_next = calculate_perplexity(pj);
	}

	if(sigma2_max - sigma2_min <= T(10e-9))
	  return false; // could not find target perplexity

	// we found sigma variance and probability distribution with target perplexity
	rji[j] = pj;
	rji[j][j] = T(0.0f);
      }
      
    }

    
    // calculates symmetric p-values
    pij.resize(x.size());

    for(unsigned int i=0;i<x.size();i++){
      pij[i].resize(x.size());

      for(unsigned int j=0;j<x.size();j++){
	if(i == j) continue;

	pij[i][j] = (rji[i][j] + rji[j][i])/T(2.0f*x.size());
      }

      pij[i][i] = T(0.0f);
    }

    return true;
  }


  // calculate dimension reduced y samples probability distribution values q
  template <typename T>
  bool TSNE<T>::calculate_qvalues(const std::vector< math::vertex<T> >& y,
				  std::vector< std::vector<T> >& qij, T& qsum)
  {
    if(y.size() == 0) return false;
    
    // calculates qij terms
    qij.resize(y.size());

    {
      math::vertex<T> delta;
      delta.resize(y[0].size());
      delta.zero();
      
      qsum = T(0.0f);
      
      for(unsigned int i=0;i<y.size();i++){
	for(unsigned int j=0;j<y.size();j++){
	  if(i == j) continue; // skip same values
	  delta = y[i] - y[j];
	  auto nrm2 = (delta*delta)[0];
	  qsum += T(1.0f)/(T(1.0f) + nrm2);
	}
      }
      
      for(unsigned int i=0;i<y.size();i++){
	qij[i].resize(y.size());
	for(unsigned int j=0;j<y.size();j++){
	  if(i == j) continue; // skip same values
	  delta = y[i] - y[j];
	  auto nrm2 = (delta*delta)[0];
	  qij[i][j] = ( T(1.0f)/(T(1.0f) + nrm2) ) / qsum;
	}

	qij[i][i] = T(0.0f);
      }
    }

    return true;
  }

  
  template <typename T>
  bool TSNE<T>::kl_divergence(const std::vector< std::vector<T> >& pij,
			      const std::vector< std::vector<T> >& qij,
			      T& klvalue)
  {
    if(pij.size() <= 0 || qij.size() <= 0) return false;
    if(pij.size() != pij[0].size()) return false;
    if(qij.size() != pij.size()) return false;
    if(qij.size() != qij[0].size()) return false;

    // calculates KL divergence
    klvalue = T(0.0f);
    
    for(unsigned int i=0;i<pij.size();i++){
      for(unsigned int j=0;j<pij.size();j++){
	if(i == j) continue; // skip zero values

	klvalue += pij[i][j]*whiteice::math::log(pij[i][j]/qij[i][j]);
      }
    }
    
    return true;
  }


  // calculates gradients of KL diverence
  template <typename T>
  bool TSNE<T>::kl_gradient(const std::vector< std::vector<T> >& pij,
			    const std::vector< std::vector<T> >& qij,
			    const T& qsum,
			    const std::vector< math::vertex<T> >& y,
			    std::vector< math::vertex<T> >& ygrad)
  {
    if(pij.size() != y.size()) return false;
    if(pij.size() <= 0 || y.size() <= 0) return false;
    if(pij.size() != pij[0].size()) return false;

    ygrad.resize(y.size());

    math::vertex<T> delta;
    delta.resize(y[0].size());
    
    // calculates gradient for each y value
    for(unsigned int m=0;m<ygrad.size();m++){
      ygrad[m].resize(y[0].size());
      ygrad[m].zero();

      // calculates gradient of each KL divergence term
      for(unsigned int i=0;i<y.size();i++){
	for(unsigned int j=0;j<y.size();j++){
	  if(i == j) continue; // skip zero/same values

	  const T scaling = -pij[i][j]/qij[i][j];

	  // gradient of qij term
	  if(m == i || m == j){
	    
	    // the first part of derivate
	    {
	      const T extra_scaling = T(1.0f)/qsum;
	      
	      if(m == i)
		delta = y[i] - y[j];
	      else
		delta = y[j] - y[i];
	      
	      T qterm = (T(1.0f) + (delta*delta)[0]);
	      qterm = T(-2.0f)/(qterm*qterm);
	      
	      ygrad[m] += scaling*extra_scaling*qterm*delta;
	    }

	    
	    // the second part of the derivate
	    {
	      delta = y[i] - y[j];
	      const T extra_scaling = -(T(1.0f)/(T(1.0f) + (delta*delta)[0]))/(qsum*qsum);
	      
	      // gradient
	      for(unsigned int l=0;l<y.size();l++){
		if(l == m) continue;
		delta = y[m] - y[l];
		
		T qterm = (T(1.0f) + (delta*delta)[0]);
		qterm = -T(4.0f)/(qterm*qterm);
		
		ygrad[m] += scaling*extra_scaling*qterm*delta;
	      }
	    }
	    
	  }
	  else{ // m != i || m != j

	    // no need for the first part of the derivate

	    // the second part of the derivate
	    {
	      delta = y[i] - y[j];
	      const T extra_scaling = -(T(1.0f)/(T(1.0f) + (delta*delta)[0]))/(qsum*qsum);
	      
	      // gradient
	      for(unsigned int l=0;l<y.size();l++){
		if(l == m) continue;
		delta = y[m] - y[l];
		
		T qterm = (T(1.0f) + (delta*delta)[0]);
		qterm = -T(4.0f)/(qterm*qterm);
		
		ygrad[m] += scaling*extra_scaling*qterm*delta;
	      }
	    }
	      
	  }
	}
      }
      
    }

    // calculated all qterms
    return true;
  }


  template class TSNE< math::blas_real<float> >;
  template class TSNE< math::blas_real<double> >;
  
};
