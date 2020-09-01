/*
 * t-SNE dimension reduction algorithm.
 * Mathmatical implementation notes/documentation in docs/TSNE_notes.tm (texmacs)
 * Tomas Ukkonen. 2020.
 * 
 */

#include "TSNE.h"

#include "linear_ETA.h"


namespace whiteice
{


  template <typename T>
  TSNE<T>::TSNE(const bool absolute_value)
  {
    // absolute value KL divergence should give better results (see TSNE_notes.tm)
    kl_absolute_value = absolute_value;
  }

  template <typename T>
  TSNE<T>::TSNE(const TSNE<T>& tsne)
  {
    this->kl_absolute_value = tsne.kl_absolute_value;
  }
  
  // dimension reduces samples to DIM dimensional vectors using t-SNE algorithm
  template <typename T>
  bool TSNE<T>::calculate(const std::vector< math::vertex<T> >& samples,
			  const unsigned int DIM,
			  std::vector< math::vertex<T> >& results,
			  const bool verbose,
			  LoggingInterface* const messages,
			  VisualizationInterface* const gui)
  {
    if(DIM <= 0) return false;
    if(samples.size() <= 0) return false;
    
    // perplexity of 30 seem to give reasonable good results
    const T PERPLEXITY = T(30.0f);

    whiteice::RNG<T> rng;

    char buffer[128];
    
    {
      float perplexityf = 0.0f;
      whiteice::math::convert(perplexityf, PERPLEXITY);
      snprintf(buffer, 128, "ESTIMATING P-VALUES WITH PERPLEXITY: %f\n",
	       perplexityf);

      if(verbose){
	printf(buffer);
	fflush(stdout);
      }

      if(messages != NULL)
	messages->printMessage(buffer);
    }

    // calculate p-values
    std::vector< std::vector<T> > pij;
    
    if(calculate_pvalues(samples, PERPLEXITY, pij) == false)
      return false;
    
    // initializes y values as gaussian unit variance around zero
    std::vector< math::vertex<T> > yvalues;
    T initial_ysigma = T(0.01f); // initial st.dev. is is 0.01 and var = 0.01^2
    
    yvalues.resize(samples.size());
    
    for(auto& y : yvalues){
      y.resize(DIM);
      rng.normal(y);
      // y *= initial_ysigma;
    }

    
    // gradient ascend search of yvalues
    {
      std::vector< std::vector<T> > qij;
      T qsum = T(0.0f);

      std::vector< math::vertex<T> > ygrad;
      ygrad = yvalues; // dummy initializer

      unsigned int iter = 0;
      const unsigned int MAXITER = 10000;

      unsigned int noimprove_counter = 0;
      const unsigned int MAXNOIMPROVE = 100;

      T lrate = T(100.0f);
      T best_klvalue = T(1000.0f);
      T klvalue = T(1000.0f);
      T prev_round_klvalue = T(1000.f);

      whiteice::linear_ETA<double> eta;
      eta.start(0.0, MAXITER);
      eta.update(0.0);

      if(gui && DIM >= 2){
	gui->show();
	gui->clear();
      }
      
      while(iter < MAXITER && noimprove_counter < MAXNOIMPROVE)
      {
	// calculates qij values
	if(calculate_qvalues(yvalues, qij, qsum) == false)
	  return false;
	
	// calculate and report the current KL divergence
	{
	  prev_round_klvalue = klvalue;
	  
	  if(kl_divergence(pij, qij, klvalue) == false)
	    return false;

	  if(iter == 0){
	    best_klvalue = klvalue;
	  }
	  
	  float klvaluef = 0.0f;
	  whiteice::math::convert(klvaluef, best_klvalue);

	  if(iter > 0){
	    snprintf(buffer, 128, "ITER %d / %d. (NOIMPROVE: %d/%d). BEST KL DIVERGENCE: %f (ETA %f hours)\n",
		     iter, MAXITER, noimprove_counter, MAXNOIMPROVE,
		     klvaluef, eta.estimate()/3600.0);
	  }
	  else{
	    snprintf(buffer, 128, "ITER %d / %d. (NOIMPROVE: %d/%d) BEST KL DIVERGENCE: %f\n",
		   iter, MAXITER, noimprove_counter, MAXNOIMPROVE,
		   klvaluef);	    
	  }

	  if(verbose){
	    printf(buffer);
	    fflush(stdout);
	  }

	  if(messages != NULL){
	    messages->printMessage(buffer);
	  }
	  
	}

	// check if solution improved
	{
	  if(klvalue <= best_klvalue){
	    best_klvalue = klvalue;
	    results = yvalues;
	    noimprove_counter = 0;
	  }
	  else{
	    noimprove_counter++;
	  }
	  
	  // adaptive learning rate
	  if(klvalue < prev_round_klvalue){
	    lrate *= T(1.5f); // solution improved so increase learning rate
	  }
	  else{ // solution worsened so use smaller step length
	    lrate /= T(1.5f);
	  }
	}

	if(gui && DIM >= 2){
	  // converts points to floating point values
	  std::vector< math::vertex< math::blas_real<float> > > points;
	  points.resize(results.size());

# pragma omp parallel for schedule(auto)
	  for(unsigned int j=0;j<results.size();j++){
	    points[j].resize(results[j].size());
	    for(unsigned int i=0;i<results[j].size();i++){
	      whiteice::math::convert(points[j][i], results[j][i]);
	    }
	  }
	  
	  gui->clear();
	  gui->adaptiveScatterPlot(points);
	  gui->updateScreen();
	}
      
	// calculate gradient
	// (minimize KL divergence to match distributions as well as possible)
	if(kl_gradient(pij, qij, qsum, yvalues, ygrad) == false)
	  return false;

#pragma omp parallel for schedule(auto)
	for(unsigned int i=0;i<ygrad.size();i++){
	  auto& y = yvalues[i];
	  const auto& g = ygrad[i];

	  y -= lrate*g;
	}

	iter++;
	eta.update(iter);
      }
      
    }

    
    return true;
  }

  // calculates p values for pj|i where i = index and sigma2 for index:th vector is given
  template <typename T>
  bool TSNE<T>::calculate_pvalue_given_sigma(const std::vector< math::vertex<T> >& x,
					     const unsigned int index, // to x vector
					     const T sigma2,
					     std::vector<T>& pj) const
  {
    if(index >= x.size()) return false;
    
    pj.resize(x.size());
    
    T rsum = T(0.0f); // calculates rsum for this index

#pragma omp parallel shared(rsum)
    {
      T rs = T(0.0f);

      math::vertex<T> delta;
      delta.resize(x[0].size());
      delta.zero();

#pragma omp for nowait schedule(auto)
      for(unsigned int k=0;k<x.size();k++){
	if(index == k) continue;
	delta = x[k] - x[index];
	
	T v = -(delta*delta)[0]/sigma2;
	
	rs += whiteice::math::exp(v);
      }

#pragma omp critical
      {
	rsum += rs;
      }
    }

    #pragma omp parallel
    {
      math::vertex<T> delta;
      delta.resize(x[0].size());
      delta.zero();
      
#pragma omp for nowait schedule(auto)      
      for(unsigned int j=0;j<x.size();j++){
	if(index == j) continue;
	
	delta = x[j] - x[index];
	T v = -(delta*delta)[0]/sigma2;
	
	pj[j] = whiteice::math::exp(v)/rsum;
      }
    }

    pj[index] = T(0.0f);

    return true;
  }


  // calculates distribution's perplexity
  template <typename T>
  T TSNE<T>::calculate_perplexity(const std::vector<T>& pj) const
  {
    if(pj.size() == 0) return T(1.0f);

    T H = T(0.0f); // entropy

#pragma omp parallel shared(H)
    {
      T h = T(0.0f);

#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<pj.size();i++){
	if(pj[i] > T(0.0f))
	  H += -pj[i]*math::log(pj[i])/math::log(T(2.0f));
      }

#pragma omp critical
      {
	H += h;
      }
    }

    const T perplexity = math::pow(T(2.0f),H);

    return perplexity;
  }


  // calculate x samples probability distribution p values
  // estimates sigma^2 variance parameter using distribution perplexity values
  template <typename T>
  bool TSNE<T>::calculate_pvalues(const std::vector< math::vertex<T> >& x,
				  const T perplexity,
				  std::vector< std::vector<T> >& pij) const
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

      if(math::mean_covariance_estimate(mx, Cxx, x) == false) // parallel now
	return false;

      total_var = T(0.0f);

      for(unsigned int i=0;i<Cxx.xsize();i++)
        total_var += Cxx(i,i);

      total_var /= T(x[0].size());
    }

    const T TOTAL_SIGMA2 = total_var;  // total variance in data
    const T MIN_SIGMA2   = TOTAL_SIGMA2/T(1000000000.0f);

    bool error = false;

#pragma omp parallel
    {
      std::vector<T> pj;

#pragma omp for nowait schedule(auto)
      for(unsigned int j=0;j<x.size();j++){
	if(error) continue; // stopped to error
	
	// searches for sigma2 variance value for x[j] centered distribution
	
	rji[j].resize(x.size());
	
	T sigma2_min = MIN_SIGMA2;
	T sigma2_max = TOTAL_SIGMA2; // maximum sigma value
	T perp_min, perp_max;
	
	{
	  // searches for minimum sigma2 value
	  calculate_pvalue_given_sigma(x, j, sigma2_min, pj);
	  perp_min = calculate_perplexity(pj);
	  
	  while(perp_min > perplexity && sigma2_min > T(0.0f)){
	    sigma2_min /= T(2.0f);
	    calculate_pvalue_given_sigma(x, j, sigma2_min, pj);
	    perp_min = calculate_perplexity(pj);	  
	  }
	  
	  if(perp_min > perplexity){
	    error = true;
	    continue;
	    // return false; // could not find minimum value
	  }
	  
	  // searches for maximum sigma2 value
	  calculate_pvalue_given_sigma(x, j, sigma2_max, pj);
	  perp_max = calculate_perplexity(pj);
	  
	  while(perp_max < perplexity && sigma2_max < T(10e9)){
	    sigma2_max *= T(2.0f);
	    calculate_pvalue_given_sigma(x, j, sigma2_max, pj);
	    perp_max = calculate_perplexity(pj);
	  }
	  
	  if(perp_max < perplexity){
	    error = true;
	    continue;
	    // return false; // could not find maximum value
	  }
	}

	// no we have minimum and maximum value for sigma2 variance term
	// search for sigma2 term with target perplexity
	{
	  T sigma2_next = (sigma2_min + sigma2_max)/T(2.0f);
	  
	  calculate_pvalue_given_sigma(x, j, sigma2_next, pj);
	  T perp_next = calculate_perplexity(pj);

	  const T epsilon = T(1e-12);

	  while(math::abs(perplexity - perp_next) > T(0.01f) &&
		(sigma2_max - sigma2_min) > epsilon)
	  {
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
	  
	  if(sigma2_max - sigma2_min <= epsilon){
	    error = true;
	    continue;
	    // return false; // could not find target perplexity
	  }

	  // we found sigma variance and probability distribution with target perplexity
	  rji[j] = pj;
	  rji[j][j] = T(0.0f);
	  
#if 0
	  T psum = T(0.0f);
	  for(unsigned int i=0;i<rji.size();i++){
	    if(rji[j][i] < T(0.0f))
	      std::cout << "NEGATIVE PROBABILITY!. INDEX: " << i << std::endl;
	    else if(rji[j][i] == T(0.0f))
	      std::cout << "ZERO PROBABILITY: INDEXES: " << j << " " << i << std::endl;
	    else
	      std::cout << "NON-ZERO PROBABILITY: " << rji[j][i] << std::endl;
	    psum += rji[j][i];
	  }
	  
	  std::cout << "PSUM: " << psum << std::endl;
	  std::cout << "FINAL PERPLEXITY: " << perp_next << std::endl;
	  std::cout << "FINAL SIGMA: " << sigma2_next << std::endl;
	  std::cout << std::flush;
#endif
	}
	  
      }
      
    }

    if(error) return false;

    
    // calculates symmetric p-values
    pij.resize(x.size());

#pragma omp parallel for schedule(auto)
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
				  std::vector< std::vector<T> >& qij,
				  T& qsum) const
  {
    if(y.size() == 0) return false;
    
    // calculates qij terms
    qij.resize(y.size());

    qsum = T(0.0f);

    const T epsilon = T(1e-12);

#pragma omp parallel shared(qsum)
    {
      math::vertex<T> delta;
      delta.resize(y[0].size());
      delta.zero();

      T qs = T(0.0f);

#pragma omp for nowait schedule(auto)
      for(unsigned int i=0;i<y.size();i++){
	qij[i].resize(y.size());
	for(unsigned int j=0;j<y.size();j++){
	  if(i == j) continue; // skip same values
	  delta = y[i] - y[j];
	  auto nrm2 = (delta*delta)[0];
	  auto pvalue = T(1.0f)/(T(1.0f) + nrm2);

	  if(pvalue < epsilon)
	    pvalue = epsilon;

	  qij[i][j] = pvalue;
	  
	  qs += pvalue;
	}

	qij[i][i] = T(0.0f);
      }

#pragma omp critical
      {
	qsum += qs;
      }
    }
      
#pragma omp parallel for schedule(auto)
    for(unsigned int i=0;i<y.size();i++){
      for(unsigned int j=0;j<y.size();j++){
	qij[i][j] /= qsum;
      }
    }

    return true;
  }

  
  template <typename T>
  bool TSNE<T>::kl_divergence(const std::vector< std::vector<T> >& pij,
			      const std::vector< std::vector<T> >& qij,
			      T& klvalue) const
  {
    if(pij.size() <= 0 || qij.size() <= 0) return false;
    if(qij.size() != pij.size()) return false;
    if(pij.size() != pij[0].size()) return false;
    if(qij.size() != qij[0].size()) return false;

    // calculates KL divergence
    klvalue = T(0.0f);

#pragma omp parallel shared(klvalue)
    {
      T klv = T(0.0f); // threadwise KL divergence terms

#pragma omp for nowait schedule(auto)  
      for(unsigned int i=0;i<pij.size();i++){
	for(unsigned int j=0;j<pij.size();j++){
	  if(i == j) continue; // skip same values
	  
	  if(pij[i][j] == T(0.0f)) // skip zero values (0*log(0) = 0)
	    continue;

	  // ignores zero q values (should not happen after the fix)
	  if(qij[i][j] > T(0.0f)){
	    if(kl_absolute_value)
	      klv += pij[i][j]*whiteice::math::abs(whiteice::math::log(pij[i][j]/qij[i][j]));
	    else
	      klv += pij[i][j]*whiteice::math::log(pij[i][j]/qij[i][j]);
	  }
	}
      }

#pragma omp critical
      {
	klvalue += klv;
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
			    std::vector< math::vertex<T> >& ygrad) const
  {
    if(pij.size() != y.size()) return false;
    if(pij.size() <= 0 || y.size() <= 0) return false;
    if(pij.size() != pij[0].size()) return false;

    ygrad.resize(y.size());

    T Ps = T(0.0f);
    if(kl_absolute_value){

#pragma omp parallel shared(Ps)
      {
	T psi = T(0.0f);

#pragma omp for nowait schedule(auto)	
	for(unsigned int k=0;k<y.size();k++){
	  for(unsigned int l=0;l<y.size();l++){
	    if(k == l) continue;
	    
	    T ratio = 1.0f;

	    if(qij[k][l] <= T(0.0f)) ratio = T(100.0f); // positive infinity means sign(x)=1
	    else if(pij[k][l] <= T(0.0f)) ratio = T(0.00001); // zero means sign(x)=-1
	    else ratio = pij[k][l]/qij[k][l];
	    
	    const T logratio = whiteice::math::log(ratio);

	    if(logratio >= T(0.0f))
	      psi += pij[k][l];
	    else
	      psi -= pij[k][l];
	  }
	}

#pragma omp critical
	{
	  Ps += psi;
	}

      }
    }

    
    // calculates gradient for each y value
#pragma omp parallel
    {
      // thread-wise variables
      math::vertex<T> delta;
      delta.resize(y[0].size());

#pragma omp for nowait schedule(auto)
      for(unsigned int m=0;m<ygrad.size();m++){
	ygrad[m].resize(y[0].size());
	ygrad[m].zero();

	if(kl_absolute_value == false){
	  for(unsigned int j=0;j<y.size();j++){
	    if(m == j) continue;
	    
	    delta = y[m] - y[j];
	    
	    ygrad[m] += (pij[m][j]-qij[m][j])*(T(4.0f)/(T(1.0f) + (delta*delta)[0]))*delta;
	  }
	}
	else{

	  for(unsigned int j=0;j<y.size();j++){
	    if(m == j) continue;
	    
	    T ratio = 1.0f;

	    if(qij[m][j] <= T(0.0f)) ratio = T(100.0f); // positive infinity means sign(x)=1
	    else if(pij[m][j] <= T(0.0f)) ratio = T(0.00001); // zero means sign(x)=-1
	    else ratio = pij[m][j]/qij[m][j];
	    
	    const T logratio = whiteice::math::log(ratio);
	    
	    delta = y[m] - y[j];

	    // multiply gradient by sign(log(pij/qij))
	    if(logratio >= T(0.0f))
	      ygrad[m] += (pij[m][j]-Ps*qij[m][j])*(T(4.0f)/(T(1.0f) + (delta*delta)[0]))*delta;
	    else
	      ygrad[m] += (-pij[m][j]-Ps*qij[m][j])*(T(4.0f)/(T(1.0f) + (delta*delta)[0]))*delta;
	  }
	}

#if 0
	// calculates gradient of each KL divergence term
	for(unsigned int i=0;i<y.size();i++){
	  for(unsigned int j=0;j<y.size();j++){
	    if(i == j) continue; // skip same values
	    
	    if(pij[i][j] <= T(0.0f)) // skip zero values
	      continue;
	    
	    if(qij[i][j] <= T(0.0f)) // skip zero values (should not happen after the fix
	      continue;

	    const T ratio   = pij[i][j]/qij[i][j];
	    T scaling = -ratio;

	    if(kl_absolute_value){
	      T lr = whiteice::math::log(ratio);
	      if(lr < T(0.0f)) scaling = -scaling; // multiply by sign(lr)
	    }
	    
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
#endif

      } // parallel OpenMP for-loop ends (ygrad[m])

      
    } // OpenMP block ends

    // calculated all qterms
    return true;
  }


  template class TSNE< math::blas_real<float> >;
  template class TSNE< math::blas_real<double> >;
  
};
