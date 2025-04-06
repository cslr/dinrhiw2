

#include "IPD_ES.h"
#include "nnetwork.h"


namespace whiteice
{

  template <typename T>
  IPD_ES<T>::IPD_ES(const unsigned int _TURNS,
		    const unsigned int _HISTORY_LENGTH) :
    TURNS(_TURNS), HISTORY_LENGTH(_HISTORY_LENGTH)
  {
    assert(TURNS > 0);
    assert(HISTORY_LENGTH > 0);

    std::vector<unsigned int> arch;

    // input = [history of own play (10 elements -1,+1) and history of opponent play (10 elements, -1, +1)], no-value = 0
    // output = [dimension=1: < 0 defect, >= 0 co-operate]

    arch.push_back(2*HISTORY_LENGTH);
    arch.push_back(10);
    arch.push_back(10);
    arch.push_back(1);

    whiteice::nnetwork<T> nn(arch, whiteice::nnetwork<T>::rectifier);
    nn.setNonlinearity(nn.getLayers()-1, whiteice::nnetwork<T>::tanh); // [-1,+1] output

    nn.randomize(2, T(0.5));
    nn.setResidual(false);
    nn.setBatchNorm(false);

    {
      std::lock_guard<std::mutex> lock(model_mutex);
      model = nn;
    }
  }


  template <typename T>
  IPD_ES<T>::~IPD_ES()
  {
  }

  
  template <typename T>
  bool IPD_ES<T>::estimateReward(const math::vertex<T>& x,
				 const std::vector< math::vertex<T> >& population,
				 T& reward) const // reward must be positive number >= 0
  {
    reward = T(0.0f);

    const unsigned int XSIZE = PARAMETER_DIMENSIONS();

    if(x.size() != XSIZE){
      reward = T(-1.0f);
      return false;
    }

    const unsigned int N_SEARCH_SIZE = 15; // population.size();
    
    for(unsigned int n=0;n<N_SEARCH_SIZE;n++){
      const unsigned int index = rng.rand() % population.size();
      // const unsigned int index = n;
      
      if(population[index].size() != XSIZE){
	reward = T(-1.0f);
	return false;
      }

      T r;
      
      //if(rng.uniformf() > 0.50f)
      r = game_ipd(x, population[index]);
	//else
	//r = game_ipd(population[index], x);

      if(r < T(0.0)){
	reward = T(-1.0f);
	return false;
      }
      
      reward += r;
    }

    // const unsigned int psize = population.size() < 10 ? population.size() : 10;
    const unsigned int psize = N_SEARCH_SIZE; // population.size();
    
    reward /= psize;

    return true;
  }


  template <typename T>
  bool IPD_ES<T>::estimateMeanRewardReference(const math::vertex<T>& x, // mean reward against reference good standard solution
					      T& reward) const
  {
    reward = T(0.0f);

    const unsigned int N = 50;

    
    const unsigned int XSIZE = PARAMETER_DIMENSIONS();

    if(x.size() != XSIZE){
      reward = T(-1.0f);
      return false;
    }
    
    for(unsigned int n=0;n<N;n++){
      T r = game_ipd_generous_tit_for_tat(x);

      if(r < T(0.0)){
	reward = T(-1.0f);
	return false;
      }
      
      reward += r;
    }

    reward /= N;

    return true;

    
  }

  
  template <typename T>
  T IPD_ES<T>::game_ipd(const math::vertex<T>& x, const math::vertex<T>& y) const
  {
    model_mutex.lock();

    auto xmodel = model;
    auto ymodel = model;

    model_mutex.unlock();

    if(xmodel.importdata(x) == false) return T(-1.0f);
    if(ymodel.importdata(y) == false) return T(-1.0f);

    math::vertex<T> xhistory;
    math::vertex<T> yhistory;
    math::vertex<T> xaction, yaction;
    math::vertex<T> hx, hy;

    xhistory.resize(HISTORY_LENGTH);
    yhistory.resize(HISTORY_LENGTH);
    xhistory.zero();
    yhistory.zero();

    xaction.resize(1);
    yaction.resize(1);
    hx.resize(2*HISTORY_LENGTH);
    hy.resize(2*HISTORY_LENGTH);

    T reward = T(0.0f);

    for(unsigned int t=0;t<TURNS;t++){

      // copy histories to input variable
      {
	for(unsigned int i=0;i<HISTORY_LENGTH;i++)
	  hx[i] = xhistory[i];
	
	for(unsigned int i=0;i<HISTORY_LENGTH;i++)
	  hx[HISTORY_LENGTH+i] = yhistory[i];
	
	for(unsigned int i=0;i<HISTORY_LENGTH;i++)
	  hx[i] = yhistory[i];
	
	for(unsigned int i=0;i<HISTORY_LENGTH;i++)
	  hy[HISTORY_LENGTH+i] = xhistory[i];
      }

      // calculates model responses
      {
	if(xmodel.calculate(hx, xaction) == false) return T(-1.0f);
	
	if(ymodel.calculate(hy, yaction) == false) return T(-1.0f);
      }

      
      if(xaction[0] < (2.0f*rng.uniformf() - 1.0f)) xaction[0] = -1.0f;
      else xaction[0] = +1.0f;
      
      if(yaction[0] < (2.0f*rng.uniformf() - 1.0f)) yaction[0] = -1.0f;
      else yaction[0] = +1.0f;
      

      if(xaction[0] < 0.0f) xaction[0] = -1.0f;
      else if(xaction[0] >= 0.0f) xaction[0] = +1.0f;
      if(yaction[0] < 0.0f) yaction[0] = -1.0f;
      else if(yaction[0] >= 0.0f) yaction[0] = +1.0f;

      // assigns rewards based on co-operation/defect choices (x is the player) [prison's dilemma rewards!]
      {	
	if(xaction[0] >= T(0.0f) && yaction[0] >= T(0.0f)){ // both x and y co-operate
	  reward += T(3.0f); // was: 3
	}
	else if(xaction[0] >= T(0.0f) && yaction[0] < T(0.0f)){ // x co-operates, y defects
	  reward += T(0.0f);
	}
	else if(xaction[0] < T(0.0f) && yaction[0] >= T(0.0f)){ // x defects, y co-operates
	  reward += T(5.0f);
	}
 	else if(xaction[0] < T(0.0f) && yaction[0] < T(0.0f)){ // x defects, y defects
	  reward += T(1.0f);
	}
      }

      // updates histories
      {
	for(unsigned int i=0;i<(HISTORY_LENGTH-1);i++)
	  xhistory[i] = xhistory[i+1];
	
	xhistory[HISTORY_LENGTH-1] = (xaction[0] >= T(0.0f)) ? T(1.0f) : T(-1.0f);

	for(unsigned int i=0;i<(HISTORY_LENGTH-1);i++)
	  yhistory[i] = yhistory[i+1];
	
	yhistory[HISTORY_LENGTH-1] = (yaction[0] >= T(0.0f)) ? T(1.0f) : T(-1.0f);
      }
    }
    
    return reward/TURNS;
  }


  template <typename T>
  T IPD_ES<T>::game_ipd_generous_tit_for_tat(const math::vertex<T>& x) const
  {
    model_mutex.lock();

    auto xmodel = model;

    model_mutex.unlock();

    if(xmodel.importdata(x) == false) return T(-1.0f);

    math::vertex<T> xhistory;
    math::vertex<T> yhistory;
    math::vertex<T> xaction, yaction;
    math::vertex<T> hx, hy;

    xhistory.resize(HISTORY_LENGTH);
    yhistory.resize(HISTORY_LENGTH);
    xhistory.zero();
    yhistory.zero();

    xaction.resize(1);
    yaction.resize(1);
    hx.resize(2*HISTORY_LENGTH);
    hy.resize(2*HISTORY_LENGTH);

    T reward = T(0.0f);

    for(unsigned int t=0;t<TURNS;t++){

      // copy histories to input variable
      {
	for(unsigned int i=0;i<HISTORY_LENGTH;i++)
	  hx[i] = xhistory[i];
	
	for(unsigned int i=0;i<HISTORY_LENGTH;i++)
	  hx[HISTORY_LENGTH+i] = yhistory[i];
	
	for(unsigned int i=0;i<HISTORY_LENGTH;i++)
	  hy[i] = yhistory[i];
	
	for(unsigned int i=0;i<HISTORY_LENGTH;i++)
	  hy[HISTORY_LENGTH+i] = xhistory[i];
      }

      // calculates model responses
      {
	if(xmodel.calculate(hx, xaction) == false) return T(-1.0f);

	// calculates generous tit for tat response
	{
	  if(t == 0){
	    yaction[0] = T(1.0f); // co-operate first round
	  }
	  else{
	    if(xhistory[HISTORY_LENGTH-1] < 0.0f){
	      yaction[0] = T(-1.0f); // defect last round so we defect too

	      if(rng.uniformf() < 0.10f) yaction[0] = T(+1.0f); // generous co-operation 10% of the time
	      
	    }
	    else if(xhistory[HISTORY_LENGTH-1] >= 0.0f){ // co-operate if last round opponent co-operated
	      yaction[0] = T(+1.0f);
	    }
	  }
	}
      }

      
      if(xaction[0] < (2.0f*rng.uniformf() - 1.0f)) xaction[0] = -1.0f;
      else xaction[0] = +1.0f;
      
      
      if(xaction[0] < 0.0f) xaction[0] = -1.0f;
      else if(xaction[0] >= 0.0f) xaction[0] = +1.0f;
      if(yaction[0] < 0.0f) yaction[0] = -1.0f;
      else if(yaction[0] >= 0.0f) yaction[0] = +1.0f;
	
      // assigns rewards based on co-operation/defect choices (x is the player) [prison's dilemma rewards!]
      {
	if(xaction[0] >= T(0.0f) && yaction[0] >= T(0.0f)){ // both x and y co-operate
	  reward += T(3.0f); // was: 3
	}
	else if(xaction[0] >= T(0.0f) && yaction[0] < T(0.0f)){ // x co-operates, y defects
	  reward += T(0.0f);	
	}
	else if(xaction[0] < T(0.0f) && yaction[0] >= T(0.0f)){ // x defects, y co-operates
	  reward += T(5.0f);
	}
	else if(xaction[0] < T(0.0f) && yaction[0] < T(0.0f)){ // x defects, y defects
	  reward += T(1.0f);
	}
      }

      // updates histories
      {
	for(unsigned int i=0;i<(HISTORY_LENGTH-1);i++)
	  xhistory[i] = xhistory[i+1];
	
	xhistory[HISTORY_LENGTH-1] = (xaction[0] >= T(0.0f)) ? T(1.0f) : T(-1.0f);

	for(unsigned int i=0;i<(HISTORY_LENGTH-1);i++)
	  yhistory[i] = yhistory[i+1];
	
	yhistory[HISTORY_LENGTH-1] = (yaction[0] >= T(0.0f)) ? T(1.0f) : T(-1.0f);
      }
    }
    
    return reward/TURNS;
  }

  

  template <typename T>
  unsigned int IPD_ES<T>::PARAMETER_DIMENSIONS() const // number of parameters in the model
  {
    std::lock_guard<std::mutex> lock(model_mutex);

    return model.exportdatasize();
  }
  

  template class IPD_ES< math::blas_real<float> >;
  template class IPD_ES< math::blas_real<double> >;
};
