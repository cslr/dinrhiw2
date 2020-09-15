/*
 * format of simplex table (initial)
 *
 *           x_1        x_2     slacks  = value/solution
 * -----------------------------------------------------
 *   F  ||    2     |    5    |  ...    | 0
 *  s_1 ||   -1     |   -2    |  ...    | 10
 *  s_2 ||   -3     |   -2    |  ...    | 24
 *  s_3 ||   -2     |   -10   |  ...    | 40
 *
 *  corresponding problem
 *
 *  maximize F(x)
 *
 *  when F(x_1, x_2) = 2*x_1 + 5*x_2
 *  and constraints (and equalities with slacks are)
 *  1*x_1 + 2*x_2 < 10
 *  3*x_1 + 2*x_2 < 24
 *  2*x_1 + 10*x_2 < 40
 *  x_1 >= 0, x_2 >= 0
 *
 *  1*x_1 + 2*x_2  + s_1 = 10
 *  3*x_1 + 2*x_2  + s_2 = 24
 *  2*x_1 + 10*x_2 + s_3 = 40
 *  x_1 >= 0, x_2 >= 0, s_1 >= 0, s_2 >= 0, s_3 >= 0
 *
 */


#include <iostream>
#include <vector>
#include <exception>
#include <stdexcept>
#include <typeinfo>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "simplex.h"
#include "blade_math.h"
#include "dinrhiw_blas.h"
#include "Log.h"
#include "vertex.h"

#ifdef WINOS
#include <windows.h>
#endif


extern "C" { static void* __simplex_thread_init(void *simplex_ptr); };


namespace whiteice
{
  namespace math
  {
    
    
    template <typename T>
    simplex<T>::simplex(unsigned int _variables,
			unsigned int _constraints)
    {
      if(typeid(T) != typeid(float) && typeid(T) != typeid(double))
	throw std::invalid_argument("template parameter isn't one of {float, double}.");
      
      if(_variables == 0 || _constraints == 0)
	throw std::invalid_argument("Zero variables or constraints");
      
      pthread_mutex_init(&simplex_lock, 0);
      // maximization_thread = 0;
      
      running    = false;
      has_result = false;
      
      numVariables   = _variables;
      constraints.resize(_constraints);
      numArtificials = 0;

      target = NULL;
      
      for(auto& c : constraints)
	c = NULL;

#ifdef CUBLAS
      // allocates memory from NVIDIA DEVICE
      
      auto e = cudaMallocManaged(&target, (_variables+1)*sizeof(T));

      if(e != cudaSuccess || target == NULL){
	whiteice::logging.error("simplex ctor: cudaMallocManaged() failed.");
	throw CUDAException("CUBLAS memory allocation failure.");
      }

      e = cudaMemset(target, 0, (_variables+1)*sizeof(T));

      if(e != cudaSuccess){
	whiteice::logging.error("simplex ctor: cudaMemset() failed.");
	throw CUDAException("CUBLAS memory init failure.");
      }

      for(unsigned int i=0;i<constraints.size();i++){

	e = cudaMallocManaged(&(constraints[i]), (_variables+1)*sizeof(T));

	auto f = cudaMemset(constraints[i], 0, (_variables+1)*sizeof(T));

	if(e != cudaSuccess  || f != cudaSuccess || constraints[i] == NULL){
	  whiteice::logging.error("simplex ctor: cudaMallocManaged()/cudaMemset() failed.");

	  cudaFree(target); target = NULL;
	  for(unsigned int j=0;j<i;j++){
	    cudaFree(constraints[j]);
	    constraints[j] = NULL;
	  }
	  
	  throw CUDAException("CUBLAS memory allocation/init failure.");
	}
      }
      
#else
      
      // allocates memory for simplex table
      target = (T*)calloc((_variables+1),sizeof(T));
      
      for(unsigned int i=0;i<constraints.size();i++)
	constraints[i] = (T*)calloc((_variables+1),sizeof(T));

#endif

      ctypes.resize(constraints.size());
    }
    
    
    template <typename T>
    simplex<T>::~simplex()
    {
      while(running){
	pthread_cancel( maximization_thread );
	sleep(1); // waits for thread to stop
      }
      
      running = false;
      has_result = false;
      
      pthread_mutex_destroy( &simplex_lock );

#ifdef CUBLAS

      if(target)
	cudaFree(target);
      
      for(unsigned int i=0;i<constraints.size();i++)
	if(constraints[i])
	  cudaFree(constraints[i]);
      
#else
      
      if(target) 
	free(target);
      
      for(unsigned int i=0;i<constraints.size();i++)
	if(constraints[i])
	  free(constraints[i]);
#endif
    }
    
    
    // format: [c_1 c_2 ... c_n], where F = SUM( c_i*x_i )
    template <typename T>
    bool simplex<T>::setTarget(const std::vector<T>& trg) 
    {
      if(running)
	return false;
      
      if(trg.size() != numVariables)
	return false;
      
      for(unsigned int i=0;i<numVariables;i++)
	target[i] = -trg[i];
      
      target[numVariables] = T(0.0); // solution
      
      return true;
    }
    
    
    template <typename T>
    bool simplex<T>::setConstraint(unsigned int index,
				   const std::vector<T>& constraint,
				   unsigned int eqtype) 
    {
      if(running)
	return false;
      
      if(index >= constraints.size() ||
	 constraint.size() != numVariables + 1)
	return false;
      
      if(eqtype >= 3)
	return false;
      
      if(constraint[numVariables] < T(0.0))
	return false; // right hand side must be positive
      
      // sets coefficients
      for(unsigned int i=0;i<(numVariables+1);i++)
	constraints[index][i] = constraint[i];
      
      ctypes[index] = eqtype;
      
      return true;
    }
    
    
    template <typename T>
    unsigned int simplex<T>::getNumberOfConstraints() const {
      return constraints.size();
    }  
    
    
    template <typename T>
    unsigned int simplex<T>::getNumberOfVariables() const {
      return numVariables;
    }
    
    
    // format - look at set setX()s documentation
    template <typename T>
    bool simplex<T>::getTarget(std::vector<T>& trg) const 
    {
      if(running)
	return false;
      
      trg.resize(numVariables);
      
      for(unsigned int i=0;i<trg.size();i++)
	trg[i] = -target[i];
      
      // right hand side
      trg.push_back(target[numVariables]);
      
      return true;
    }
    
    
    template <typename T>
    bool simplex<T>::getConstraint(unsigned int index,
				   std::vector<T>& constraint) const 
    {
      if(running)
	return false;
      
      if(index >= constraints.size())
	return false;
      
      constraint.resize(numVariables);
      
      for(unsigned int i=0;i<constraint.size();i++)
	constraint[i] = constraints[index][i];
      
      // right hand side
      constraint.push_back(constraints[index][numVariables]);
      
      return true;
    }
    
    
    template <typename T>
    bool simplex<T>::maximize() 
    {
      if(running) // check without grabbing lock
	return false;
      
      // starting new maximization thread
      // might be possible
      pthread_mutex_lock( &simplex_lock );
      
      // real check
      if(running){
	pthread_mutex_unlock( &simplex_lock );
	return false;
      }
      
      running = false;
      iter = 0;
      
      if(pthread_create(&maximization_thread, 0,
			__simplex_thread_init,
			(void*)this) != 0){
	
	pthread_mutex_unlock( &simplex_lock );
	return false;
      }
	
#ifndef WINOS
      unsigned int counter = 0;
      struct timespec ts;
      
      // waits 1 sec for thread to start
      while(counter < 100 && running == false){
	ts.tv_sec  = 0;
	ts.tv_nsec = 100000000; // 10 ms
	nanosleep(&ts, 0);
	counter++;
      }
#else
      Sleep(1000); // sleep for windows/mingw
#endif
      
      if(running == false){
	pthread_cancel( maximization_thread );
	running = false;
	pthread_mutex_unlock( &simplex_lock );
	return false;
      }
	
      pthread_mutex_unlock( &simplex_lock );
      
      return true; // running == true
    }
    
    
    template <typename T>
    void simplex<T>::threadloop()
    {
      if(typeid(T) == typeid(float))
      {
	running = true;
	has_result = false;
	
	// sync with maximize()
	pthread_mutex_lock( &simplex_lock );
	pthread_mutex_unlock( &simplex_lock );
      
	
	// index of variable to enter (eindex) and leave (lindex)
	unsigned int eindex, lindex; 
	
	
	// adds slacks and finds feasible solution
	{
	  // adds artificial variables that are
	  // needed to solve problem
	  
	  unsigned int numArtificials  = 0;
	  unsigned int numPseudoSlacks = 0;
	  unsigned int numRealArtificials = 0;
	  T* pseudotarget = 0;
	  
	  // 'real' slacks and surplusses have
	  // variable number numVariables+i
	  // pseudoslacks have variable numbers:
	  // numVariables+numArtificials-numPseudoSlacks+i
	  
	  // 'normal variables first'
	  for(unsigned int i=0;i<constraints.size();i++){
	    if(ctypes[i] == 0){ // adds slack
	      numArtificials++;
	    }
	    else if(ctypes[i] == 2){ // adds surplus
	      numArtificials += 2; // surplus AND pseudoslack
	      numPseudoSlacks++;
	    }
	    else{ // adds pseudoslack only
	      numArtificials++;
	      numPseudoSlacks++;
	    }
	  }
	  
	  numRealArtificials = numArtificials - numPseudoSlacks;
	  
	  // transforms problem to problem
	  // with all needed slacks and surplusses
	  
	  // does all memory allocations first
	  // (target isn't used in removal of pseudoSlacks)
	  T* temp = NULL;

#ifdef CUBLAS

	  auto e = cudaMallocManaged(&temp,
				     sizeof(T)*(numVariables + numRealArtificials + 1));

	  if(e != cudaSuccess){
	    running = false;
	    has_result = false;

	    whiteice::logging.error("simplex<>::threadloop(): cudaMallocManaged() failed.");
	    return; // silent failure in a thread loop (no cuda exception)
	  }

	  if(target) cudaFree(target);
	  target = temp;

	  e = cudaMallocManaged(&temp,
				sizeof(T)*(numVariables + numRealArtificials + 1));

	  if(e != cudaSuccess){
	    running = false;
	    has_result = false;
	    
	    whiteice::logging.error("simplex<>::threadloop(): cudaMallocManaged() failed.");
	    return; // silent failure in a thread loop (no cuda exception)
	  }

	  pseudotarget = temp;

	  std::vector<T*> new_constraints = constraints;

	  for(unsigned int i=0;i<new_constraints.size();i++){
	    e = cudaMallocManaged(&(new_constraints[i]),
				  sizeof(T)*(numVariables + numArtificials + 1));

	    if(e != cudaSuccess){
	      cudaFree(pseudotarget);

	      for(unsigned int j=0;j<i;j++)
		cudaFree(new_constraints[j]);
	    }

	    running = false;
	    has_result = false;
	    
	    whiteice::logging.error("simplex<>::threadloop(): cudaMallocManaged() failed.");
	    return; // silent failure
	  }

	  for(unsigned int i=0;i<constraints.size();i++){
	    cudaFree(constraints[i]);
	  }

	  constraints = new_constraints;

	  // sets memory areas to zero (needed?)
	  {
	    std::vector<bool> ok;
	      
	    e = cudaMemset(target, 0, sizeof(T)*(numVariables + numArtificials + 1));
	    if(e != cudaSuccess) ok.push_back(false);

	    e = cudaMemset(pseudotarget, 0, sizeof(T)*(numVariables + numArtificials + 1));
	    if(e != cudaSuccess) ok.push_back(false);

	    for(unsigned int i=0;i<constraints.size();i++){
	      e = cudaMemset(constraints[i], 0,
			     sizeof(T)*(numVariables + numArtificials + 1));
	      if(e != cudaSuccess) ok.push_back(false);
	    }

	    if(ok.size() > 0){ // failures in memsets
	      cudaFree(pseudotarget);

	      running = false;
	      has_result = false;
	      
	      whiteice::logging.error("simplex<>::threadloop(): cudaMemset() failed.");
	      return; // silent failure
	    }
	  }
	  
#else
	  
	  if((temp = (T*)realloc(target,
				 sizeof(T)*(numVariables + numRealArtificials + 1))) == 0){
	    running = false;
	    has_result = false;
	    return;
	  }
	  
	  
	  target = temp;
	  
	  // memory for pseudotarget (used to get feasible solution)
	  temp = (T*)calloc(numVariables+numArtificials+1,sizeof(T));
	  if(temp == 0){
	    temp = (T*)realloc(target, sizeof(T)*(numVariables + 1));
	    if(temp)
	      target = temp;
	    
	    running = false;
	    has_result = false;
	    
	    return;
	  }
	  
	  pseudotarget = temp;
	  
	  for(unsigned int i=0;i<constraints.size();i++){
	    temp = (T*)realloc(constraints[i], sizeof(T)*(numVariables + numArtificials + 1));
	    
	    if(temp == 0){ // failure, back to original state
	      for(unsigned int j=0;j<i;j++){
		temp = (T*)realloc(constraints[j], sizeof(T)*(numVariables + 1));
		if(temp)
		  constraints[j] = temp;
	      }
	      
	      temp = (T*)realloc(target, sizeof(T)*(numVariables + 1));
	      if(temp)
		target = temp;
	      
	      free(pseudotarget);
	      
	      running = false;
	      has_result = false;
	      
	      return;
	    }
	    
	    constraints[i] = temp;
	  }

#endif
	  
	  
	  // memory allocations were ok (in theory)
	  // actually setups simplex table correctly
	  
	  T RHS = target[numVariables];
	  memset(&(target[numVariables]), 0, numRealArtificials*sizeof(T)); // TODO USE CUDA
	  target[numVariables+numRealArtificials] = RHS;
	  
	  unsigned int countReals   = 0;
	  unsigned int countPseudos = 0;
	  
	  // setups also basic (slacks) and non basic variables
	  basic.clear();
	  nonbasic.clear();
	  
	  for(unsigned int i=0;i<numVariables;i++)
	    nonbasic.push_back(i);
	  
	  for(unsigned int i=0;i<constraints.size();i++){
	    T RHS = constraints[i][numVariables];
	    memset(&(constraints[i][numVariables]), 0, numArtificials*sizeof(T)); // TODO USE CUDA
	    constraints[i][numVariables+numArtificials] = RHS;
	    
	    // setups constraint coefficients
	    if(ctypes[i] == 0){ // slack
	      basic.push_back(numVariables+countReals); // slack variable number
	      constraints[i][numVariables+countReals] = T(1.0);
	      countReals++;	    
	    }
	    else if(ctypes[i] == 2){ // surplus + pseudoslack
	      basic.push_back(numVariables+numRealArtificials+countPseudos);
	      nonbasic.push_back(numVariables+countReals);
	      
	      constraints[i][numVariables+countReals]   = T(-1.0); // surplus
	      constraints[i][numVariables+numRealArtificials+countPseudos] = T(+1.0); // pseudoslack
	      countReals++;
	      countPseudos++;
	    }
	    else{ // pseudoslack only
	      basic.push_back(numVariables+numRealArtificials+countPseudos);
	      
	      constraints[i][numVariables+numRealArtificials+countPseudos] = T(+1.0); // pseudoslack
	      countPseudos++;
	    }
	  }
	  
	  // creates target for minimizing sum of pseudo variables
	  
	  for(unsigned int i=0;i<countPseudos;i++)
	    pseudotarget[numVariables+numRealArtificials+i] = T(-1.0);
	  
	  pseudotarget[numVariables+numArtificials] = T(0.0); // sets RHS	
	  
	  this->numArtificials = numArtificials;
	  
	  
	  // removes slacks from initial pseudotarget function
	  for(unsigned int i=0;i<basic.size();i++){
	    // initial basic variables are slacks
	    
	    if(pseudotarget[basic[i]] != T(0.0)){
	      for(unsigned int j=0;j<numVariables+numArtificials+1;j++)
		pseudotarget[j] += constraints[i][j];
	    }
	  }
	  
	  
	  // transforms minimization to maximization task
	  // min f(x) = max -f(x)
	  for(unsigned int j=0;j<numVariables+numArtificials+1;j++)
	    pseudotarget[j] = -pseudotarget[j];
	  
	  
	  // minimizes pseudotarget via normal method
	  {
	    while(find_indexes(pseudotarget, constraints,
			       eindex, lindex)){
	      
	      // calculates new lindex row
	      
	      T c = T(1.0)/constraints[lindex][eindex];

#ifdef CUBLAS
	      auto e = cublasSscal(cublas_handle, numVariables+numArtificials+1,
				   (const float*)&c,
				   (float*)constraints[lindex], 1);

	      if(e != CUBLAS_STATUS_SUCCESS){
		whiteice::logging.error("simplex<>::threadloop(): cublasSscal() failed.");
		cudaFree(pseudotarget);
		running = false;
		has_result = false;
		return;
	      }
#else
	      cblas_sscal(numVariables+numArtificials+1, *((float*)&c), 
			  (float*)constraints[lindex], 1);
#endif
	      
	      // updates pseudotarget
	      
	      c = -pseudotarget[eindex];

#ifdef CUBLAS
	      e = cublasSaxpy(cublas_handle, numVariables+numArtificials+1,
			      (const float*)&c,
			      (const float*)constraints[lindex], 1,
			      (float*)pseudotarget, 1);

	      if(e != CUBLAS_STATUS_SUCCESS){
		whiteice::logging.error("simplex<>::threadloop(): cublasSaxpy() failed.");
		cudaFree(pseudotarget);
		running = false;
		has_result = false;
		return;
	      }
#else
	      
	      cblas_saxpy(numVariables+numArtificials+1, *((float*)&c),
			  (float*)constraints[lindex], 1, (float*)pseudotarget, 1);
#endif
	      
	      // updates other rows
	      
	      if(lindex != 0){
		for(unsigned int i=0;i<lindex;i++){
		  c = -constraints[i][eindex];

#ifdef CUBLAS
		  e = cublasSaxpy(cublas_handle, numVariables+numArtificials+1,
				  (const float*)&c,
				  (const float*)constraints[lindex], 1,
				  (float*)constraints[i], 1);

		  if(e != CUBLAS_STATUS_SUCCESS){
		    whiteice::logging.error("simplex<>::threadloop(): cublasSaxpy() failed.");
		    cudaFree(pseudotarget);
		    running = false;
		    has_result = false;
		    return;
		  }
#else
		  cblas_saxpy(numVariables+numArtificials+1, *((float*)&c),
			      (float*)constraints[lindex], 1, (float*)constraints[i], 1);
#endif
		}
	      }
	      
	      
	      if(lindex != constraints.size()-1){
		for(unsigned int i=lindex+1;i<constraints.size();i++){
		  c = -constraints[i][eindex];

#ifdef CUBLAS
		  e = cublasSaxpy(cublas_handle, numVariables+numArtificials+1,
				  (const float*)&c,
				  (const float*)constraints[lindex], 1,
				  (float*)constraints[i], 1);

		  if(e != CUBLAS_STATUS_SUCCESS){
		    whiteice::logging.error("simplex<>::threadloop(): cublasSaxpy() failed.");
		    cudaFree(pseudotarget);
		    running = false;
		    has_result = false;
		    return;
		  }		  
#else
		  
		  cblas_saxpy(numVariables+numArtificials+1, *((float*)&c),
			      (float*)constraints[lindex], 1, (float*)constraints[i], 1);
#endif
		}
	      }
	      
	      
	      // updates basic/non-basic sets
	      
	      {
		unsigned int varleave = basic[lindex];
		basic[lindex] = eindex;
		
		for(unsigned int i=0;i<nonbasic.size();i++){
		  if(nonbasic[i] == eindex){
		    nonbasic[i] = varleave;
		    break;
		  }
		}
	      }
	      
	      
	      this->iter++;
	    }
	  }
	  
	  
	  if(abs(pseudotarget[numVariables+numArtificials]) >= T(1e-5)){
	    running = false;
	    has_result = false;
	    
	    return;
	  }
	  
	  
	  // checks if pseudoslack variable is still
	  // in basic solution set (must be zero)
	  {
	    bool hasPseudoSlackVar = false;
	  
	    for(unsigned int i=0;i<basic.size();i++){
	      if(basic[i] >= numVariables + numRealArtificials){
		hasPseudoSlackVar = true;
		break;
	      }
	    }
	  
	    if(hasPseudoSlackVar){
	      // find_indexes2() is same as find_indexes()
	      // except that pseudoslacks are given preference
	      // when considering leaving variables
	      
	      while(find_indexes2(pseudotarget, constraints,
				  eindex, lindex, basic,
				  numVariables + numRealArtificials)){
		
		// calculates new lindex row
	      
		T c = T(1.0)/constraints[lindex][eindex];

#ifdef CUBLAS
		auto e = cublasSscal(cublas_handle, numVariables+numArtificials+1,
				     (const float*)&c,
				     (float*)constraints[lindex], 1);
		
		if(e != CUBLAS_STATUS_SUCCESS){
		  whiteice::logging.error("simplex<>::threadloop(): cublasSscal() failed.");
		  cudaFree(pseudotarget);
		  running = false;
		  has_result = false;
		  return;
		}
		
#else
		cblas_sscal(numVariables+numArtificials+1, *((float*)&c), 
			    (float*)constraints[lindex], 1);
#endif
		
		// updates pseudotarget
		
		c = -pseudotarget[eindex];

#ifdef CUBLAS
		
		e = cublasSaxpy(cublas_handle, numVariables+numArtificials+1,
				(const float*)&c,
				(const float*)constraints[lindex], 1,
				(float*)pseudotarget, 1);
		
		if(e != CUBLAS_STATUS_SUCCESS){
		  whiteice::logging.error("simplex<>::threadloop(): cublasSaxpy() failed.");
		  cudaFree(pseudotarget);
		  running = false;
		  has_result = false;
		  return;
		}
		
#else
		cblas_saxpy(numVariables+numArtificials+1, *((float*)&c),
			    (float*)constraints[lindex], 1, (float*)pseudotarget, 1);
#endif
		
		// updates other rows
		
		if(lindex != 0){
		  for(unsigned int i=0;i<lindex;i++){
		    c = -constraints[i][eindex];

#ifdef CUBLAS
		    e = cublasSaxpy(cublas_handle, numVariables+numArtificials+1,
				    (const float*)&c,
				    (const float*)constraints[lindex], 1,
				    (float*)constraints[i], 1);
		    
		    if(e != CUBLAS_STATUS_SUCCESS){
		      whiteice::logging.error("simplex<>::threadloop(): cublasSaxpy() failed.");
		      cudaFree(pseudotarget);
		      running = false;
		      has_result = false;
		      return;
		    }
		    
#else
		    cblas_saxpy(numVariables+numArtificials+1, *((float*)&c),
				(float*)constraints[lindex], 1, (float*)constraints[i], 1);
#endif
		  }
		}
		
		
		if(lindex != constraints.size()-1){
		  for(unsigned int i=lindex+1;i<constraints.size();i++){
		    c = -constraints[i][eindex];

#ifdef CUBLAS
		    e = cublasSaxpy(cublas_handle, numVariables+numArtificials+1,
				    (const float*)&c,
				    (const float*)constraints[lindex], 1,
				    (float*)constraints[i], 1);
		    
		    if(e != CUBLAS_STATUS_SUCCESS){
		      whiteice::logging.error("simplex<>::threadloop(): cublasSaxpy() failed.");
		      cudaFree(pseudotarget);
		      running = false;
		      has_result = false;
		      return;
		    }
		    
#else
		    cblas_saxpy(numVariables+numArtificials+1, *((float*)&c),
				(float*)constraints[lindex], 1, (float*)constraints[i], 1);
#endif
		  }
		}
		
		
		// updates basic/non-basic sets
		
		{
		  unsigned int varleave = basic[lindex];
		  basic[lindex] = eindex;
		  
		  for(unsigned int i=0;i<nonbasic.size();i++){
		    if(nonbasic[i] == eindex){
		      nonbasic[i] = varleave;
		      break;
		    }
		  }
		}
		
		
		this->iter++;
	      }
	      
	      
	      /////////////////////////////////////////////////
	      // checks if solution is feasible
	      if(abs(pseudotarget[numVariables+numArtificials]) >= T(1e-5)){
		running = false;
		has_result = false;
		return;
	      }
	      
	      // checks if solution still have pseudoslacks
	      
	      for(unsigned int i=0;i<basic.size();i++){
		if(basic[i] >= numVariables + numRealArtificials){
		  running = false;
		  has_result = false;		
		  return;
		}
	      }
	      
	    }
	  }

#ifdef CUBLAS
	  cudaFree(pseudotarget);
#else
	  free(pseudotarget);
#endif
	  
	  // pseudoslacks are now zero so it is
	  // safe to remove them
	  
	  // resizes constraints

#ifdef CUBLAS

	  for(unsigned int i=0;i<constraints.size();i++){
	    T RHS = constraints[i][numVariables + numArtificials];

	    auto e = cudaMallocManaged(&temp, sizeof(T)*(numVariables + numRealArtificials + 1));

	    if(e != cudaSuccess){
	      whiteice::logging.error("simplex<>::threadloop(): cudaMallocManaged() failed.");
	      
	      running = false; // silent failure
	      has_result = false;
	      return;
	    }

	    const int original_size = numVariables + numArtificials + 1;
	    const int new_size = numVariables + numRealArtificials + 1;

	    if(new_size <= original_size){
	      e = cudaMemcpy(temp, constraints[i], new_size*sizeof(T),
			     cudaMemcpyDeviceToDevice);
	    }
	    else{
	      e = cudaMemcpy(temp, constraints[i], original_size*sizeof(T),
			     cudaMemcpyDeviceToDevice);
	    }

	    if(e != cudaSuccess){
	      cudaFree(temp);
	      whiteice::logging.error("simplex<>::threadloop(): cudaMemcpy() failed.");
	      
	      running = false; // silent failure
	      has_result = false;
	      return;
	    }

	    cudaFree(constraints[i]);
	    constraints[i] = temp;
	    constraints[i][numVariables+numRealArtificials] = RHS;
	  }
	  
	  this->numArtificials = numRealArtificials;
	  
	  std::vector<unsigned int>::iterator i;
	  i = nonbasic.begin();
	  
	  while(i != nonbasic.end()){
	    if(*i >= numVariables + numRealArtificials)
	      i = nonbasic.erase(i);
	    else
	      i++;
	  }
	  
#else
	  for(unsigned int i=0;i<constraints.size();i++){
	    T RHS = constraints[i][numVariables + numArtificials];
	    temp = (T*)realloc(constraints[i], sizeof(T)*(numVariables + numRealArtificials + 1));
	    
	    if(temp == 0){ // hard to return back to original from here
	      running = false;
	      has_result = false;		
	      return;
	    }
	    
	    constraints[i] = temp;
	    constraints[i][numVariables+numRealArtificials] = RHS;
	  }
	  
	  this->numArtificials = numRealArtificials;
	  
	  std::vector<unsigned int>::iterator i;
	  i = nonbasic.begin();
	  
	  while(i != nonbasic.end()){
	    if(*i >= numVariables + numRealArtificials)
	      i = nonbasic.erase(i);
	    else
	      i++;
	  }
#endif
	}

	
	// simplex table fixup
	// sets target function coefficients of
	// basic variables to zero
	// (needed because target function changed:
	//  switch to real target function)
	
	for(unsigned int i=0;i<basic.size();i++){
	  T c = target[basic[i]];
	  
	  for(unsigned int j=0;j<numVariables+numArtificials+1;j++)
	    target[j] -= c*constraints[i][j];
	}
	
	
	
	while(find_indexes(target, constraints,
			   eindex, lindex)){
	    
	  // calculates new lindex row
	  
	  T c = T(1.0)/constraints[lindex][eindex];

#ifdef CUBLAS
	  
	  auto e = cublasSscal(cublas_handle, numVariables+numArtificials+1,
			       (const float*)&c,
			       (float*)constraints[lindex], 1);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("simplex<>::threadloop(): cublasSscal() failed.");
	    running = false;
	    has_result = false;
	    return;
	  }
	  
#else
	  cblas_sscal(numVariables+numArtificials+1, *((float*)&c), 
		      (float*)constraints[lindex], 1);
#endif
	  
	  // updates target
	  
	  c = -target[eindex];

#ifdef CUBLAS

	  e = cublasSaxpy(cublas_handle, numVariables+numArtificials+1,
			  (const float*)&c,
			  (const float*)constraints[lindex], 1,
			  (float*)target, 1);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("simplex<>::threadloop(): cublasSaxpy() failed.");
	    running = false;
	    has_result = false;
	    return;
	  }
	  
#else
	  cblas_saxpy(numVariables+numArtificials+1, *((float*)&c),
		      (float*)constraints[lindex], 1, (float*)target, 1);
#endif
	  
	  // updates other rows
	  
	  if(lindex != 0){
	    for(unsigned int i=0;i<lindex;i++){
	      c = -constraints[i][eindex];

#ifdef CUBLAS
	      e = cublasSaxpy(cublas_handle, numVariables+numArtificials+1,
			      (const float*)&c,
			      (const float*)constraints[lindex], 1,
			      (float*)constraints[i], 1);
	      
	      if(e != CUBLAS_STATUS_SUCCESS){
		whiteice::logging.error("simplex<>::threadloop(): cublasSaxpy() failed.");
		running = false;
		has_result = false;
		return;
	      }
#else
	      cblas_saxpy(numVariables+numArtificials+1, *((float*)&c),
			  (float*)constraints[lindex], 1, (float*)constraints[i], 1);
#endif
	    }
	  }
	  
	  
	  if(lindex != constraints.size()-1){
	    for(unsigned int i=lindex+1;i<constraints.size();i++){
	      c = -constraints[i][eindex];

#ifdef CUBLAS
	      e = cublasSaxpy(cublas_handle, numVariables+numArtificials+1,
			      (const float*)&c,
			      (const float*)constraints[lindex], 1,
			      (float*)constraints[i], 1);
	      
	      if(e != CUBLAS_STATUS_SUCCESS){
		whiteice::logging.error("simplex<>::threadloop(): cublasSaxpy() failed.");
		running = false;
		has_result = false;
		return;
	      }		  
	      
#else
	      cblas_saxpy(numVariables+numArtificials+1, *((float*)&c),
			  (float*)constraints[lindex], 1, (float*)constraints[i], 1);
#endif
	    }
	  }
	  
	  
	  // updates basic/non-basic sets
	  {
	    unsigned int varleave = basic[lindex];
	    basic[lindex] = eindex;
	  
	    for(unsigned int i=0;i<nonbasic.size();i++){
	      if(nonbasic[i] == eindex){
		nonbasic[i] = varleave;
		break;
	      }
	    }
	  }
	  
	  
	  this->iter++;
	}
	
	
	has_result = true;
	// maximization_thread = 0;
	running = false;
	
	return;
      }
      else if(typeid(T) == typeid(double))
      {
	
	running = true;
	has_result = false;
	
	// sync with maximize()
	pthread_mutex_lock( &simplex_lock );
	pthread_mutex_unlock( &simplex_lock );
	
	
	// index of variable to enter (eindex) and leave (lindex)
	unsigned int eindex, lindex; 
	
	
	// adds slacks and finds feasible solution
	{
	  // adds artificial variables that are
	  // needed to solve problem
	  
	  unsigned int numArtificials  = 0;
	  unsigned int numPseudoSlacks = 0;
	  unsigned int numRealArtificials = 0;
	  T* pseudotarget = 0;
	  
	  // 'real' slacks and surplusses have
	  // variable number numVariables+i
	  // pseudoslacks have variable numbers:
	  // numVariables+numArtificials-numPseudoSlacks+i
	  
	  // 'normal variables first'
	  for(unsigned int i=0;i<constraints.size();i++){
	    if(ctypes[i] == 0){ // adds slack
	      numArtificials++;
	    }
	    else if(ctypes[i] == 2){ // adds surplus
	      numArtificials += 2; // surplus AND pseudoslack
	      numPseudoSlacks++;
	    }
	    else{ // adds pseudoslack only
	      numArtificials++;
	      numPseudoSlacks++;
	    }
	  }
	  
	  numRealArtificials = numArtificials - numPseudoSlacks;
	  
	  // transforms problem to problem
	  // with all needed slacks and surplusses
	  
	  // does all memory allocations first
	  // (target isn't used in removal of pseudoSlacks)
	  T* temp = NULL;

#ifdef CUBLAS

	  auto e = cudaMallocManaged(&temp,
				     sizeof(T)*(numVariables + numRealArtificials + 1));

	  if(e != cudaSuccess){
	    running = false;
	    has_result = false;

	    whiteice::logging.error("simplex<>::threadloop(): cudaMallocManaged() failed.");
	    return; // silent failure in a thread loop (no cuda exception)
	  }

	  if(target) cudaFree(target);
	  target = temp;

	  e = cudaMallocManaged(&temp,
				sizeof(T)*(numVariables + numRealArtificials + 1));

	  if(e != cudaSuccess){
	    running = false;
	    has_result = false;
	    
	    whiteice::logging.error("simplex<>::threadloop(): cudaMallocManaged() failed.");
	    return; // silent failure in a thread loop (no cuda exception)
	  }

	  pseudotarget = temp;

	  std::vector<T*> new_constraints = constraints;

	  for(unsigned int i=0;i<new_constraints.size();i++){
	    e = cudaMallocManaged(&(new_constraints[i]),
				  sizeof(T)*(numVariables + numArtificials + 1));

	    if(e != cudaSuccess){
	      cudaFree(pseudotarget);

	      for(unsigned int j=0;j<i;j++)
		cudaFree(new_constraints[j]);
	    }

	    running = false;
	    has_result = false;
	    
	    whiteice::logging.error("simplex<>::threadloop(): cudaMallocManaged() failed.");
	    return; // silent failure
	  }

	  for(unsigned int i=0;i<constraints.size();i++){
	    cudaFree(constraints[i]);
	  }

	  constraints = new_constraints;

	  // sets memory areas to zero (needed?)
	  {
	    std::vector<bool> ok;
	      
	    e = cudaMemset(target, 0, sizeof(T)*(numVariables + numArtificials + 1));
	    if(e != cudaSuccess) ok.push_back(false);

	    e = cudaMemset(pseudotarget, 0, sizeof(T)*(numVariables + numArtificials + 1));
	    if(e != cudaSuccess) ok.push_back(false);

	    for(unsigned int i=0;i<constraints.size();i++){
	      e = cudaMemset(constraints[i], 0,
			     sizeof(T)*(numVariables + numArtificials + 1));
	      if(e != cudaSuccess) ok.push_back(false);
	    }

	    if(ok.size() > 0){ // failures in memsets
	      cudaFree(pseudotarget);

	      running = false;
	      has_result = false;
	      
	      whiteice::logging.error("simplex<>::threadloop(): cudaMemset() failed.");
	      return; // silent failure
	    }
	  }
	  
#else
	  
	  if((temp = (T*)realloc(target,
				 sizeof(T)*(numVariables + numRealArtificials + 1))) == 0){
	    running = false;
	    has_result = false;
	    return;
	  }
	  
	  
	  target = temp;
	  
	  // memory for pseudotarget (used to get feasible solution)
	  temp = (T*)calloc(numVariables+numArtificials+1,sizeof(T));
	  if(temp == 0){
	    temp = (T*)realloc(target, sizeof(T)*(numVariables + 1));
	    if(temp)
	      target = temp;
	    
	    running = false;
	    has_result = false;
	    
	    return;
	  }
	  
	  pseudotarget = temp;
	  
	  for(unsigned int i=0;i<constraints.size();i++){
	    temp = (T*)realloc(constraints[i], sizeof(T)*(numVariables + numArtificials + 1));
	    
	    if(temp == 0){ // failure, back to original state
	      for(unsigned int j=0;j<i;j++){
		temp = (T*)realloc(constraints[j], sizeof(T)*(numVariables + 1));
		if(temp)
		  constraints[j] = temp;
	      }
	      
	      temp = (T*)realloc(target, sizeof(T)*(numVariables + 1));
	      if(temp)
		target = temp;
	      
	      free(pseudotarget);
	      
	      running = false;
	      has_result = false;
	      
	      return;
	    }
	    
	    constraints[i] = temp;
	  }

#endif
	  
	  
	  // memory allocations were ok (in theory)
	  // actually setups simplex table correctly
	  
	  T RHS = target[numVariables];
	  memset(&(target[numVariables]), 0, numRealArtificials*sizeof(T)); // TODO USE CUDA
	  target[numVariables+numRealArtificials] = RHS;
	  
	  unsigned int countReals   = 0;
	  unsigned int countPseudos = 0;
	  
	  // setups also basic (slacks) and non basic variables
	  basic.clear();
	  nonbasic.clear();
	  
	  for(unsigned int i=0;i<numVariables;i++)
	    nonbasic.push_back(i);
	  
	  for(unsigned int i=0;i<constraints.size();i++){
	    T RHS = constraints[i][numVariables];
	    memset(&(constraints[i][numVariables]), 0, numArtificials*sizeof(T)); // TODO USE CUDA
	    constraints[i][numVariables+numArtificials] = RHS;
	    
	    // setups constraint coefficients
	    if(ctypes[i] == 0){ // slack
	      basic.push_back(numVariables+countReals); // slack variable number
	      constraints[i][numVariables+countReals] = T(1.0);
	      countReals++;	    
	    }
	    else if(ctypes[i] == 2){ // surplus + pseudoslack
	      basic.push_back(numVariables+numRealArtificials+countPseudos);
	      nonbasic.push_back(numVariables+countReals);
	      
	      constraints[i][numVariables+countReals]   = T(-1.0); // surplus
	      constraints[i][numVariables+numRealArtificials+countPseudos] = T(+1.0); // pseudoslack
	      countReals++;
	      countPseudos++;
	    }
	    else{ // pseudoslack only
	      basic.push_back(numVariables+numRealArtificials+countPseudos);
	      
	      constraints[i][numVariables+numRealArtificials+countPseudos] = T(+1.0); // pseudoslack
	      countPseudos++;
	    }
	  }
	  
	  // creates target for minimizing sum of pseudo variables
	  
	  for(unsigned int i=0;i<countPseudos;i++)
	    pseudotarget[numVariables+numRealArtificials+i] = T(-1.0);
	  
	  pseudotarget[numVariables+numArtificials] = T(0.0); // sets RHS	
	  
	  this->numArtificials = numArtificials;
	  
	  
	  // removes slacks from initial pseudotarget function
	  for(unsigned int i=0;i<basic.size();i++){
	    // initial basic variables are slacks
	    
	    if(pseudotarget[basic[i]] != T(0.0)){
	      for(unsigned int j=0;j<numVariables+numArtificials+1;j++)
		pseudotarget[j] += constraints[i][j];
	    }
	  }
	  
	  
	  // transforms minimization to maximization task
	  // min f(x) = max -f(x)
	  for(unsigned int j=0;j<numVariables+numArtificials+1;j++)
	    pseudotarget[j] = -pseudotarget[j];
	  
	  
	  // minimizes pseudotarget via normal method
	  {
	    while(find_indexes(pseudotarget, constraints,
			       eindex, lindex)){
	      
	      // calculates new lindex row
	      
	      T c = T(1.0)/constraints[lindex][eindex];
	      
#ifdef CUBLAS
	      auto e = cublasDscal(cublas_handle, numVariables+numArtificials+1,
				   (const double*)&c,
				   (double*)constraints[lindex], 1);

	      if(e != CUBLAS_STATUS_SUCCESS){
		whiteice::logging.error("simplex<>::threadloop(): cublasDscal() failed.");
		cudaFree(pseudotarget);
		running = false;
		has_result = false;
		return;
	      }
#else
	      cblas_dscal(numVariables+numArtificials+1, *((double*)&c), 
			  (double*)constraints[lindex], 1);
#endif
	      
	      // updates pseudotarget
	      
	      c = -pseudotarget[eindex];

#ifdef CUBLAS
	      e = cublasDaxpy(cublas_handle, numVariables+numArtificials+1,
			      (const double*)&c,
			      (const double*)constraints[lindex], 1,
			      (double*)pseudotarget, 1);

	      if(e != CUBLAS_STATUS_SUCCESS){
		whiteice::logging.error("simplex<>::threadloop(): cublasDaxpy() failed.");
		cudaFree(pseudotarget);
		running = false;
		has_result = false;
		return;
	      }
#else
	      
	      cblas_daxpy(numVariables+numArtificials+1, *((double*)&c),
			  (double*)constraints[lindex], 1, (double*)pseudotarget, 1);
#endif
	      
	      // updates other rows
	      
	      if(lindex != 0){
		for(unsigned int i=0;i<lindex;i++){
		  c = -constraints[i][eindex];

#ifdef CUBLAS
		  e = cublasDaxpy(cublas_handle, numVariables+numArtificials+1,
				  (const double*)&c,
				  (const double*)constraints[lindex], 1,
				  (double*)constraints[i], 1);

		  if(e != CUBLAS_STATUS_SUCCESS){
		    whiteice::logging.error("simplex<>::threadloop(): cublasDaxpy() failed.");
		    cudaFree(pseudotarget);
		    running = false;
		    has_result = false;
		    return;
		  }
#else
		  cblas_daxpy(numVariables+numArtificials+1, *((double*)&c),
			      (double*)constraints[lindex], 1, (double*)constraints[i], 1);
#endif
		}
	      }
	      
	      
	      if(lindex != constraints.size()-1){
		for(unsigned int i=lindex+1;i<constraints.size();i++){
		  c = -constraints[i][eindex];

#ifdef CUBLAS
		  e = cublasDaxpy(cublas_handle, numVariables+numArtificials+1,
				  (const double*)&c,
				  (const double*)constraints[lindex], 1,
				  (double*)constraints[i], 1);

		  if(e != CUBLAS_STATUS_SUCCESS){
		    whiteice::logging.error("simplex<>::threadloop(): cublasDaxpy() failed.");
		    cudaFree(pseudotarget);
		    running = false;
		    has_result = false;
		    return;
		  }
#else
		  
		  cblas_daxpy(numVariables+numArtificials+1, *((double*)&c),
			      (double*)constraints[lindex], 1, (double*)constraints[i], 1);
#endif
		}
	      }
	      
	      
	      // updates basic/non-basic sets
	      
	      {
		unsigned int varleave = basic[lindex];
		basic[lindex] = eindex;
		
		for(unsigned int i=0;i<nonbasic.size();i++){
		  if(nonbasic[i] == eindex){
		    nonbasic[i] = varleave;
		    break;
		  }
		}
	      }
	      
	      
	      this->iter++;
	    }
	  }
	  
	  
	  if(abs(pseudotarget[numVariables+numArtificials]) >= T(1e-5)){
	    running = false;
	    has_result = false;
	    
	    return;
	  }
	  
	  
	  // checks if pseudoslack variable is still
	  // in basic solution set (must be zero)
	  {
	    bool hasPseudoSlackVar = false;
	  
	    for(unsigned int i=0;i<basic.size();i++){
	      if(basic[i] >= numVariables + numRealArtificials){
		hasPseudoSlackVar = true;
		break;
	      }
	    }
	  
	    if(hasPseudoSlackVar){
	      // find_indexes2() is same as find_indexes()
	      // except that pseudoslacks are given preference
	      // when considering leaving variables
	      
	      while(find_indexes2(pseudotarget, constraints,
				  eindex, lindex, basic,
				  numVariables + numRealArtificials)){
		
		// calculates new lindex row
	      
		T c = T(1.0)/constraints[lindex][eindex];

#ifdef CUBLAS
		auto e = cublasDscal(cublas_handle, numVariables+numArtificials+1,
				     (const double*)&c,
				     (double*)constraints[lindex], 1);
		
		if(e != CUBLAS_STATUS_SUCCESS){
		  whiteice::logging.error("simplex<>::threadloop(): cublasDscal() failed.");
		  cudaFree(pseudotarget);
		  running = false;
		  has_result = false;
		  return;
		}
		
#else
		cblas_dscal(numVariables+numArtificials+1, *((double*)&c), 
			    (double*)constraints[lindex], 1);
#endif
		
		// updates pseudotarget
		
		c = -pseudotarget[eindex];

#ifdef CUBLAS
		
		e = cublasDaxpy(cublas_handle, numVariables+numArtificials+1,
				(const double*)&c,
				(const double*)constraints[lindex], 1,
				(double*)pseudotarget, 1);
		
		if(e != CUBLAS_STATUS_SUCCESS){
		  whiteice::logging.error("simplex<>::threadloop(): cublasDaxpy() failed.");
		  cudaFree(pseudotarget);
		  running = false;
		  has_result = false;
		  return;
		}
		
#else
		cblas_daxpy(numVariables+numArtificials+1, *((double*)&c),
			    (double*)constraints[lindex], 1, (double*)pseudotarget, 1);
#endif
		
		// updates other rows
		
		if(lindex != 0){
		  for(unsigned int i=0;i<lindex;i++){
		    c = -constraints[i][eindex];

#ifdef CUBLAS
		    e = cublasDaxpy(cublas_handle, numVariables+numArtificials+1,
				    (const double*)&c,
				    (const double*)constraints[lindex], 1,
				    (double*)constraints[i], 1);
		    
		    if(e != CUBLAS_STATUS_SUCCESS){
		      whiteice::logging.error("simplex<>::threadloop(): cublasDaxpy() failed.");
		      cudaFree(pseudotarget);
		      running = false;
		      has_result = false;
		      return;
		    }
		    
#else
		    cblas_daxpy(numVariables+numArtificials+1, *((double*)&c),
				(double*)constraints[lindex], 1, (double*)constraints[i], 1);
#endif
		  }
		}
		
		
		if(lindex != constraints.size()-1){
		  for(unsigned int i=lindex+1;i<constraints.size();i++){
		    c = -constraints[i][eindex];

#ifdef CUBLAS
		    e = cublasDaxpy(cublas_handle, numVariables+numArtificials+1,
				    (const double*)&c,
				    (const double*)constraints[lindex], 1,
				    (double*)constraints[i], 1);
		    
		    if(e != CUBLAS_STATUS_SUCCESS){
		      whiteice::logging.error("simplex<>::threadloop(): cublasDaxpy() failed.");
		      cudaFree(pseudotarget);
		      running = false;
		      has_result = false;
		      return;
		    }
		    
#else
		    cblas_daxpy(numVariables+numArtificials+1, *((double*)&c),
				(double*)constraints[lindex], 1, (double*)constraints[i], 1);
#endif
		  }
		}
		
		
		// updates basic/non-basic sets
		
		{
		  unsigned int varleave = basic[lindex];
		  basic[lindex] = eindex;
		  
		  for(unsigned int i=0;i<nonbasic.size();i++){
		    if(nonbasic[i] == eindex){
		      nonbasic[i] = varleave;
		      break;
		    }
		  }
		}
		
		
		this->iter++;
	      }
	      
	      
	      /////////////////////////////////////////////////
	      // checks if solution is feasible
	      if(abs(pseudotarget[numVariables+numArtificials]) >= T(1e-5)){
		running = false;
		has_result = false;
		return;
	      }
	      
	      // checks if solution still have pseudoslacks
	      
	      for(unsigned int i=0;i<basic.size();i++){
		if(basic[i] >= numVariables + numRealArtificials){
		  running = false;
		  has_result = false;		
		  return;
		}
	      }
	      
	    }
	  }

#ifdef CUBLAS
	  cudaFree(pseudotarget);
#else
	  free(pseudotarget);
#endif
	  
	  // pseudoslacks are now zero so it is
	  // safe to remove them
	  
	  // resizes constraints

#ifdef CUBLAS

	  for(unsigned int i=0;i<constraints.size();i++){
	    T RHS = constraints[i][numVariables + numArtificials];
	    
	    auto e = cudaMallocManaged(&temp, sizeof(T)*(numVariables + numRealArtificials + 1));

	    if(e != cudaSuccess){
	      whiteice::logging.error("simplex<>::threadloop(): cudaMallocManaged() failed.");
	      
	      running = false; // silent failure
	      has_result = false;
	      return;
	    }

	    const int original_size = numVariables + numArtificials + 1;
	    const int new_size = numVariables + numRealArtificials + 1;

	    if(new_size <= original_size){
	      e = cudaMemcpy(temp, constraints[i], new_size*sizeof(T),
			     cudaMemcpyDeviceToDevice);
	    }
	    else{
	      e = cudaMemcpy(temp, constraints[i], original_size*sizeof(T),
			     cudaMemcpyDeviceToDevice);
	    }

	    if(e != cudaSuccess){
	      cudaFree(temp);
	      whiteice::logging.error("simplex<>::threadloop(): cudaMemcpy() failed.");
	      
	      running = false; // silent failure
	      has_result = false;
	      return;
	    }

	    cudaFree(constraints[i]);
	    constraints[i] = temp;
	    constraints[i][numVariables+numRealArtificials] = RHS;
	  }
	  
	  this->numArtificials = numRealArtificials;
	  
	  std::vector<unsigned int>::iterator i;
	  i = nonbasic.begin();
	  
	  while(i != nonbasic.end()){
	    if(*i >= numVariables + numRealArtificials)
	      i = nonbasic.erase(i);
	    else
	      i++;
	  }
	  
#else
	
	  for(unsigned int i=0;i<constraints.size();i++){
	    T RHS = constraints[i][numVariables + numArtificials];
	    temp = (T*)realloc(constraints[i], sizeof(T)*(numVariables + numRealArtificials + 1));
	    
	    if(temp == 0){ // hard to return back to original from here
	      running = false;
	      has_result = false;		
	      return;
	    }
	    
	    constraints[i] = temp;
	    constraints[i][numVariables+numRealArtificials] = RHS;
	  }
	  
	  this->numArtificials = numRealArtificials;
	  
	  std::vector<unsigned int>::iterator i;
	  i = nonbasic.begin();
	  
	  while(i != nonbasic.end()){
	    if(*i >= numVariables + numRealArtificials)
	      i = nonbasic.erase(i);
	    else
	      i++;
	  }
#endif
	}
	
	
	// simplex table fixup
	// sets target function coefficients of
	// basic variables to zero
	// (needed because target function changed:
	//  switch to real target function)
	
	for(unsigned int i=0;i<basic.size();i++){
	  T c = target[basic[i]];
	  
	  for(unsigned int j=0;j<numVariables+numArtificials+1;j++)
	    target[j] -= c*constraints[i][j];
	}
	
	
	
	while(find_indexes(target, constraints,
			   eindex, lindex)){
	    
	  // calculates new lindex row
	  
	  T c = T(1.0)/constraints[lindex][eindex];

#ifdef CUBLAS
	  
	  auto e = cublasDscal(cublas_handle, numVariables+numArtificials+1,
			       (const double*)&c,
			       (double*)constraints[lindex], 1);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("simplex<>::threadloop(): cublasDscal() failed.");
	    running = false;
	    has_result = false;
	    return;
	  }
	  
#else
	  cblas_dscal(numVariables+numArtificials+1, *((double*)&c), 
		      (double*)constraints[lindex], 1);
#endif
	  
	  // updates target
	  
	  c = -target[eindex];

#ifdef CUBLAS

	  e = cublasDaxpy(cublas_handle, numVariables+numArtificials+1,
			  (const double*)&c,
			  (const double*)constraints[lindex], 1,
			  (double*)target, 1);
	  
	  if(e != CUBLAS_STATUS_SUCCESS){
	    whiteice::logging.error("simplex<>::threadloop(): cublasDaxpy() failed.");
	    running = false;
	    has_result = false;
	    return;
	  }
	  
#else
	  cblas_daxpy(numVariables+numArtificials+1, *((double*)&c),
		      (double*)constraints[lindex], 1, (double*)target, 1);
#endif
	  
	  // updates other rows
	  
	  if(lindex != 0){
	    for(unsigned int i=0;i<lindex;i++){
	      c = -constraints[i][eindex];

#ifdef CUBLAS
	      e = cublasDaxpy(cublas_handle, numVariables+numArtificials+1,
			      (const double*)&c,
			      (const double*)constraints[lindex], 1,
			      (double*)constraints[i], 1);
	      
	      if(e != CUBLAS_STATUS_SUCCESS){
		whiteice::logging.error("simplex<>::threadloop(): cublasDaxpy() failed.");
		running = false;
		has_result = false;
		return;
	      }
#else
	      cblas_daxpy(numVariables+numArtificials+1, *((double*)&c),
			  (double*)constraints[lindex], 1, (double*)constraints[i], 1);
#endif
	    }
	  }
	  
	  
	  if(lindex != constraints.size()-1){
	    for(unsigned int i=lindex+1;i<constraints.size();i++){
	      c = -constraints[i][eindex];

#ifdef CUBLAS
	      e = cublasDaxpy(cublas_handle, numVariables+numArtificials+1,
			      (const double*)&c,
			      (const double*)constraints[lindex], 1,
			      (double*)constraints[i], 1);
	      
	      if(e != CUBLAS_STATUS_SUCCESS){
		whiteice::logging.error("simplex<>::threadloop(): cublasDaxpy() failed.");
		running = false;
		has_result = false;
		return;
	      }		  
	      
#else
	      cblas_daxpy(numVariables+numArtificials+1, *((double*)&c),
			  (double*)constraints[lindex], 1, (double*)constraints[i], 1);
#endif
	    }
	  }
	  
	  
	  // updates basic/non-basic sets
	  {
	    unsigned int varleave = basic[lindex];
	    basic[lindex] = eindex;
	  
	    for(unsigned int i=0;i<nonbasic.size();i++){
	      if(nonbasic[i] == eindex){
		nonbasic[i] = varleave;
		break;
	      }
	    }
	  }
	  
	  
	  this->iter++;
	}
    
	
	has_result = true;
	// maximization_thread = 0;
	running = false;
	
	return;
      }
      else{	
	has_result = false;
	// maximization_thread = 0;
	running = false;
	
	return;
      }
    }
    
    
    
    // finds pivot row and pivot element for simplex algorithm
    template <typename T>
    bool simplex<T>::find_indexes(T*& target, std::vector<T*>& constraints,
				  unsigned int& eindex, unsigned int& lindex) const 
    {
      // finds the smallest negative non-zero variable
      // (variable to enter)
      
      const unsigned int RHS = numVariables + numArtificials;
      bool found = false;
      T smallest = T(0.0);            
      
      for(unsigned int i=0;i<RHS;i++){
	if(target[i] < T(0.0)){
	  
	  // checks if all constraint coefficients
	  // are negative. If they are, then this
	  // variable can increase indefinitely
	  // and solution is unbounded so it is
	  // good idea to select variable
	  
	  unsigned int c = 0;
	  
	  for(unsigned int j=0;j<constraints.size();j++){
	    if(constraints[j][i] <= T(0.0))
	      c++;
	  }
	  
	  
	  if(c == constraints.size()){
	    eindex = i;
	    
	    // handles this special case right here
	    // by selecting row with the biggest value
	    
	    lindex = 0;
	    smallest = constraints[lindex][i];
	    
	    for(unsigned int j=0;j<constraints.size();j++){
	      if(constraints[j][i] > smallest){
		smallest = constraints[j][i];
		lindex = j;
	      }
	    }
	    
	    // found good variable swap
	    return true;
	  }
	  else if(target[i] < smallest){
	    smallest = target[i];
	    eindex = i;
	    found = true;
	  }
	}
      }
      
      
      // no variable to enter
      if(found == false)
	return false;
      
      // finds row to leave
      // (the smallest positive ratio)
      
      lindex = 0;
      
      while(lindex < constraints.size()){
	// finds the first 'ok' row
	
	if(constraints[lindex][eindex] > T(0.0))
	  break;
	
	lindex++;
      }
      
      
      if(lindex >= constraints.size())
	return false; // no row/variable to leave
      
      
      // initial ratio
      smallest = constraints[lindex][RHS] / constraints[lindex][eindex];
      
      if(lindex < constraints.size()-1){
	// tries to find smaller ratio		
	
	for(unsigned int i=lindex+1;i<constraints.size();i++){
	  
	  if(constraints[i][eindex] > T(0.0)){
	    T ratio = constraints[i][RHS] / constraints[i][eindex];
	    
	    if(ratio < smallest){
	      smallest = ratio;
	      lindex = i;
	    }
	  }
	}
		
      }
      
      // lindex has row to leave
      // (variable to leave is basic[lindex])
      
      return true;
    }



    // finds pivot row and pivot element for simplex algorithm
    template <typename T>
    bool simplex<T>::find_indexes2(T*& target, std::vector<T*>& constraints,
				   unsigned int& eindex, unsigned int& lindex,
				   const std::vector<unsigned int>& bsols,
				   const unsigned int& pseudoStart) const 
    {
      // pseudoStart containts index of the first pseudovariable
      
      // finds the smallest negative non-zero variable from
      // non-pseudo variables (variable to enter)
      
      for(unsigned int i=0;i<pseudoStart;i++){
	if(abs(target[i]) > T(1e-5)){
	  // checks if some pseudovariable in
	  // solution has non-zero coefficient
	  
	  for(unsigned int j=0;j<constraints.size();j++){
	    // checks j is pseudovariable with
	    // non-zero entry
	    if(bsols[j] >= pseudoStart){
	      if(abs(constraints[j][i]) > T(1e-5)){
		
		// found good row
		eindex = i;
		lindex = j;
		
		return true;
	      }
	    }
	    
	  }
	  
	}
      }
      
      
      // couldn't find any swaps
      // that would replace pseudovariable from
      // solution set with non-pseudovariable
      
      return false;
    }
    
    
    template <typename T>
    bool simplex<T>::hasResult() 
    {
      return (running == false && has_result == true);
    }
    
    
    // format [x_1, x_2,... x_n, F], where F = SUM(c_i*x_i) (optimum value)
    template <typename T>
    bool simplex<T>::getSolution(std::vector<T>& solution) const
    {
      if(running == true || has_result == false)
	return false;
      
      pthread_mutex_lock( &simplex_lock );
      if(running == true || has_result == false){
	pthread_mutex_unlock( &simplex_lock );
	return false;
      }
      
      // solution
      solution.resize(numVariables);
      solution.push_back(target[numVariables+numArtificials]);
      
      // sets basic variables/solution
      for(unsigned int i=0;i<basic.size();i++){
	if(basic[i] >= numVariables)
	  return false;
	solution[basic[i]] = target[basic[i]];
      }
      
      
      pthread_mutex_unlock( &simplex_lock );
      
      return true;
    }
    
    
    // displays simplex solution
    template <typename T>
    bool simplex<T>::show_simplex(T* pseudotarget) const 
    {
      for(unsigned int i=0;i<numVariables+numArtificials+1;i++)
	printf("%+2.2f   ", pseudotarget[i]);
      printf("\n");
      
      for(unsigned int i=0;i<constraints.size();i++){
	for(unsigned int j=0;j<numVariables+numArtificials+1;j++)
	  printf("%+2.2f   ", constraints[i][j]);
	printf("\n");
      }
      
      printf("\n");
      
      // shows basic information
      printf("Variables in basic solution set: ");
      for(unsigned int i=0;i<basic.size();i++)
	printf("%d ", basic[i]);
      printf("\n");
      
      
      // shows nonbasic information
      printf("Non basic variables: ");
      for(unsigned int i=0;i<nonbasic.size();i++)
	printf("%d ", nonbasic[i]);
      printf("\n\n");
      
      return true;
    }
    

    // displays simplex solution
    template <typename T>
    bool simplex<T>::show_simplex() const 
    {
      for(unsigned int i=0;i<numVariables+numArtificials+1;i++)
	printf("%+2.2f   ", target[i]);
      printf("\n");
      
      for(unsigned int i=0;i<constraints.size();i++){
	for(unsigned int j=0;j<numVariables+numArtificials+1;j++)
	  printf("%+2.2f   ", constraints[i][j]);
	printf("\n");
      }
      
      printf("\n");
      
      // shows basic information
      printf("Variables in basic solution set: ");
      for(unsigned int i=0;i<basic.size();i++)
	printf("%d ", basic[i]);
      printf("\n");
      
      
      // shows nonbasic information
      printf("Non basic variables: ");
      for(unsigned int i=0;i<nonbasic.size();i++)
	printf("%d ", nonbasic[i]);
      printf("\n\n");
      
      return true;
    }
    
    
    
    ////////////////////////////////////////////////////////////
    
    template class simplex<float>;
    template class simplex<double>;
    
  };
};


extern "C" { 
  void* __simplex_thread_init(void *simplex_ptr)
  {
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
    
    if(simplex_ptr) // somewhat unsafe if T != float
      ((whiteice::math::simplex<float>*)simplex_ptr)->threadloop();
    
    return 0;
  }
};



