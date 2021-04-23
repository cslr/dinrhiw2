/*
 * Simple rotation test problem: max(x), 

 * when a sphere point in x^2 + y^2 + z^2 = 1 
 * is rotated around origo for 100 iterations.
 */

#ifndef __whiteice__rotation_problem_h
#define __whiteice__rotation_problem_h

#include "RIFL_abstract2.h"

#include <condition_variable>


namespace whiteice
{

  template <typename T>
  class AdditionProblem : public RIFL_abstract2<T>
  {
  public:
    AdditionProblem();
    ~AdditionProblem();

    bool additionIsRunning(){ return this->running; }
      
  protected:
    
    virtual bool getState(whiteice::math::vertex<T>& state);
    
    virtual bool performAction(const whiteice::math::vertex<T>& action,
			       whiteice::math::vertex<T>& newstate,
			       T& reinforcement,
			       bool& endFlag);
    
  protected:

    // resets rotation variables to random values
    void reset();

    whiteice::math::vertex<T> state;
    
    bool resetLastStep;
    int iteration;

    volatile bool running;
  };

  
  extern template class AdditionProblem< math::blas_real<float> >;
  extern template class AdditionProblem< math::blas_real<double> >;
};


#endif
