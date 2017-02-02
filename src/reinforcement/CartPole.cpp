

#include "CartPole.h"

#include <stdio.h>
#include <math.h>

#include <unistd.h>


namespace whiteice
{

  template <typename T>
  CartPole<T>::CartPole() : RIFL_abstract<T>(21, 4)
  {
    {
      g = T(9.81); // gravity
      l = T(1.0);  // 1 meter long pole
      
      mc = T(2.000); // cart weight (2.0 kg)
      mp = T(0.200); // pole weight (200g)
      
      up = T(0.1);  // friction forces
      uc = T(0.05);
      
      Nc = T(0.0);
      
      reset();
      
      // simulatiom timestep
      dt = T(0.010); // 10ms
      
      iteration = 0;
    }

    // starts physics thread
    {
      std::lock_guard<std::mutex> lock(physics_mutex);
      
      running = true;
      physics_thread = new thread(std::bind(&CartPole<T>::physicsLoop, this));
    }
    
  }


  template <typename T>
  CartPole<T>::~CartPole()
  {
    // stops physics thread
    {
      std::lock_guard<std::mutex> lock(physics_mutex);

      running = false;
      physics_thread->join();
    }
  }


  template <typename T>
  void CartPole<T>::reset()
  {
    // cart-pole system system state
    theta = T(0.1);
    theta_dot = T(0.0);
    theta_dotdot = T(0.0);
    
    x = T(0.0);
    x_dot = T(0.0);
    x_dotdot = T(0.0);

    // external force
    F = T(0.0);
    F_processed = false;
  }


  template <typename T>
  bool CartPole<T>::getState(whiteice::math::vertex<T>& state)
  {
    state.resize(4);

    state[0] = theta;
    state[1] = theta_dot;
    state[2] = x;
    state[3] = x_dot;

    return true;
  }

  
  template <typename T>
  bool CartPole<T>::performAction(const unsigned int action,
				  whiteice::math::vertex<T>& newstate,
				  T& reinforcement)
  {
    // converts action to control in newtons
    // printf("%d action\n", action);
    double Fstep = 0.0;

    double a = (((double)action) - 10.0)/10.0; // [-1,+1] (0.1 step length)
    Fstep = 5.0*a; // [-5, +5]

    printf("%d FORCE: %f\n", iteration, Fstep);
    
    {
      {
	std::lock_guard<std::mutex> lock(F_change);
	F = Fstep;
	F_processed = false;
      }

      while(F_processed == false){
	usleep(1); // waits till F is processed
	if(running == false) return false;
      }

      {
	std::lock_guard<std::mutex> lock(F_change);

	{
	  // keeps range between [-2*pi, +2*pi]
	  T a = theta/T(2.0*M_PI);
	  a = a - floor(a);
	  a = T(2.0*M_PI)*a;

	  // converts range between [-pi, pi]
	  if(a > T(M_PI)){
	    a = T(-1.0)*(T(2.0*M_PI) - a);
	  }

	  // converts range between [-1.0, +1.0]
	  a = a / T(M_PI);
	  
	  newstate.resize(4);
	  newstate[0] = a;
	  newstate[1] = theta_dot;
	  newstate[2] = x;
	  newstate[3] = x_dot;
	}
	
	// our target is to keep theta at zero (the largest reinforcement value)
	{
	  // keeps range between [-2*pi, +2*pi]
	  T a = theta/T(2.0*M_PI);
	  a = a - floor(a);
	  a = T(2.0*M_PI)*a;

	  // converts range between [-pi, pi]
	  if(a > T(M_PI)){
	    a = T(-1.0)*(T(2.0*M_PI) - a);
	  }

	  // converts range between [-180.0, +180.0]
	  a = T(180.0)* a / T(M_PI);

	  if(a <= T(5.0)){
	    a = T(0.4);
	  }
	  else{ // range is ] -5/180, -1.0];
	    a = -T(1.0)*abs(a)/T(180.0);
	  }
	  
	  reinforcement = a;
	}
      }
	
    }

    return true;
  }

  
  template <typename T>
  T CartPole<T>::sign(T value)
  {
    if(value >= T(0.0)) return T(+1.0);
    else return T(-1.0);
  }

  
  template <typename T>
  void CartPole<T>::physicsLoop()
  {    
    double t = 0.0;

    std::vector<T> thetas;
    T mth = T(0.0);
    T sth = T(0.0);
    
    
    while(running){
      
      {
	while(F_processed == true){
	  usleep(1); // waits for new F
	  if(running == false) return;
	}
	
	std::lock_guard<std::mutex> lock(F_change);

	theta_dotdot =
	  g*sin(theta) +
	  cos(theta) * ( (-F - mp*l*theta_dot*theta_dot*
			  (sin(theta) + uc*sign(Nc*x_dot)*cos(theta)))/(mc + mp) +
			 uc*g*sign(Nc*x_dot)) - up*theta_dot/(mp*l);
	theta_dotdot /= l*(T(4.0/3.0) -
			   (mp*cos(theta)/(mc+mp))*(cos(theta) - uc*sign(Nc*x_dot)));
	
	auto oldNc = Nc;
	
	Nc = (mc + mp)*g -
	  mp*l*(theta_dotdot*sin(theta) + theta_dot*theta_dot*cos(theta));
	
	if(sign(oldNc) != sign(Nc)){
	  theta_dotdot =
	    g*sin(theta) +
	    cos(theta) * ( (-F - mp*l*theta_dot*theta_dot*
			    (sin(theta) + uc*sign(Nc*x_dot)*cos(theta)))/(mc + mp) +
			   uc*g*sign(Nc*x_dot)) - up*theta_dot/(mp*l);
	  theta_dotdot /= l*(T(4.0/3.0) -
			     (mp*cos(theta)/(mc+mp))*(cos(theta) - uc*sign(Nc*x_dot)));
	}
	
	x_dotdot = F + mp*l*(theta_dot*theta_dot*sin(theta) -
			     theta_dotdot*cos(theta)) - uc*Nc*sign(Nc*x_dot);
	x_dotdot /= mc + mp;

	////////////////////////////////////////////////////////////////////////
	// we stimulate system for a single timestep
	
	x = x + x_dot*dt + T(0.5)*x_dotdot*dt*dt;
	theta = theta + theta_dot*dt + T(0.5)*theta_dotdot*dt*dt;
	
	x_dot = x_dot + x_dotdot*dt;
	theta_dot = theta_dot + theta_dotdot*dt;

	t += dt.c[0];
	
	// prints theta in degrees
	{
	  // keeps range between [-2*pi, +2*pi]
	  T a = theta/T(2.0*M_PI);
	  a = a - floor(a);
	  a = T(2.0*M_PI)*a;

	  if(a > T(M_PI)){
	    a = T(-1.0)*(T(2.0*M_PI) - a);
	  }

	  auto degrees = 360.0*(a.c[0]/(2.0*M_PI));

	  thetas.push_back(degrees);

	  auto temp = abs(mth) + sth;
	  
	  printf("TIME %f theta = %f deg [%f]\n",
		 t, degrees, temp.c[0]);
	  
	  fflush(stdout);
	}
	
	F_processed = true;
      }

      if(t >= 20.0){
	iteration++;
	t = 0.0;

	mth = T(0.0);
	sth = T(0.0);

	for(auto& d : thetas){
	  mth += d;
	  sth += d*d;
	}

	mth /= T((double)thetas.size());
	sth /= T((double)thetas.size());

	sth = sth - mth*mth;
	sth = sqrt(abs(sth));
	
	thetas.clear();
	
	reset(); // reset parameters of cart-pole
      }

      // usleep((unsigned int)(dt.c[0]*1000.0));  // waits for a single timestep
      
    }
    
    
  }
  


  template class CartPole< math::blas_real<float> >;
  template class CartPole< math::blas_real<double> >;
};