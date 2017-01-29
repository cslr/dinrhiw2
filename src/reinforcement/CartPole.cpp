

#include "CartPole.h"
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
      
      up = T(0.01);  // friction forces
      uc = T(0.01);
      
      Nc = T(0.0);
      
      // cart-pole system system state
      theta = T(0.1);
      theta_dot = T(0.0);
      theta_dotdot = T(0.0);
      
      x = T(0.0);
      x_dot = T(0.0);
      x_dotdot = T(0.0);
      
      // external force
      F = T(0.0);

      // simulatiom timestep
      dt = T(0.010); // 10ms
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
				  T& r)
  {
    // converts action to control in newtons
    double Fstep = 10.0*(((double)action)/10.0 - 1.0);

    {
      F_change.lock();
      F = Fstep;

      while(F != T(0.0)){
	F_change.unlock();
	usleep((unsigned int)(dt.c[0]*10.0));  // waits for single time step to complete
	F_change.lock();
      }

      newstate.resize(4);
      newstate[0] = theta;
      newstate[1] = theta_dot;
      newstate[2] = x;
      newstate[3] = x_dot;      

      F_change.unlock();
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

    
    while(running){

      {
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


	F = T(0.0);
      }

      ////////////////////////////////////////////////////////////////////////
      // we stimulate system for a single timestep

      x = x + x_dot*dt + T(0.5)*x_dotdot*dt*dt;
      theta = theta + theta_dot*dt + T(0.5)*theta_dotdot*dt*dt;

      x_dot = x_dot + x_dotdot*dt;
      theta_dot = theta_dot + theta_dotdot*dt;
      
      usleep((unsigned int)(dt.c[0]*1000.0));  // waits for a single timestep
      
    }
    
    
  }
  


  template class CartPole< math::blas_real<float> >;
  template class CartPole< math::blas_real<double> >;
};
