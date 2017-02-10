

#include "CartPole.h"

#include <stdio.h>
#include <math.h>

#include <chrono>

#include <unistd.h>

#ifdef USE_SDL
#include <SDL.h>
#endif

using namespace std::chrono_literals;

namespace whiteice
{

  template <typename T>
  CartPole<T>::CartPole() : RIFL_abstract<T>(5, 4)
  {
    {
      g = T(9.81); // gravity
      l = T(1.0);  // 1 meter long pole
      
      mc = T(2.000); // cart weight (2.0 kg)
      mp = T(0.200); // pole weight (200g)
      
      up = T(0.01);  // friction forces [pole]
      uc = T(0.1);   // friction between track and a cart
      
      Nc = T(0.0);
      
      reset();
      
      // simulatiom timestep
      dt = T(0.010); // 10ms
      
      iteration = 0;
    }

    
#ifdef USE_SDL
    // create SDL display
    {
      SDL_Init(SDL_INIT_VIDEO);

      window = NULL;
      renderer = NULL;
      
      W = 800;
      H = 600;

      SDL_DisplayMode mode;

      if(SDL_GetCurrentDisplayMode(0, &mode) == 0){
	W = (4*mode.w)/5;
	H = (3*mode.h)/4;
	H = H/2;
      }
      
      
      SDL_CreateWindowAndRenderer(W, H, 0, &window, &renderer);
    }
#endif

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
#ifdef USE_SDL
    {
      if(renderer)
	SDL_DestroyRenderer(renderer);
      
      if(window)
	SDL_DestroyWindow(window);
      
      SDL_Quit();
    }
#endif
    
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
    theta = this->rng.uniform()*T(2.0*M_PI) - T(M_PI); // angle is [-PI, PI]
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

    state[0] = normalizeTheta(theta)/T(M_PI);
    state[1] = theta_dot;
    state[2] = x;
    state[3] = x_dot;

#ifdef USE_SDL
    {
      auto theta = this->theta;
      auto x     = state[2];
      
      SDL_Event event;
      while(SDL_PollEvent(&event)){
	if(event.type == SDL_QUIT){
	  this->running = false;
	}
      }

      // drawing
      {
	// black background
	SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
	// SDL_RenderFillRect(renderer, NULL);
	SDL_RenderClear(renderer);

	SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);

	double y0 = H/2.0;
	double x0 = W/2.0 + x.c[0]/2;

	double l = 100.0; // line length

	double y1 = y0 - l*cos(theta.c[0]);
	double x1 = x0 + l*sin(theta.c[0]);

	SDL_RenderDrawLine(renderer, (int)x0, (int)y0, (int)x1, (int)y1);
	SDL_RenderPresent(renderer);
      }
    }
#endif

    return true;
  }


  template <typename T>
  T CartPole<T>::normalizeTheta(const T t) const throw()
  {
    // sets range between [-2*pi, +2*pi]
    T a = t/T(2.0*M_PI);

    // printf("N: %f\n", a.c[0]);

    // take the fractional part 2.3 -> 0.3, -2.3 -> -0.3
    if(a >= T(0.0)){
      a = a - floor(a);
    }
    else{
      a = -(abs(a) - floor(abs(a)));
    }

    // printf("A: %f\n", a.c[0]); // debugging..

    a = T(2.0*M_PI)*a; // [-2*pi, 2*pi]

    // converts range between [-pi, pi]
    if(a > T(M_PI)){
      a = T(-1.0)*(T(2.0*M_PI) - a);
    }
    else if(a < T(-M_PI)){
      a = T(2.0*M_PI) + a;
    }

    // auto temp = a / T(2.0*M_PI);
    // printf("T: %f\n", temp.c[0]);

    return a;
  }

  
  template <typename T>
  bool CartPole<T>::performAction(const unsigned int action,
				  whiteice::math::vertex<T>& newstate,
				  T& reinforcement)
  {
    // converts action to control in newtons
    // printf("%d action\n", action);
    double Fstep = 0.0;

    double a = ((double)action)/((double)(this->numActions - 1)); // [0,1]
    a = 2.0*a - 1.0; // [-1.0,+1.0]
    Fstep = 25.0*a; // [-25, +25]

    printf("%d FORCE: %f\n", iteration, Fstep);
    
    {
      {
	std::unique_lock<std::mutex> lock(F_change);
	F = Fstep;
	F_processed = false;
	F_processed_cond.notify_all();

	while(F_processed == false){
	  auto now = std::chrono::system_clock::now();
	  F_processed_cond.wait_until(lock, now + 10*100ms);
	  if(running == false)
	    return false;
	}
	
	{
	  newstate.resize(4);
	  newstate[0] = normalizeTheta(theta)/T(M_PI);
	  newstate[1] = theta_dot;
	  newstate[2] = x;
	  newstate[3] = x_dot;
	}
	
	// our target is to keep theta at zero (the largest reinforcement value)
	{
	  // converts range between [-180.0, +180.0]
	  T a = T(180.0)* normalizeTheta(theta) / T(M_PI);

	  // range is ] -1.0, 0.0]; // bigger is better (closer to zero)
	  a = -T(1.0)*abs(a)/T(180.0);

	  
	  // additionally we add minus term for being too far from zero (W)

#if 1
	  T x_reinforcement = abs(x);
	  {
	    if(x_reinforcement > T(W/2.0)){
	      // [0,1] (within screen boundaries)
	      x_reinforcement = (x_reinforcement - T(W/2.0))/T(W/2.0);
	    }
	    else{
	      x_reinforcement = T(0.0);
	    }
	    
	    // (closer to zero is better) [-1.0, 0]
	    x_reinforcement = -abs(x_reinforcement);
	    
	  
	    reinforcement = T(0.5)*(a + x_reinforcement); // [-1.0, 0]
	  }
#endif
	    
	  reinforcement = T(1.0) + a; // we keep things between [0,1]

	  reinforcement = T(0.5)*reinforcement; // keep at scale [0.0,0.5]

	  printf("REINFORCEMENT: %f\n", reinforcement.c[0]);
	  fflush(stdout);
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
	std::unique_lock<std::mutex> lock(F_change);

	while(F_processed == true){
	  auto now = std::chrono::system_clock::now();
	  F_processed_cond.wait_until(lock, now + 10*100ms);
	  if(running == false) return;
	}

	
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

	  thetas.push_back(abs(degrees));

	  auto temp = abs(mth); //  + sth; [DO NOT ADD STANDARD DEVIATION]
	  
	  printf("TIME %f theta = %f deg [%f]\n",
		 t, degrees, temp.c[0]);
	  
	  fflush(stdout);
	}
	
	F_processed = true;
	F_processed_cond.notify_all();
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
