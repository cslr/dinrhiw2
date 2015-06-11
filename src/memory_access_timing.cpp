
#include <iostream>
#include <cstdlib>
#ifndef WINOS
#include <sys/times.h>
#endif
#include <unistd.h>
#include <ctime>
#include <math.h>
// #include <tgmath.h>

#include <vector>
#include "memory_access_timing.h"

using namespace std;

/*
 * timing
 */
static void timing_start();
static float timing_end();
static float timing_accuracy();

/*
 * memory access
 */
static void block_rw(unsigned int *m, int size, int iter);

// statistics

static bool statistics_test(const std::vector<int>& mem_size,
	const std::vector<double>& avg_access_time);


/* T-distribution pdf and pmf functions for T-test */
double pmf_t(double t, double v);
double pdf_t(double t, double v);
static float mean_percent(const std::vector<double>& v, float p1, float p2);


double mem_statistics(unsigned int mem_size, std::vector<double>& means, int& N);


/*
 * saves found memory access time change limits in given vectors
 * returns true if there was no errors (possible zero amount of limits found)
 * returns false in case of failure (vector data unspecified).
 */
bool memory_access_timing(
		std::vector<int>& memborders,
		std::vector<double>& access_times)
{
		/*
		 * strategy: allocates N bytes, writes and then reads everything,
		 * does it M times. calculates average value.
		 * increases N. when avg_access time is significantly bigger than
		 * data found so far set limit to N-5% (adhoc value).
		 * test using average and variance from previous data. assume normal
		 * distribution (adhoc choice). - check if this works. try to do
		 * formal/exact/mathematically correct version later. (best/perfect
		 * cutoff N-x%. and test assume mean increases as a big jump from K bytes
		 * and variance hmm?? - is scaled according to access time scale change? ),
		 * is normal distribution correct.
		 *
		 * other way:
		 * use N_next = 2N sizes, when change is seen, fork the range for more accurate value.
		 * stop at 512 bytes.
		 */
				
		int mem = 1024; // bytes		
		bool done = false;
		
		while(!done)
		{
				unsigned int* memory = 0;
				bool border_found = false;
				
				std::vector<double> means;
				int N;
				double mean[2], var[2], datasize[2];
				
				mem_size.clear();
				avg_access_time.clear();
		
				const int MEMLIMIT = 1024*1025*16; // find way to get total phycial memory use half of it.
				const int TIMELIMIT = 10;
				const int ITERLIMIT = 10;
				int iter;
				
				/* reference data */
				
				mem_statistics(mem, means, N);
				
				for(int i=0, mean[0]=0, var[0]=0;i<means.size();i++){
						mean[0] += means[i];
						var[0]  += 
				}
				
				
				/* increase memory size still there's change */
				while(!border_found && mem < MEMLIMIT) 
				{		
						memory = (unsigned int*)malloc(mem);
						
						/* BRING CODE TO CACHE */
						block_rw(memory,mem, 1);
						
						/* FIND ITERATIONS, TIME > ACCURACY */
						iter = 1;
						float t = 0;
						
						while(t < 10*timing_accuracy())
						{
								timing_start();
								block_rw(memory,mem,iter);
								t = timing_end();
								iter += 4;
								
								if(t > TIMELIMIT) break;
								if(iter >ITERLIMIT) break;
						}
						
						if(t > TIMELIMIT) break;
						
						/* GET SINGLE MEM OPERATION TIME */
						
						timing_start();
						block_rw(memory,mem,iter);
						t = timing_end();
						
						mem_size.push_back(mem/1024);
						avg_access_time.push_back((double)( t / ((double)(mem*iter/(4*1024))) ));
						free(memory);
						
						/* STATISTICAL MEAN CHANGE TEST */
						
						if(mem_size.size() > 10)
						{
								if(statistics_test(mem_size, avg_access_time) == false)
								{
										/* mean change detected */
										border_found = true;
										break;
								}
						}
						
						mem = (int)(1.05*mem);
				}
				
				if(border_found) /* add limit and continue */
				{ 
						memborders.push_back( mem_size[(unsigned int)(mem_size.size()*0.95)] );
						access_times.push_back( mean_percent(avg_access_time, 0.95, 1.00) );
												
						mem = mem_size[(unsigned int)(mem_size.size()*0.96)];
				}
				else{
						done = true; /* all possible limits found */
				}
				
		}
		
		// TODO: add heuristics
		//   try to remove bogus borders - borders where b1 <= b2 <= 4*b1 aren't
		//   probably real. (especially avg. time change is small (reference: all other
		//   avg. time changes in 'neighbourhood'.
		
		return done;
}


/*
 * simple confidence interval test to test if last 20 of
 * material have different mean than first 20  of material
 * uses Welch/Smith-Satterthwaitens test
 * delta_of_means = 0
 * assumes normal distribution of error
 *
 * returns false if mean of data has changed with 90% confidence.
 * 
 */
static bool statistics_test(const std::vector<int>& mem_size,
	const std::vector<double>& avg_access_time)
{
		double alpha = 0.10; // risk level alpha
		
		unsigned int n1 = (unsigned int)(mem_size.size()*0.60);
		unsigned int n2 = mem_size.size() - n1;
		
		if(n1 < 1 || n2 < 1) return true;
		
		/* estimates from data */
		double mean_a[2], var_a[2]; 
		
		mean_a[0] = mean_a[1] = 0.0;
		var_a[0] = var_a[1] = 0.0;
		
		for(unsigned int i=0;i<avg_access_time.size();i++){
				if(i < n1){
						mean_a[0] += avg_access_time[i]; // E[X]*N
				}
				else if(i >= n2){
						mean_a[1] += avg_access_time[i];
				}
		}
		
		mean_a[0] /= n1; // E[X]
		mean_a[1] /= n2;
		
		for(unsigned int i=0;i<avg_access_time.size();i++){
				if(i < n1){
						var_a[0]  += (avg_access_time[i] - mean_a[0])*(avg_access_time[i] - mean_a[0]);
				}
				else if(i >= n2){
						var_a[1]  += (avg_access_time[i] - mean_a[1])*(avg_access_time[i] - mean_a[1]);
				}
		}
				
		var_a[0] /= n1 - 1; // S1^2
		var_a[1] /= n2 - 1;
				
		cout << "means: " << mean_a[0] << " " << mean_a[1] << endl;
		cout << "vars : " << var_a[0] <<  " " << var_a[1] << endl;
		cout << "N = " << mem_size.size() << " MEM = " << mem_size[mem_size.size()-1] << endl;
		
		double t0 = (mean_a[0] - mean_a[1]) / sqrt(var_a[0]/n1 + var_a[1]/n2);
		double degree_freedoms = 
				( (var_a[0]/n1 + var_a[1]/n2)*(var_a[0]/n1 + var_a[1]/n2) ) /
				(  (var_a[0]/n1)*(var_a[0]/n1)/(n1-1) + (var_a[1]/n2)*(var_a[1]/n2)/(n2-1) );
		
		if(t0 < 0) t0 = -t0;
		
		double Pvalue = 2*( 1 - pmf_t( t0, degree_freedoms) );
		
		cout << "STATTEST: " << Pvalue << "\n";
		
		if(Pvalue  < alpha ){
				return false;
		}
		else return true; // everything's fine.
}


static double my_gamma(double v)
{
		double lgam = lgamma(v);
		return signgam*exp(lgam);
}


/*
 * probability density function of T-variable
 */
double pdf_t(double t, double v)
{
  double r = my_gamma((v+1.0)/2.0)/(my_gamma(v/2.0)*sqrt(v*M_PI));
  
  r = r * powf( 1 + t*t/v , -(v+1)/2.0 );
  
  return r;
}

/* 
 * derivate of T-variable pdf
 */
double dpdf_t(double t, double v)
{
  double r = my_gamma((v+1.0)/2.0)/(my_gamma(v/2.0)*sqrt(v*M_PI));
  
  r = r * (-(v+1)/v) * powf( 1 + t*t/v , -((v+1)/2.0) - 1.0 ) * t;
  
  return r;
}


double pmf_t(double t, double v)
{
  /* 
   * numerical integration of pdf_t (steiner's rule?)
   * - palkki + kolmio based approximation, - inf .. inf (-100..
   * step length = step/f'(x)
   */
  
  double x = -5.0;
  double step = 0.01;
  double h = 0.01;
  double sum = 0;
  
  if(t <  x) return 0.0;
  if(t > -x) return 1.0;
  
  while(x<t){
    
    sum += pdf_t(x, v)*step;                 // bar
    sum += 0.5*(pdf_t(x+step,v)-pdf_t(x,v))*step; // triangle
    x += step;
  }
  
  x -= step;
  
  // final step
  sum += pdf_t(x, v)*(t-x); // bar
  sum += 0.5*(pdf_t(t,v)-pdf_t(x,v))*(t-x); // triangle
  
  return sum;
}


/*
 * calculates mean average memory access time when memory size is
 * 
double mem_statistics(unsigned int mem_size, std::vector<double>& means, int& N)
{
}


/*
 * mean of values between p1% .. p2%
 */
static float mean_percent(const std::vector<double>& v, float p1, float p2)
{
  int n1 = (int)(p1*v.size());
  int n2 = (int)(p2*v.size());
  
  float sum = 0;
  
  for(int i=n1;i<n2;i++){
    sum += v[i];
  }
  
  sum /= (n2-n1);
  
  return sum;
}



/*
 * writes and reads memory
 */
static void block_rw(unsigned int *m, int size, int iter)
{	
  for(int j=0;j<iter;j++){
    for(int i=0;i<size/4;i++){
      if(m[i]) m[i] = i;
      else m[i] = 2*i;
    }
  }
}


/*
 * timing rutints
 */

static tms ts[2];

static void timing_start()
{
  times(&ts[0]);
}



static float timing_end()
{
  times(&ts[1]);
  return ((float)(ts[1].tms_utime - ts[0].tms_utime) / 
	  (float)sysconf(_SC_CLK_TCK));
}


static float timing_accuracy(){
  return (1.0/(float)sysconf(_SC_CLK_TCK));
}












