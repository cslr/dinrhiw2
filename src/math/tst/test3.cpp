/*
 * optimization methods
 */

#include <iostream>
#include <exception>
#include <stdlib.h>
#include <time.h>

#include "optimized_function.h"
#include "maximizer.h"
#include "blas_primitives.h"
#include "integer.h"

#include "odefunction.h"
#include "RungeKutta.h"
#include "real.h"


void test_optimization();
void test_permfact();
void test_rungekutta();
void test_realnumber();

using namespace whiteice::math;



int main(int argc, char** argv)
{
  try{
    srand(time(0));  
    
    test_rungekutta();
    
    // test_optimization();
    // test_permfact();
    
    test_realnumber();
    
    return 0;
  }
  catch(std::exception& e){
    std::cout << "Unhandled exception: "
	      << e.what() << std::endl;
    
    return -1;
  }
}


/* simple search problem for
 * high dimensions. 
 * N randomly located 
 * unscaled gaussian distributions (exponentials) in D
 * dimensional space with variance of 1/D
 *
 * unless balls are close to each other (unlikely)
 * maximal value is 1
 *
 * this should be quite realistic high-dimensional
 * probability distribution
 */

class gaussian_balls : public whiteice::optimized_function< blas_real<float> >
{
public:
  
  gaussian_balls(unsigned int N, unsigned int D){
    centers.resize(N);
    dim = D;
    
    for(unsigned int i=0;i<centers.size();i++){
      centers[i].resize(D);
      for(unsigned int j=0;j<D;j++)
	centers[i][j] = 2.0f*((float)rand()/((float)rand())) - 1.0f;
      
    }
    
    variance = 10.0f; // 'easyness'
  }
  
  virtual ~gaussian_balls(){
  }
  
  unsigned int dimension() const throw() {
    return dim;
  }
  
  // calculates value of function
  blas_real<float> operator() (const whiteice::math::vertex<blas_real<float> >& x) const {
    return this->calculate(x);
  }
  
  // calculates value
  blas_real<float> calculate(const whiteice::math::vertex<blas_real<float> >& x) const {
    
    whiteice::math::vertex<blas_real<float> > r(dim);
    blas_real<float> result = 0.0f;    
    
    for(unsigned int i=0;i<centers.size();i++){
      r = (centers[i] - x)/variance;
      result += whiteice::math::exp((-(r * r))[0]);
    }
    
    return result;
  }
  
  
  // creates copy of object
  whiteice::function<whiteice::math::vertex<blas_real<float> >,blas_real<float> >* clone() const {
    gaussian_balls* gb = 
      new gaussian_balls(centers.size(), dim);
    
    for(unsigned int i=0;i<centers.size();i++)
      for(unsigned int j=0;j<centers[i].size();j++)
	gb->centers[i][j] = centers[i][j];
    
    gb->variance = variance;
    gb->dim = dim;
    
    return gb;
  }
  
  blas_real<float> goodvalue() const{
    return calculate(centers[0]);
  }
  

  void calculate(const whiteice::math::vertex< whiteice::math::blas_real<float> >& x, blas_real<float>& y) const{
    y = calculate(x);
    return; }
  
  // dummy functions
  
  bool hasGradient() const throw(){ return false; }
  whiteice::math::vertex< blas_real<float> > grad(whiteice::math::vertex< blas_real<float> >& x) const{ return x; }
  void grad(whiteice::math::vertex< blas_real<float> >& x, whiteice::math::vertex< blas_real<float> >& y) const{ return; }
      
  bool hasHessian() const throw(){ return false; }
  whiteice::math::matrix< blas_real<float> > hessian(whiteice::math::vertex< blas_real<float> >& x) const{ return x.outerproduct(); }
  void hessian(whiteice::math::vertex< blas_real<float> >& x, whiteice::math::matrix< blas_real<float> >& y) const{ return; }

private:
  std::vector< whiteice::math::vertex<blas_real<float> > > centers;
  unsigned int dim;
  blas_real<float> variance;
};



class gaussian_halfballs : public whiteice::optimized_function< blas_real<float> >
{
public:
  
  gaussian_halfballs(unsigned int N, unsigned int D){
    centers.resize(N);
    dim = D;
    
    for(unsigned int i=0;i<centers.size();i++){
      centers[i].resize(D);
      for(unsigned int j=0;j<D;j++)
	centers[i][j] = 2.0f*((float)rand()/((float)rand())) - 1.0f;
      
      for(unsigned int j=50;j<D;j++)
	centers[i][j] = 0.0f;
      
    }
    
    variance = 10.0f; // 'easyness'
  }
  
  virtual ~gaussian_halfballs(){
  }
  
  unsigned int dimension() const throw() {
    return dim;
  }
  
  // calculates value of function
  blas_real<float> operator() (const whiteice::math::vertex<blas_real<float> >& x) const {
    return this->calculate(x);
  }
  
  // calculates value
  blas_real<float> calculate(const whiteice::math::vertex<blas_real<float> >& x) const {
    
    whiteice::math::vertex<blas_real<float> > r(dim);
    blas_real<float> result = 0.0f;    
    
    for(unsigned int i=0;i<centers.size();i++){
      r = (centers[i] - x)/variance;
      result += whiteice::math::exp((-(r * r))[0]);
    }
    
    return result;
  }
  
  // creates copy of object
  whiteice::function<whiteice::math::vertex<blas_real<float> >,blas_real<float> >* clone() const {
    gaussian_halfballs* gb = 
      new gaussian_halfballs(centers.size(), dim);
    
    for(unsigned int i=0;i<centers.size();i++)
      for(unsigned int j=0;j<centers[i].size();j++)
	gb->centers[i][j] = centers[i][j];
    
    gb->variance = variance;
    gb->dim = dim;
    
    return gb;
  }
  
  blas_real<float> goodvalue() const{
    return calculate(centers[0]);
  }
  
  
  void calculate(const whiteice::math::vertex< whiteice::math::blas_real<float> >& x, blas_real<float>& y) const{
    y = calculate(x);
    return;
  }
  
  // dummy functions
  
  bool hasGradient() const throw(){ return false; }
  whiteice::math::vertex< blas_real<float> > grad(whiteice::math::vertex< blas_real<float> >& x) const{ return x; }
  void grad(whiteice::math::vertex< blas_real<float> >& x, whiteice::math::vertex< blas_real<float> >& y) const{ return; }
      
  bool hasHessian() const throw(){ return false; }
  whiteice::math::matrix< blas_real<float> > hessian(whiteice::math::vertex< blas_real<float> >& x) const{ return x.outerproduct(); }
  void hessian(whiteice::math::vertex< blas_real<float> >& x, whiteice::math::matrix< blas_real<float> >& y) const{ return; }
  

private:
  std::vector< whiteice::math::vertex<blas_real<float> > > centers;
  unsigned int dim;
  blas_real<float> variance;
};



void test_optimization()
{
  try{
    std::cout << "OPTIMIZATION TESTS" << std::endl;
    
    whiteice::math::StochasticOptimizer<blas_real<float> >* so;
    
    // so = new whiteice::math::IHRSearch<blas_real<float> >();
    so = new whiteice::math::GradientDescent<blas_real<float> >();
    whiteice::optimized_function< blas_real<float> >* of;
    
    // 10 balls in 1000 dimensions
    // of = new gaussian_balls(10, 1000);
    of = new gaussian_halfballs(10, 1000);

    std::cout << "GOOD VALUE: " 
	      << ((gaussian_balls*)of)->goodvalue()
	      << std::endl;
    

    std::cout << "Optimizing in 10 seconds intervals.."
	      << std::endl;
    
    so->optimize(of, 10.0f);
    
    while(1){
      whiteice::math::vertex<blas_real<float> > v;
      
      std::cout << "BEST SOLUTION: "
		<< so->getSolution(v)
		<< std::endl;
      
      std::cout << v << std::endl;
      
      so->optimizeMore(10.0f);
    }
    
    
    delete so;
    
  }
  catch(std::exception& e){
    std::cout << "unexpected exception: " 
	      << e.what()
	      << std::endl;
  }
}



void test_permfact()
{
  try{
    whiteice::math::integer i, j, k;
    
    i = 1024*1024*1024;
    j = 1024*1024;
    k = 1024;
    
    std::cout << "i  = " << i << std::endl;
    std::cout << "j  = " << j << std::endl;
    std::cout << "k  = " << k << std::endl;
    
    std::cout << "j! = " << whiteice::math::factorial(j) << std::endl;
    
    //std::cout << "C(i,k) = " << whiteice::math::combinations(i,k) << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "unexpected exception: "
	      << e.what() << std::endl;
  }
  
}




class odeproblem1 : public odefunction< blas_real<double> >
{
public:
  odeproblem1(){ }
  ~odeproblem1(){ }
  
  // calculates value
  whiteice::math::vertex< blas_real<double> > calculate
  (const odeparam< blas_real<double> >& x) const PURE_FUNCTION {
    // simple problem
    // 
    // x'(t) = v
    // v'(t) = -k*x
    // x(0) = 0, v(0) = sqrt(k)
    // 
    // analytical solution is:
    // 
    // x = sin(sqrt(k)*t)
    // 
    // in matrix form:
    // z'(t) = A*z(t), where z = [x v]
    // A = [0 1; -k 0]
    
    blas_real<double> k = 2.0;
    
    whiteice::math::vertex< blas_real<double> > dz(2);
    dz[0] = x.y[1];
    dz[1] = -k*x.y[0];
    
    return dz;
  }
  
  // returns number of input dimensions
  unsigned int dimensions() const PURE_FUNCTION {
    return 2;
  }
  
  // calculates value of function
  whiteice::math::vertex< blas_real<double> > operator() (const odeparam< blas_real<double> >& x) const PURE_FUNCTION {
    return calculate(x);
  }
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  void calculate(const odeparam< blas_real<double> >& x,
		 whiteice::math::vertex< blas_real<double> >& y) const {
    y = calculate(x);
  }
  
  // creates copy of object
  whiteice::function< odeparam< blas_real<double> >, whiteice::math::vertex< blas_real<double> > >* clone() const {
    return new odeproblem1();
  }
  
private:
  
};

class odeproblem2 : public odefunction< blas_real<double> >
{
public:
  odeproblem2(){ }
  ~odeproblem2(){ }
  
  // calculates value
  whiteice::math::vertex< blas_real<double> > calculate
  (const odeparam< blas_real<double> >& x) const PURE_FUNCTION {
    // harder problem which 
    // requires adaptive step-length
    // velocity increases as a function of time so that
    // h must become smaller and smaller
    //
    // x'(t) = v
    // v'(t) = v/t - 4*k*t^2*x
    // t E [0,10]
    // boundary conditions: (t=v=0)
    // x(t=v) = sqrt(k)*v^2 = 0
    // v(t=v) = 2*sqrt(k)*v = 0
    // 
    // solution is x(t) = sin(sqrt(k)*t^2)
    // 
    // in matrix form:
    // z'(t) = A(t)*z(t), where z = [x v]
    // A(t) = [0 1; -4*k*t^2 1/t]
    
    blas_real<double> k4 = 8.0; // k = 2.0
    
    whiteice::math::vertex< blas_real<double> > dz(2);
    dz[0] = x.y[1];
    dz[1] = x.y[1]/x.t - k4*x.t*x.t*x.y[0];
    
    return dz;
  }
  

  
  // returns number of input dimensions
  unsigned int dimensions() const PURE_FUNCTION {
    return 2;
  }
  
  // calculates value of function
  whiteice::math::vertex< blas_real<double> > operator() (const odeparam< blas_real<double> >& x) const PURE_FUNCTION {
    return calculate(x);
  }
  
  // calculates value 
  // (optimized version, this is faster because output value isn't copied)
  void calculate(const odeparam< blas_real<double> >& x,
		 whiteice::math::vertex< blas_real<double> >& y) const {
    y = calculate(x);
  }
  
  // creates copy of object
  whiteice::function< odeparam< blas_real<double> >, whiteice::math::vertex< blas_real<double> > >* clone() const {
    return new odeproblem2();
  }
  
private:
  
};



void test_rungekutta()
{
  try{
    std::cout << "4TH ORDER RUNGE-KUTTA WITH ADAPTIVE STEP LENGTH TESTS"
	      << std::endl;
    
    // tests Runge-Kutta integrator with two test problems
    RungeKutta< blas_real<double> > rk;
    
    // simple problem
    // 
    // x'(t) = v
    // v'(t) = -k*x
    // x(0) = 0, v(0) = sqrt(k)
    // 
    // analytical solution is:
    // 
    // x = sin(sqrt(k)*t)
    // 
    // in matrix form:
    // z'(t) = A*z(t), where z = [x v]
    // A = [0 1; -k 0]
    
    odeproblem1 ode1;
    rk.setFunction(&ode1);
    
    vertex< blas_real<double> > y0(2);
    blas_real<double> k = 2.0;
    
    std::vector< vertex< blas_real<double> > > points;
    std::vector< blas_real<double> > times;
    y0[0] = 0.0;
    y0[1] = whiteice::math::sqrt(k);
    
    std::cout << "ODE Problem 1 calculations started" << std::endl;
    
    rk.calculate(0.0, 10.0, y0, points, times);
    
    std::cout << points.size() << " data points." << std::endl;
    
    blas_real<double> error = 0.0;
    for(unsigned int i=0;i<points.size();i++){
      y0[0] = sin(whiteice::math::sqrt(k)*times[i]);
      
      error += abs(y0[0] - points[i][0])/blas_real<double>((double)points.size());
    }
    
    std::cout << "average error: " << error << std::endl << std::endl;
    
    // harder problem which 
    // requires adaptive step-length
    // velocity increases as a function of time so that
    // h must become smaller and smaller
    //
    // x'(t) = v
    // v'(t) = v/t - 4*k*t^2*x
    // t E [0,10]
    // boundary conditions: (t=v=0)
    // x(t=v) = sqrt(k)*v^2 = 0
    // v(t=v) = 2*sqrt(k)*v = 0
    // 
    // solution is x(t) = sin(sqrt(k)*t^2)
    // 
    // in matrix form:
    // z'(t) = A(t)*z(t), where z = [x v]
 
    // A(t) = [0 1; -4*k*t^2 1/t]
    odeproblem2 ode2;
    rk.setFunction(&ode2);
    
    points.clear();
    times.clear();
    
    blas_real<double> v = 0.01;
    
    y0[0] = whiteice::math::sqrt(k)*v*v;
    y0[1] = 2.0*whiteice::math::sqrt(k)*v;
    
    std::cout << "ODE Problem 2 calculations started" << std::endl;
    
    rk.calculate(v, 10.0, y0, points, times);
    
    std::cout << points.size() << " data points." << std::endl;
    
    error = 0.0;
    for(unsigned int i=0;i<points.size();i++){
      y0[0] = sin(whiteice::math::sqrt(k)*times[i]*times[i]);
      
      error += abs(y0[0] - points[i][0])/blas_real<double>((double)points.size());
    }
    
    std::cout << "average error: " << error << std::endl << std::endl;
    
  }
  catch(std::exception& e){
  std::cout << "unexpected exception: "
    << e.what() << std::endl;
  }
}



void* allocate_function(size_t allocSize){ 
  return malloc(allocSize);
}

void* reallocate_function(void* ptr, size_t oldSize, size_t newSize){ 
  return realloc(ptr, newSize);
}

void deallocate_function(void* ptr, size_t size){
  free(ptr);
}


void test_realnumber()
{
  std::cout << "ARBITRARY PRECISION NUMBER TESTS" << std::endl;
  
  mp_set_memory_functions(allocate_function,
			  reallocate_function,
			  deallocate_function);
  
  
  // basic arithmetic tests
  try{
    std::vector<realnumber> a(100);
    std::vector<double> b(a.size());
    bool ok = true;
    
    for(unsigned int i=0;i<a.size();i++){
      a[i] = ((double)rand())/((double)RAND_MAX);
      b[i] = a[i].getDouble();
    }
    
    for(unsigned int i=0;i<a.size();i++){
      if(abs(a[i] - b[i]) > 0.00001){
	std::cout << "assignment error: "  << i << std::endl;
	std::cout << "a[i] = " << a[i] << std::endl;
	std::cout << "b[i] = " << b[i] << std::endl;
	return;
      }
	
    }
      
    
    for(unsigned int n=0;n<1000;n++){
      unsigned int op = rand() % 4;
      unsigned int i = rand() % a.size();
      unsigned int j = rand() % a.size();
      unsigned int k = rand() % a.size();
      
      if(op == 3)
	if(a[j] < 0.01 || b[j] < 0.01)
	  continue;
      
      switch(op){
      case 0: // add
	a[k] = a[i] + a[j];
	b[k] = b[i] + b[j];
	break;
      case 1: // sub
	a[k] = a[i] - a[j];
	b[k] = b[i] - b[j];
	break;
      case 2: // mul
	a[k] = a[i] * a[j];
	b[k] = b[i] * b[j];
	break;
      case 3: // div
	a[k] = a[i] / a[j];
	b[k] = b[i] / b[j];
	break;
      default:
	break;
      };

      if(abs(a[k] - b[k]) > 0.00001*whiteice::math::abs(a[k])){
	std::cout << "MAYBE ERROR OR DOUBLE OUT OF RANGE:" << std::endl;
	std::cout << "DELTA: " << (a[k] - b[k]) << std::endl;
	std::cout << "OPERATOR: " << op << " ERROR" << std::endl;
	std::cout << a[i] << " == " << b[i] << std::endl;
	std::cout << a[j] << " == " << b[j] << std::endl;
	std::cout << a[k] << " == " << b[k] << std::endl;
	ok = false;
      }
      
    }
    
    if(ok)
      std::cout << "BASIC ARITHMETIC TESTS: OK" << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "unexpected exception: "
	      << e.what() << std::endl;
    return;
  }
  
  
  // function tests
  try{
    std::vector<realnumber> a(100);
    std::vector<double> b(a.size());
    bool ok = true;
    
    for(unsigned int i=0;i<a.size();i++){
      a[i] = 10.0*(((double)rand())/((double)RAND_MAX) - 0.5);
      b[i] = a[i].getDouble();
    }
    
    for(unsigned int i=0;i<a.size();i++){
      if(abs(a[i] - b[i]) > 0.00001){
	std::cout << "assignment error: "  << i << std::endl;
	std::cout << "a[i] = " << a[i] << std::endl;
	std::cout << "b[i] = " << b[i] << std::endl;
	return;
      }
	
    }
      
    
    for(unsigned int n=0;n<1000;n++){
      unsigned int op = rand() % 14;
      unsigned int i = n % a.size();
      unsigned int j = (n+50) % a.size();
      unsigned int k = (n+99) % a.size();
      
      if(a[i] > 10000.0 || a[i] < 10000.0 || isnan(b[i]) || isinf(b[i])){
	a[i] = 10.0*(((double)rand())/((double)RAND_MAX) - 0.5);
	b[i] = a[i].getDouble();
      }
      
      if(a[j] > 100.0 || a[j] < 100.0 || isnan(b[j]) || isinf(b[j])){
	a[j] = 10.0*(((double)rand())/((double)RAND_MAX) - 0.5);
	b[j] = a[j].getDouble();
      }
      
      std::cout << "OPERATOR: " << op << std::endl;
      
      switch(op){
      case 0: // exp()
	a[k] = whiteice::math::exp(a[i]);
	b[k] = whiteice::math::exp(b[i]);
	break;
      case 1: // log()
	if(b[i] > 0.0){
	  a[k] = whiteice::math::log(a[i]);
	  b[k] = whiteice::math::log(b[i]);
	}
	break;
      case 2: // sin()
	std::cout << "SIN(X) TEST DO NOT WORK" << std::endl;
	/*
	  a[k] = whiteice::math::sin(a[i]);
	  b[k] = whiteice::math::sin(b[i]);
	*/
	break;
      case 3: // cos()
	 std::cout << "COS(X) TEST DO NOT WORK" << std::endl;
	 /*
	a[k] = whiteice::math::cos(a[i]);
	b[k] = whiteice::math::cos(b[i]);
	 */
	break;
      case 4: // sqrt()
	if(b[i] > 0.0){
	  a[k] = whiteice::math::sqrt(a[i]);
	  b[k] = whiteice::math::sqrt(b[i]);
	  std::cout << "IN: " << a[i] << " OUT: " << a[k] << std::endl;
	}	
	break;
      case 5: // pow()
	if(b[i] > 0.0){
	  a[k] = whiteice::math::pow(a[i], a[j]);
	  b[k] = whiteice::math::pow(b[i], b[j]);
	}

	break;
      case 6: // abs()
	a[k] = whiteice::math::abs(a[i]);
	b[k] = whiteice::math::abs(b[i]);
	break;
      case 7: // arg()
	a[k] = whiteice::math::arg(a[i]);
	b[k] = whiteice::math::arg(b[i]);
	break;
      case 8: // conj()
	a[k] = whiteice::math::conj(a[i]);
	b[k] = whiteice::math::conj(b[i]);
	break;
      case 9: // real()
	a[k] = whiteice::math::real(a[i]);
	b[k] = whiteice::math::real(b[i]);
	break;
      case 10: // imag()
	a[k] = whiteice::math::imag(a[i]);
	b[k] = whiteice::math::imag(b[i]);
	break;
      case 11: // ceil()
	a[k] = whiteice::math::ceil(a[i]);
	b[k] = whiteice::math::ceil(b[i]);
	break;
      case 12: // floor()
	a[k] = whiteice::math::floor(a[i]);
	b[k] = whiteice::math::floor(b[i]);
	break;
      case 13: // trunc()
	a[k] = whiteice::math::trunc(a[i]);
	b[k] = whiteice::math::trunc(b[i]);
	break;
      default:
	break;
      };

      if(!isinf(b[k]) && !isnan(b[k])){
	 if(abs(a[k] - b[k]) > 0.00001*whiteice::math::abs(a[k])){
	   std::cout << "MAYBE ERROR OR DOUBLE OUT OF RANGE:" << std::endl;
	   std::cout << "DELTA: " << (a[k] - b[k]) << std::endl;
	   std::cout << "OPERATOR: " << op << " ERROR" << std::endl;
	   std::cout << a[i] << " == " << b[i] << std::endl;
	   std::cout << a[j] << " == " << b[j] << std::endl;
	   std::cout << a[k] << " == " << b[k] << std::endl;
	   ok = false;
	   
	   return; // stop
	 }
      }
      
    }
    
    
    if(ok)
      std::cout << "FUNCTION TESTS: OK" << std::endl;
    
    
  }
  catch(std::exception& e){
    std::cout << "unexpected exception: "
	      << e.what() << std::endl;
    return;
  }
  
  
  try{
    realnumber p = whiteice::math::pi(1024);
    
    std::cout << "pi(1024) = " << p << std::endl;
    
  }
  catch(std::exception& e){
    std::cout << "unexpected exception: "
	      << e.what() << std::endl;
    return;
  }
  
}
