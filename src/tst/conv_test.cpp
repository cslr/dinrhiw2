/*
 * testcases for
 * order preserving isomorphic conversion between
 * ieee floating point numbers and integers
 */


#include <iostream>
#include <string.h>
#include <exception>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <new>

#include "fast_radix.h"
#include "conversion.h"

using namespace whiteice;


void conversion_tests();
void floating_point_radix_tests();

extern "C" {
  int float_compare(const void* f1, const void* f2);
};

// timing
static double get_time();



int main()
{
  srand(time(0));
  
  std::cout << "ORDER-ISOMORPHIC TESTS" << std::endl;
  conversion_tests();  
  std::cout << std::endl;
  
  std::cout << "FLOATING POINT RADIX SORT TESTS" << std::endl;
  floating_point_radix_tests();
  std::cout << std::endl;
  
  return 0;
}



void floating_point_radix_tests()
{
  // tests only with floats
  
  try{
    // -> aprox 500 000 seems to be value where converting is worth it
    const unsigned int SIZE = 100000000;
    
    float* table  = new float[SIZE];
    float* table2 = new float[SIZE];
    
    float scale = 100.0 * (((double)rand())/((double)RAND_MAX));
    
    std::cout << "Test material size: " << SIZE << std::endl;
    std::cout << "Scaling: " << scale << std::endl;
    
    for(unsigned int i=0;i<SIZE;i++){
      table[i] = ((((double)rand())/((double)RAND_MAX)) - 0.5);
      table[i] *= scale;
    }
    
    memcpy(table2, table, sizeof(float)*SIZE);
    
    //////////////////////////////////////////////////
    // sorts floating point numbers with radix sort
    // this is only O(n), although conversions inbetween
    // causes rather big coefficient for n term.
    
    // converts to comparision-isomorphic integer format: O(n)
    
    double rt1, rt2;
    
    std::cout << "RADIX STARTED" << std::endl;
    std::cout.flush();
    rt1 = get_time();
    
    for(unsigned int i=0;i<SIZE;i++){
      ((unsigned int*)table)[i] =
	ieee754spf2uint32(table[i]);
    }
    
    fast_radix<unsigned int>* rs =
      new fast_radix<unsigned int>((unsigned int*)table, SIZE);
    
    if( rs->sort() == false ){ // radix sort: O(n)
      std::cout << "ERROR: RADIX SORT FAILED" << std::endl;
    }
    
    // converts back to floats: O(n)
    for(unsigned int i=0;i<SIZE;i++){
      table[i] = uint322ieee754spf(((unsigned int*)table)[i]);
    }
    
    rt2 = get_time();
    std::cout << "RADIX ENDED" << std::endl;
    std::cout.flush();
    
    
    //////////////////////////////////////////////////
    // sorts floating point numbers with quicksort
    // this is only O(n*log n)
    
    double qt1, qt2;
    
    std::cout << "QSORT STARTED" << std::endl;
    std::cout.flush();
    qt1 = get_time();
    
    qsort(table2, SIZE, sizeof(float), float_compare);
    
    qt2 = get_time();
    std::cout << "QSORT ENDED" << std::endl;
    std::cout.flush();
    
    std::cout << std::endl;
    
    std::cout << "EXEC. TIME" << std::endl;
    std::cout << "float radix sort " << (rt2 - rt1) << " time units" << std::endl;
    std::cout << "float qsort sort " << (qt2 - qt1) << " time units" << std::endl;

    //////////////////////////////////////////////////
    // compares sorting results
    
    for(unsigned int i=0;i<SIZE;i++){
      if(table[i] != table2[i]){
	std::cout << "ERROR: different sorting order: quicksort/radix sort" 
		  << std::endl;
      }
    }
    
    
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception thrown" << std::endl;
  }  
}



void conversion_tests()
{
  ////////////////////////////////////////////////////////////
  // SINGLE PRECISION TESTS
  
  // note: negative nan-generation doesn't work:
  // -> bug in "-nan" code is bug in testing code.
  
  std::cout << "SINGLE PRECISION" << std::endl;
  
  // test with zero, infinities and NaNs
  try
  {
    float zero, negzero; // this is postive zero(?)
    float posinf,neginf;    
    float posnan,negnan;
    
    unsigned int i;
    i = 0;
    memcpy(&zero, &i, sizeof(i));
    i = 0x80000000;
    memcpy(&negzero, &i, sizeof(i));
    
    posinf = 1.0;
    neginf = -1.0;
    
    posinf /= zero;
    neginf /= zero;
    
    posnan = zero*posinf;
    negnan = zero*neginf;
    
    // printing tests
    std::cout << "These should be correct" << std::endl;
    std::cout << "+zero = " << zero  << " -zero = " << negzero << std::endl;
    std::cout << "+inf  = " << posinf << " -inf  = " << neginf << std::endl;
    std::cout << "+nan  = " << posnan << " -nan  = " << negnan << std::endl;
    
    unsigned int izero, inegzero, iposinf, ineginf, iposnan, inegnan;
    unsigned int izero2, inegzero2, iposinf2, ineginf2, iposnan2, inegnan2;
    float zero2, negzero2, posinf2, neginf2, posnan2, negnan2;
    
    memcpy(&izero,    &zero,    sizeof(zero));
    memcpy(&inegzero, &negzero, sizeof(negzero));
    memcpy(&iposinf,  &posinf,  sizeof(posinf));
    memcpy(&ineginf,  &neginf,  sizeof(neginf));
    memcpy(&iposnan,  &posnan,  sizeof(posnan));
    memcpy(&inegnan,  &negnan,  sizeof(negnan));
    
    izero2    = ieee754spf2uint32(zero);
    inegzero2 = ieee754spf2uint32(negzero);
    iposinf2  = ieee754spf2uint32(posinf);
    ineginf2  = ieee754spf2uint32(neginf);
    iposnan2  = ieee754spf2uint32(posnan);
    inegnan2  = ieee754spf2uint32(negnan);

    zero2    = uint322ieee754spf(izero2);
    negzero2 = uint322ieee754spf(inegzero2);
    posinf2  = uint322ieee754spf(iposinf2);
    neginf2  = uint322ieee754spf(ineginf2);
    posnan2  = uint322ieee754spf(iposnan2);
    negnan2  = uint322ieee754spf(inegnan2);
    
    
    std::cout << "Results after map and inverse mapping:" << std::endl;
    std::cout << "+zero = " << zero2  << " -zero = " << negzero2 << std::endl;
    std::cout << "+inf  = " << posinf2 << " -inf  = " << neginf2 << std::endl;
    std::cout << "+nan  = " << posnan2 << " -nan  = " << negnan2 << std::endl;    
    
    // extra printing tests
    std::cout << "Original floating point numbers in hex:" << std::endl;
    std::cout << std::hex;
    std::cout << "+zero = 0x" << izero  << " -zero = 0x" << inegzero << std::endl;
    std::cout << "+inf  = 0x" << iposinf << " -inf  = 0x" << ineginf << std::endl;
    std::cout << "+nan  = 0x" << iposnan << " -nan  = 0x" << inegnan << std::endl;    
    
    std::cout << "Numbers in isomorphic integer format:" << std::endl;
    std::cout << std::hex;
    std::cout << "+zero = 0x" << izero2  << " -zero = 0x" << inegzero2 << std::endl;
    std::cout << "+inf  = 0x" << iposinf2 << " -inf  = 0x" << ineginf2 << std::endl;
    std::cout << "+nan  = 0x" << iposnan2 << " -nan  = 0x" << inegnan2 << std::endl;
    std::cout << std::dec;

    // checks isomorphic properties
    
    if(ineginf2 >= iposinf2)
      std::cout << "ERROR: ordering is incorrect with infinities" << std::endl;
    
    if(ineginf2 >= izero2)
      std::cout << "ERROR: bad ordering: -inf, +zero" << std::endl;

    if(iposinf2 <= izero2)
      std::cout << "ERROR: bad ordering: +inf, +zero" << std::endl;
    
    // there's difference between positive and negative zeros: 
    // this is a feature, not a bug (preserving sign can help 
    // in certain special cases)

    
    memcpy(&izero2,    &zero2,    sizeof(zero));
    memcpy(&inegzero2, &negzero2, sizeof(negzero));
    memcpy(&iposinf2,  &posinf2,  sizeof(posinf));
    memcpy(&ineginf2,  &neginf2,  sizeof(neginf));
    memcpy(&iposnan2,  &posnan2,  sizeof(posnan));
    memcpy(&inegnan2,  &negnan2,  sizeof(negnan));
    
    std::cout << "Mapped + Inverse mapped floating point numbers in hex:" << std::endl;
    std::cout << std::hex;
    std::cout << "+zero = 0x" << izero2  << " -zero = 0x" << inegzero2 << std::endl;
    std::cout << "+inf  = 0x" << iposinf2 << " -inf  = 0x" << ineginf2 << std::endl;
    std::cout << "+nan  = 0x" << iposnan2 << " -nan  = 0x" << inegnan2 << std::endl;    
    std::cout << std::dec;
    
    // check bijective properties of conversion
    
    if(zero != zero2)
      std::cout << "ERROR: positive zeros differ" << std::endl;
    
    if(negzero != negzero2)
      std::cout << "ERROR: negative zeros differ" << std::endl;
    
    if(zero != negzero2 || zero2 != negzero)
      std::cout << "ERROR: mixed sign zeros differ" << std::endl;
    
    if(posinf != posinf2)
      std::cout << "ERROR: positive infinities differ" << std::endl;
    
    if(neginf != neginf2)
      std::cout << "ERROR: negative infinities differ" << std::endl;
    
    if(posinf == neginf2 || posinf2 == neginf)
      std::cout << "ERROR: mixed sign infinities are equal" << std::endl;
    
    // cannot compare NaNs, according to standard NaNs
    // are never equal
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception thrown" << std::endl;
  }
  
  
  // test with random numbers
  try
  {
    float *farray;
    unsigned int* iarray;
    
    const unsigned int ARRAY_SIZE = 1000;
    
    farray  = new float[ARRAY_SIZE];
    iarray  = new unsigned int[ARRAY_SIZE];
    
    float scale = 100.0 * (((double)rand())/((double)RAND_MAX));
    
    std::cout << "Scaling: " << scale << std::endl;
    
    // adds special numbers into test
    farray[0] = 0.0;
    farray[1] = +1.0 / farray[0]; // +inf
    farray[2] = -1.0 / farray[0]; // -inf
    iarray[0] = ieee754spf2uint32(farray[0]);
    iarray[1] = ieee754spf2uint32(farray[1]);
    iarray[2] = ieee754spf2uint32(farray[2]);
    
    for(unsigned int i=3;i<ARRAY_SIZE;i++){
      farray[i] = ((((double)rand())/((double)RAND_MAX)) - 0.5);
      farray[i] *= scale;
      iarray[i] = ieee754spf2uint32(farray[i]);
    }   
    
    // tests order-preserving property of isomorpic mapping
    
    for(unsigned int i=0;i<ARRAY_SIZE;i++){
      for(unsigned int j=0;j<ARRAY_SIZE;j++){
	if(i != j){
	  if(farray[i] < farray[j]){
	    if(iarray[i] >= iarray[j]){
	      std::cout << "ERROR: order-preserving property doesn't hold  ";
	      std::cout << farray[i] << " < " << farray[j] << " but " 
			<< std::hex
			<< "0x" << iarray[i] << " >= 0x" << iarray[j] << std::endl;
	      
	      unsigned int a,b;
	      memcpy(&a, &(farray[i]),sizeof(float));
	      memcpy(&b, &(farray[j]),sizeof(float));
	      
	      std::cout << std::hex
			<<"hex(farray[i]) = 0x" << std::hex << a
			<< " and hex(farray[j]) = 0x" << b << std::endl << std::dec;
	    }
	  }
	  else if(farray[i] > farray[j]){
	    if(iarray[i] <= iarray[j]){
	      std::cout << "ERROR: order-preserving property doesn't hold  ";
	      std::cout << farray[i] << " > " << farray[j] << " but " 
			<< std::hex
			<< "0x" << iarray[i] << " <= 0x" << iarray[j] << std::endl;
	      
	      unsigned int a,b;
	      memcpy(&a, &(farray[i]),sizeof(float));
	      memcpy(&b, &(farray[j]),sizeof(float));
	      
	      std::cout << std::hex
			<<"hex(farray[i]) = 0x" << std::hex << a
			<< " and hex(farray[j]) = 0x" << b << std::endl << std::dec;
	    }
	  }
	}
      }
    }
    
    
    for(unsigned int i=0;i<ARRAY_SIZE;i++){
      
      float f = uint322ieee754spf(iarray[i]);
      
      if(f != farray[i]){
	std::cout << "ERROR: invmap(map(x)) != x. "
		  << "values (correct, processed): "
		  << farray[i] << " , " << f << std::endl;
	
	unsigned int a,b;
	memcpy(&a, &(farray[i]), sizeof(float));
	memcpy(&b, &f, sizeof(float));
	
	std::cout << "values in hex: " << std::hex
		  << "0x" << a << " , 0x" << b
		  << " converted value: 0x" << iarray[i] << std::endl << std::dec;

      }
    }
    
    delete[] farray;
    delete[] iarray;    
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception thrown" << std::endl;
  }
  
  
  // test with numbers having denormalized representation
  // (with only denormalized numbers and denormalized + 'normal' numbers)
  try
  {
    float *farray;
    unsigned int* iarray;
    
    const unsigned int ARRAY_SIZE = 1000;
    
    farray  = new float[ARRAY_SIZE];
    iarray  = new unsigned int[ARRAY_SIZE];
    
    
    float scale;
    
    // very small scaling to produce denormalized numbers
    {
      unsigned int iscale = 0x0000FFFF;
      memcpy(&scale, &iscale, sizeof(iscale));
    }        
    
    std::cout << "Scaling: " << scale << std::endl;
    
    // adds special numbers into test
    farray[0] = 0.0;
    farray[1] = +1.0 / farray[0]; // +inf
    farray[2] = -1.0 / farray[0]; // -inf
    iarray[0] = ieee754spf2uint32(farray[0]);
    iarray[1] = ieee754spf2uint32(farray[1]);
    iarray[2] = ieee754spf2uint32(farray[2]);
    
    for(unsigned int i=3;i<ARRAY_SIZE;i++){
      farray[i] = ((((double)rand())/((double)RAND_MAX)) - 0.5);
      farray[i] *= scale;
      iarray[i] = ieee754spf2uint32(farray[i]);
    }   
    
    // tests order-preserving property of isomorpic mapping
    
    for(unsigned int i=0;i<ARRAY_SIZE;i++){
      for(unsigned int j=0;j<ARRAY_SIZE;j++){
	if(i != j){
	  if(farray[i] < farray[j]){
	    if(iarray[i] >= iarray[j]){
	      std::cout << "ERROR: order-preserving property doesn't hold  ";
	      std::cout << farray[i] << " < " << farray[j] << " but " 
			<< std::hex
			<< "0x" << iarray[i] << " >= 0x" << iarray[j] << std::endl;
	      
	      unsigned int a,b;
	      memcpy(&a, &(farray[i]),sizeof(float));
	      memcpy(&b, &(farray[j]),sizeof(float));
	      
	      std::cout << std::hex
			<<"hex(farray[i]) = 0x" << std::hex << a
			<< " and hex(farray[j]) = 0x" << b << std::endl << std::dec;
	    }
	  }
	  else if(farray[i] > farray[j]){
	    if(iarray[i] <= iarray[j]){
	      std::cout << "ERROR: order-preserving property doesn't hold  ";
	      std::cout << farray[i] << " > " << farray[j] << " but " 
			<< std::hex
			<< "0x" << iarray[i] << " <= 0x" << iarray[j] << std::endl;
	      
	      unsigned int a,b;
	      memcpy(&a, &(farray[i]),sizeof(float));
	      memcpy(&b, &(farray[j]),sizeof(float));
	      
	      std::cout << std::hex
			<<"hex(farray[i]) = 0x" << std::hex << a
			<< " and hex(farray[j]) = 0x" << b << std::endl << std::dec;
	    }
	  }
	}
      }
    }
    
    
    for(unsigned int i=0;i<ARRAY_SIZE;i++){
      
      float f = uint322ieee754spf(iarray[i]);
      
      if(f != farray[i]){
	std::cout << "ERROR: invmap(map(x)) != x. "
		  << "values (correct, processed): "
		  << farray[i] << " , " << f << std::endl;
	
	unsigned int a,b;
	memcpy(&a, &(farray[i]), sizeof(float));
	memcpy(&b, &f, sizeof(float));
	
	std::cout << "values in hex: " << std::hex
		  << "0x" << a << " , 0x" << b
		  << " converted value: 0x" << iarray[i] << std::endl << std::dec;

      }
    }
    
    delete[] farray;
    delete[] iarray;
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception thrown" << std::endl;
  }
  
  
  ////////////////////////////////////////////////////////////
  // DOUBLE PRECISION TESTS
  
  std::cout << "DOUBLE PRECISION" << std::endl;
  
  
  // test with zero, infinities and NaNs
  try
  {
    double zero, negzero; // this is postive zero(?)
    double posinf,neginf;    
    double posnan,negnan;
    
    unsigned long long i;
    i = 0x0ULL;
    memcpy(&zero, &i, sizeof(i));
    i = 0x8000000000000000ULL;
    memcpy(&negzero, &i, sizeof(i));
    
    posinf = 1.0;
    neginf = -1.0;
    
    posinf /= zero;
    neginf /= zero;
    
    posnan = zero*posinf;
    negnan = zero*neginf;
    
    // printing tests
    std::cout << "These should be correct" << std::endl;
    std::cout << "+zero = " << zero  << " -zero = " << negzero << std::endl;
    std::cout << "+inf  = " << posinf << " -inf  = " << neginf << std::endl;
    std::cout << "+nan  = " << posnan << " -nan  = " << negnan << std::endl;
    
    unsigned long long izero, inegzero, iposinf, ineginf, iposnan, inegnan;
    unsigned long long izero2, inegzero2, iposinf2, ineginf2, iposnan2, inegnan2;
    double zero2, negzero2, posinf2, neginf2, posnan2, negnan2;
    
    memcpy(&izero,    &zero,    sizeof(zero));
    memcpy(&inegzero, &negzero, sizeof(negzero));
    memcpy(&iposinf,  &posinf,  sizeof(posinf));
    memcpy(&ineginf,  &neginf,  sizeof(neginf));
    memcpy(&iposnan,  &posnan,  sizeof(posnan));
    memcpy(&inegnan,  &negnan,  sizeof(negnan));
    
    izero2    = ieee754dpf2uint64(zero);
    inegzero2 = ieee754dpf2uint64(negzero);
    iposinf2  = ieee754dpf2uint64(posinf);
    ineginf2  = ieee754dpf2uint64(neginf);
    iposnan2  = ieee754dpf2uint64(posnan);
    inegnan2  = ieee754dpf2uint64(negnan);

    zero2    = uint642ieee754dpf(izero2);
    negzero2 = uint642ieee754dpf(inegzero2);
    posinf2  = uint642ieee754dpf(iposinf2);
    neginf2  = uint642ieee754dpf(ineginf2);
    posnan2  = uint642ieee754dpf(iposnan2);
    negnan2  = uint642ieee754dpf(inegnan2);
    
    
    std::cout << "Results after map and inverse mapping:" << std::endl;
    std::cout << "+zero = " << zero2  << " -zero = " << negzero2 << std::endl;
    std::cout << "+inf  = " << posinf2 << " -inf  = " << neginf2 << std::endl;
    std::cout << "+nan  = " << posnan2 << " -nan  = " << negnan2 << std::endl;    
    
    // extra printing tests
    std::cout << "Original floating point numbers in hex:" << std::endl;
    std::cout << std::hex;
    std::cout << "+zero = 0x" << izero  << " -zero = 0x" << inegzero << std::endl;
    std::cout << "+inf  = 0x" << iposinf << " -inf  = 0x" << ineginf << std::endl;
    std::cout << "+nan  = 0x" << iposnan << " -nan  = 0x" << inegnan << std::endl;    
    
    std::cout << "Numbers in isomorphic integer format:" << std::endl;
    std::cout << std::hex;
    std::cout << "+zero = 0x" << izero2  << " -zero = 0x" << inegzero2 << std::endl;
    std::cout << "+inf  = 0x" << iposinf2 << " -inf  = 0x" << ineginf2 << std::endl;
    std::cout << "+nan  = 0x" << iposnan2 << " -nan  = 0x" << inegnan2 << std::endl;
    std::cout << std::dec;

    // checks isomorphic properties
    
    if(ineginf2 >= iposinf2)
      std::cout << "ERROR: ordering is incorrect with infinities" << std::endl;
    
    if(ineginf2 >= izero2)
      std::cout << "ERROR: bad ordering: -inf, +zero" << std::endl;

    if(iposinf2 <= izero2)
      std::cout << "ERROR: bad ordering: +inf, +zero" << std::endl;
    
    // there's difference between positive and negative zeros: 
    // this is a feature, not a bug (preserving sign can help 
    // in certain special cases)

    
    memcpy(&izero2,    &zero2,    sizeof(zero));
    memcpy(&inegzero2, &negzero2, sizeof(negzero));
    memcpy(&iposinf2,  &posinf2,  sizeof(posinf));
    memcpy(&ineginf2,  &neginf2,  sizeof(neginf));
    memcpy(&iposnan2,  &posnan2,  sizeof(posnan));
    memcpy(&inegnan2,  &negnan2,  sizeof(negnan));
    
    std::cout << "Mapped + Inverse mapped floating point numbers in hex:" << std::endl;
    std::cout << std::hex;
    std::cout << "+zero = 0x" << izero2  << " -zero = 0x" << inegzero2 << std::endl;
    std::cout << "+inf  = 0x" << iposinf2 << " -inf  = 0x" << ineginf2 << std::endl;
    std::cout << "+nan  = 0x" << iposnan2 << " -nan  = 0x" << inegnan2 << std::endl;    
    std::cout << std::dec;
    
    // check bijective properties of conversion
    
    if(zero != zero2)
      std::cout << "ERROR: positive zeros differ" << std::endl;
    
    if(negzero != negzero2)
      std::cout << "ERROR: negative zeros differ" << std::endl;
    
    if(zero != negzero2 || zero2 != negzero)
      std::cout << "ERROR: mixed sign zeros differ" << std::endl;
    
    if(posinf != posinf2)
      std::cout << "ERROR: positive infinities differ" << std::endl;
    
    if(neginf != neginf2)
      std::cout << "ERROR: negative infinities differ" << std::endl;
    
    if(posinf == neginf2 || posinf2 == neginf)
      std::cout << "ERROR: mixed sign infinities are equal" << std::endl;
    
    // cannot compare NaNs, according to standard NaNs
    // are never equal
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception thrown" << std::endl;
  }
  
  
  // test with random numbers
  try
  {
    double *farray;
    unsigned long long* iarray;
    
    const unsigned long long ARRAY_SIZE = 1000;
    
    farray  = new double[ARRAY_SIZE];
    iarray  = new unsigned long long[ARRAY_SIZE];
    
    double scale = 100.0 * (((double)rand())/((double)RAND_MAX));
    
    std::cout << "Scaling: " << scale << std::endl;
    
    // adds special numbers into test
    farray[0] = 0.0;
    farray[1] = +1.0 / farray[0]; // +inf
    farray[2] = -1.0 / farray[0]; // -inf
    iarray[0] = ieee754dpf2uint64(farray[0]);
    iarray[1] = ieee754dpf2uint64(farray[1]);
    iarray[2] = ieee754dpf2uint64(farray[2]);
    
    for(unsigned long long i=3;i<ARRAY_SIZE;i++){
      farray[i] = ((((double)rand())/((double)RAND_MAX)) - 0.5);
      farray[i] *= scale;
      iarray[i] = ieee754dpf2uint64(farray[i]);
    }
    
    // tests order-preserving property of isomorpic mapping
    
    for(unsigned long long i=0;i<ARRAY_SIZE;i++){
      for(unsigned long long j=0;j<ARRAY_SIZE;j++){
	if(i != j){
	  if(farray[i] < farray[j]){
	    if(iarray[i] >= iarray[j]){
	      std::cout << "ERROR: order-preserving property doesn't hold  ";
	      std::cout << farray[i] << " < " << farray[j] << " but " 
			<< std::hex
			<< "0x" << iarray[i] << " >= 0x" << iarray[j] << std::endl;
	      
	      unsigned long long a,b;
	      memcpy(&a, &(farray[i]),sizeof(double));
	      memcpy(&b, &(farray[j]),sizeof(double));
	      
	      std::cout << std::hex
			<<"hex(farray[i]) = 0x" << std::hex << a
			<< " and hex(farray[j]) = 0x" << b << std::endl << std::dec;
	    }
	  }
	  else if(farray[i] > farray[j]){
	    if(iarray[i] <= iarray[j]){
	      std::cout << "ERROR: order-preserving property doesn't hold  ";
	      std::cout << farray[i] << " > " << farray[j] << " but " 
			<< std::hex
			<< "0x" << iarray[i] << " <= 0x" << iarray[j] << std::endl;
	      
	      unsigned long long a,b;
	      memcpy(&a, &(farray[i]),sizeof(double));
	      memcpy(&b, &(farray[j]),sizeof(double));
	      
	      std::cout << std::hex
			<<"hex(farray[i]) = 0x" << std::hex << a
			<< " and hex(farray[j]) = 0x" << b << std::endl << std::dec;
	    }
	  }
	}
      }
    }
    
    
    for(unsigned long long i=0;i<ARRAY_SIZE;i++){
      
      double f = uint642ieee754dpf(iarray[i]);
      
      if(f != farray[i]){
	std::cout << "ERROR: invmap(map(x)) != x. "
		  << "values (correct, processed): "
		  << farray[i] << " , " << f << std::endl;
	
	unsigned long long a,b;
	memcpy(&a, &(farray[i]), sizeof(double));
	memcpy(&b, &f, sizeof(double));
	
	std::cout << "values in hex: " << std::hex
		  << "0x" << a << " , 0x" << b
		  << " converted value: 0x" << iarray[i] << std::endl << std::dec;

      }
    }
    
    delete[] farray;
    delete[] iarray;    
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception thrown" << std::endl;
  }
  
  
  // test with numbers having denormalized representation
  // (with only denormalized numbers and denormalized + 'normal' numbers)
  try
  {
    double *farray;
    unsigned long long* iarray;
    
    const unsigned long long ARRAY_SIZE = 1000;
    
    farray  = new double[ARRAY_SIZE];
    iarray  = new unsigned long long[ARRAY_SIZE];
    
    
    double scale;
    
    // very small scaling to produce denormalized numbers
    {
      unsigned long long iscale = 0x000000000000FFFFULL;
      memcpy(&scale, &iscale, sizeof(iscale));
    }        
    
    std::cout << "Scaling: " << scale << std::endl;
    
    // adds special numbers into test
    farray[0] = 0.0;
    farray[1] = +1.0 / farray[0]; // +inf
    farray[2] = -1.0 / farray[0]; // -inf
    iarray[0] = ieee754dpf2uint64(farray[0]);
    iarray[1] = ieee754dpf2uint64(farray[1]);
    iarray[2] = ieee754dpf2uint64(farray[2]);
    
    for(unsigned long long i=3;i<ARRAY_SIZE;i++){
      farray[i] = ((((double)rand())/((double)RAND_MAX)) - 0.5);
      farray[i] *= scale;
      iarray[i] = ieee754dpf2uint64(farray[i]);
    }   
    
    // tests order-preserving property of isomorpic mapping
    
    for(unsigned long long i=0;i<ARRAY_SIZE;i++){
      for(unsigned long long j=0;j<ARRAY_SIZE;j++){
	if(i != j){
	  if(farray[i] < farray[j]){
	    if(iarray[i] >= iarray[j]){
	      std::cout << "ERROR: order-preserving property doesn't hold  ";
	      std::cout << farray[i] << " < " << farray[j] << " but " 
			<< std::hex
			<< "0x" << iarray[i] << " >= 0x" << iarray[j] << std::endl;
	      
	      unsigned long long a,b;
	      memcpy(&a, &(farray[i]),sizeof(double));
	      memcpy(&b, &(farray[j]),sizeof(double));
	      
	      std::cout << std::hex
			<<"hex(farray[i]) = 0x" << std::hex << a
			<< " and hex(farray[j]) = 0x" << b << std::endl << std::dec;
	    }
	  }
	  else if(farray[i] > farray[j]){
	    if(iarray[i] <= iarray[j]){
	      std::cout << "ERROR: order-preserving property doesn't hold  ";
	      std::cout << farray[i] << " > " << farray[j] << " but " 
			<< std::hex
			<< "0x" << iarray[i] << " <= 0x" << iarray[j] << std::endl;
	      
	      unsigned long long a,b;
	      memcpy(&a, &(farray[i]),sizeof(double));
	      memcpy(&b, &(farray[j]),sizeof(double));
	      
	      std::cout << std::hex
			<<"hex(farray[i]) = 0x" << std::hex << a
			<< " and hex(farray[j]) = 0x" << b << std::endl << std::dec;
	    }
	  }
	}
      }
    }
    
    
    for(unsigned long long i=0;i<ARRAY_SIZE;i++){
      
      double f = uint642ieee754dpf(iarray[i]);
      
      if(f != farray[i]){
	std::cout << "ERROR: invmap(map(x)) != x. "
		  << "values (correct, processed): "
		  << farray[i] << " , " << f << std::endl;
	
	unsigned long long a,b;
	memcpy(&a, &(farray[i]), sizeof(double));
	memcpy(&b, &f, sizeof(double));
	
	std::cout << "values in hex: " << std::hex
		  << "0x" << a << " , 0x" << b
		  << " converted value: 0x" << iarray[i] << std::endl << std::dec;

      }
    }
    
    delete[] farray;
    delete[] iarray;
  }
  catch(std::exception& e){
    std::cout << "ERROR: unexpected exception thrown" << std::endl;
  }
  
}



extern "C" {
  
  int float_compare(const void* f1, const void* f2)
  {
    
    if( *((float*)f1) < *((float*)f2) ){
      return -1;
    }
    else if( *((float*)f1) > *((float*)f2) ){
      return 1;
    }
    else{
      return 0;
    }
    
  }
};



/*
 * calculates process time
 */
static double get_time()
{  
  struct tms t1;
  
  if(times(&t1) == -1) return -1.0;

  double t = ( (double)t1.tms_utime + ((double)t1.tms_stime)*0.000001 );
  return t;
}
