/*
 * RNG.cpp
 *
 *  Created on: 28.6.2015
 *      Author: Tomas Ukkonen
 */

#include "RNG.h"

#ifndef RNG_CPP
#define RNG_CPP

#include <string>
#include <math.h>
#include <time.h>
#include <stdlib.h>

// Ziggurat code is almost directly from the paper of George Marsaglia:
// "The Ziggurat Method for Generating Random Variableas" (2000)

namespace whiteice {

template <typename T>
RNG<T>::RNG(const bool usehw)
{
  // uses CPUID to check for RDRAND instruction
  bool has_rdrand = false;
  {
    unsigned int regs[4];
    
    // get vendor
    char vendor[12];
    cpuid(0, 0, regs);
    ((unsigned int *)vendor)[0] = regs[1]; // EBX
    ((unsigned int *)vendor)[1] = regs[3]; // EDX
    ((unsigned int *)vendor)[2] = regs[2]; // ECX
    std::string cpuvendor = std::string(vendor, 12);
    
    // printf("CPUVENDOR: %s\n", cpuvendor.c_str());
    
    if(cpuvendor == "GenuineIntel"){
      cpuid(1, 0, regs);
      if((regs[2] & 0x40000000) == 0x40000000)
	has_rdrand = true;
    }
    else if(cpuvendor == "AuthenticAMD"){
      cpuid(1, 0, regs);
      if((regs[2] & 0x40000000) == 0x40000000) // 30th bit ECX is 1
	has_rdrand = true;
    }
  }
  
  // setups function pointers to be used for rng
  if(has_rdrand && usehw){
    rdrand32 = &whiteice::RNG<T>::_rdrand32;
    rdrand64 = &whiteice::RNG<T>::_rdrand64;
  }
  else{
    srand(time(0));
    rdrand32 = &whiteice::RNG<T>::_rand32; // uses C rand()
    rdrand64 = &whiteice::RNG<T>::_rand64; // uses C rand()
  }
  
  // calculates ziggurat tables for normal and exponential distribution
  calculate_ziggurat_tables();
}

template <typename T>
unsigned int RNG<T>::rand() const{ return (this->*rdrand32)(); }

template <typename T>
unsigned long long RNG<T>::rand64() const{ return (this->*rdrand64)(); } // 64bit


template <typename T>
T RNG<T>::uniform() const // [0,1]
{
  // const double MAX = (double)((unsigned long long)(-1LL)); // 2**64 - 1
  // return T(rdrand64()/MAX);
  return T(unid());
}


template <typename T>
void RNG<T>::uniform(math::vertex<T>& u) const{
  // const double MAX = (double)((unsigned long long)(-1LL)); // 2**64 - 1
  
  for(unsigned int i=0;i<u.size();i++){
    // u[i] = T(rdrand64()/MAX);
    u[i] = T(unid());
  }
}


template <typename T>
T RNG<T>::normal() const{
  return T(rnor());
}
  

template <typename T>
void RNG<T>::normal(math::vertex<T>& n) const
{
  for(unsigned int i=0;i<n.size();i++)
    n[i] = T(rnor());
}


template <typename T>
T RNG<T>::exp() const
{
  const float e = rexp();
  
  return T(e >= 0.0f ? e : (-e));
}


template <typename T>
void RNG<T>::exp(math::vertex<T>& ev) const
{
  for(unsigned int i=0;i<ev.size();i++){
    const float e = rexp();
    ev[i] = T(e >= 0.0f ? e : (-e));
  }
}



/////////////////////////////////////////////////////////////////////////////////////////////7

template <typename T>
float RNG<T>::rnor() const
{
        int hz = (signed)(this->*rdrand32)();
	unsigned int iz = hz & 127;

	if((unsigned)abs(hz) < kn[iz]){
		return hz*wn[iz];
	}
	else{
		// nfix()
		const float r = 3.442620f;
		float x, y;

		for(;;){
			x=hz*wn[iz];

			if(iz==0){
				do{ x=-math::log(unid())*0.2904764; y=-math::log(unid()); } while(y+y<x*x);

				return (hz>0) ? r+x : -r-x;
			}

			if( fn[iz]+unid()*(fn[iz-1]-fn[iz]) < math::exp(-.5*x*x) )
				return x;

			hz=(signed)(this->*rdrand32)();
			iz=hz&127;

			if((unsigned int)math::abs(hz)<kn[iz])
				return (hz*wn[iz]);
		}
	}
}


template <typename T>
float RNG<T>::rexp() const
{
	int jz = (this->*rdrand32)();
	unsigned int iz = jz & 255;

	if( jz <(signed)ke[iz]){ // added (signed)
		return jz*we[iz];
	}
	else{
		// efix()
		float x;

		for(;;){
			if(iz==0)
				return (7.69711-math::log(unid()));

			x=jz*we[iz];
			if( fe[iz]+unid()*(fe[iz-1]-fe[iz]) < math::exp(-x) )
				return (x);

			jz=(this->*rdrand32)();
			iz=(jz&255);

			if(jz<(signed)ke[iz]) // added (signed)
				return (jz*we[iz]);
		}
	}
}


template <typename T>
void RNG<T>::calculate_ziggurat_tables()
{
	const double m1 = 2147483648.0, m2 = 4294967296.;
	double dn=3.442619855899,tn=dn,vn=9.91256303526217e-3, q;
	double de=7.697117470131487, te=de, ve=3.949659822581572e-3;
	int i;

	// jsr=jsrseed;

	/* Tables for RNOR (normal distribution): */
	q=vn/math::exp(-.5*dn*dn);
	kn[0]=(dn/q)*m1;
	kn[1]=0;
	wn[0]=q/m1;
	wn[127]=dn/m1;
	fn[0]=1.;
	fn[127]=math::exp(-.5*dn*dn);

	for(i=126;i>=1;i--) {
		dn=math::sqrt(-2.*math::log(vn/dn+math::exp(-.5*dn*dn)));
		kn[i+1]=(dn/tn)*m1; tn=dn;
		fn[i]=math::exp(-.5*dn*dn); wn[i]=dn/m1;
	}

	/* Tables for REXP (exponential distribution) */
	q = ve/math::exp(-de);
	ke[0]=(de/q)*m2;
	ke[1]=0;
	we[0]=q/m2;
	we[255]=de/m2;
	fe[0]=1.;
	fe[255]=math::exp(-de);

	for(i=254;i>=1;i--) {
		de=-math::log(ve/de+math::exp(-de));
		ke[i+1]= (de/te)*m2; te=de;
		fe[i]=math::exp(-de); we[i]=de/m2;
	}
	
}



// floating point uniform distribution [for ziggurat method]
template <typename T>
float RNG<T>::unif() const
{
  return (0.5 + ((signed)((this->*rdrand32)())) * .2328306e-9);
}


template <typename T>
double RNG<T>::unid() const
{
  return (0.5 + ((signed)((this->*rdrand32)())) * .2328306e-9);
}


template <typename T>
unsigned int RNG<T>::_rdrand32() const
{
	unsigned int lvalue;
	unsigned char ok = 0;

	while(!ok)
		asm volatile ("rdrand %0; setc %1" : "=r" (lvalue), "=qm" (ok));

	return lvalue;
}


template <typename T>
unsigned long long RNG<T>::_rdrand64() const
{
  unsigned long long lvalue;
  unsigned char ok = 0;
  
  while(!ok)
    asm volatile ("rdrand %0; setc %1" : "=r" (lvalue), "=qm" (ok));
  
  return lvalue;
}

template <typename T>
unsigned int RNG<T>::_rand32() const
{
  unsigned int r1 = (unsigned int)::rand();
  unsigned int r2 = (unsigned int)::rand();
  unsigned int r = (r1 << 16) ^ (r2);

  return r;
}

template <typename T>
unsigned long long RNG<T>::_rand64() const
{
  unsigned long long r1 = (unsigned long long)::rand();
  unsigned long long r2 = (unsigned long long)::rand();
  unsigned long long r3 = (unsigned long long)::rand();
  unsigned long long r4 = (unsigned long long)::rand();

  return ((r1) ^ (r2 << 16) ^ (r3 << 32) ^ (r4 << 48));
}

template <typename T>
void RNG<T>::cpuid(unsigned int leaf, unsigned int subleaf, unsigned int regs[4])
{
  asm volatile("cpuid" : "=a" (regs[0]), "=b" (regs[1]), "=c" (regs[2]), "=d" (regs[3])
	       : "a" (leaf), "c" (subleaf));
}


} /* namespace whiteice */

#endif

