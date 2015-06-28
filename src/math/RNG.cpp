/*
 * RNG.cpp
 *
 *  Created on: 28.6.2015
 *      Author: Tomas
 */

#ifndef RNG_CPP
#define RNG_CPP

#include "RNG.h"
#include <math.h>

// Ziggurat code is almost directly from the paper of George Marsaglia:
// "The Ziggurat Method for Generating Random Variableas" (2000)

namespace whiteice {

// throws runtime error if RDRAND is not supported
template <typename T>
RNG<T>::RNG() throw(std::runtime_error)
{
	// FIXME check for RDRAND instruction

	// calculates ziggurat tables for normal and exponential distribution
	calculate_ziggurat_tables();
}

template <typename T>
unsigned int RNG<T>::rand() const{ return rdrand32(); }

template <typename T>
unsigned long long RNG<T>::rand64() const{ return rdrand64(); } // 64bit


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
		u[i] = unid();
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
	int hz = rdrand32();
	unsigned int iz = hz & 127;

	if(abs(hz) < kn[iz]){
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

			hz=rdrand32();
			iz=hz&127;

			if(math::abs(hz)<kn[iz])
				return (hz*wn[iz]);
		}
	}
}


template <typename T>
float RNG<T>::rexp() const
{
	int jz = rdrand32();
	unsigned int iz = jz & 255;

	if( jz <ke[iz]){
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

			jz=rdrand32();
			iz=(jz&255);

			if(jz<ke[iz])
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
	// const float MAX = (float)((unsigned long long)(-1LL)); // 2**64 - 1
	// return (float)(rdrand64()/MAX);

	return (0.5 + (signed)rdrand32() * .2328306e-9);
}


template <typename T>
double RNG<T>::unid() const
{
	return (0.5 + (signed)rdrand32() * .2328306e-9);
}


template <typename T>
unsigned int RNG<T>::rdrand32() const
{
	unsigned int lvalue;
	unsigned char ok = 0;

	while(!ok)
		asm volatile ("rdrand %0; setc %1"
				: "=r" (lvalue), "=qm" (ok));

	return lvalue;
}


template <typename T>
unsigned long long RNG<T>::rdrand64() const
{
	unsigned long long lvalue;
	unsigned char ok = 0;

	while(!ok)
		asm volatile ("rdrand %0; setc %1"
				: "=r" (lvalue), "=qm" (ok));

	return lvalue;
}


} /* namespace whiteice */

#endif

