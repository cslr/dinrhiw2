
#include "conversion.h"
#include <string.h>


namespace whiteice
{


  // maps IEEE 754 single precision floating point number to
  // unsigned 32-bit integer
  unsigned int ieee754spf2uint32(const float& f)
  {
    unsigned int i;
    
    memcpy(&i, &f, sizeof(f));
    
    // checks if number is NaN or Inf
    if(((i >> 23) & 0xFF) == 0xFF){
      
      // checks lowest 23 bits (mantissa)
      if((i & 0x7FFFFF) != 0){ // NaN
	// two forms of NaN: plus or minus      
	if((i & 0x80000000) == 0){ // non-negative
	  return 0xFFFFFFFF; // NaN are deicided to be "biggest" values
	}
	else{ // sign bit on: negative
	  return 0xFFFFFFFE;
	}
      }
      else{ // infinities
	if((i & 0x80000000) == 0){ // non-negative
	  return 0xFFFFFFFD; // +Inf is biggest right after NaNs
	}
	else{ // sign bit on: negative
	  return 0; // -Inf is smallest value
	}
      }
    }
    else if(((i >> 23) & 0xFF) == 0){ // zero exponent
      
      if((i & 0x7FFFFF) == 0){ // zero
	if(i & 0x80000000){ // negative zero
	  
	  return 0x80000000; // sign bit on, zero exponent, zero mantissa
	}
	else{ // positive zero
	  
	  return 0x80000001; // sign bit on, zero exponent, mantissa = 1
	}
      }
    }
    
    // handles 'normal' number representation
    
    // toggles the sign bit
    i = i ^ 0x80000000;
    
    // if f is negative, reverses mantissa, reverses exponent
    if((i & 0x80000000) == 0){      
      //   reverse(exp)                      reverses mantissa
      i = (0x7F800000 - (i & 0x7F800000)) + (0x7FFFFF - (i & 0x7FFFFF));
    }
    else{ // positive numbers
      // makes room for two zeros
      i += 2;
    }
    
    return i;
  }
  
  
  
  float uint322ieee754spf(const unsigned int& i)  // inverse
  {
    float f;
    unsigned int j;
    
    // checks for NaNs, Infs and zeros
    
    if(i == 0xFFFFFFFF){ // positive NaN
      j = 0x7FFFFFFF;
    }
    else if(i == 0xFFFFFFFE){ // negative NaN
      j = 0xFFFFFFFF;
    }
    else if(i == 0xFFFFFFFD){ // +Inf
      j = 0x7F800000;
    }
    else if(i == 0){ // -Inf
      j = 0xFF800000;
    }
    else if(i == 0x80000000){ // negative zero
      j = 0x80000000;
    }
    else if(i == 0x80000001){ // positive zero
      j = 0x0;
    }
    else{
      // 'normal' numbers         
      
      if((i & 0x80000000) == 0){ // negative number
	
	//  reverses exponent                 reverses mantissa
	j = (0x7F800000 - (i & 0x7F800000)) + (0x7FFFFF - (i & 0x7FFFFF));
	
	j = j ^ 0x80000000; // toggles signbit
      }
      else{ // positive floating point number
	j = i - 2;
	
	j = j ^ 0x80000000; // changes signbit
      }
    }
    
    
    memcpy(&f, &j, sizeof(f));
    
    return f;             
  }
  

  /*************************************************************/

  
  // maps IEEE 754 double precision floating point to
  // unsigned 64-bit integer
  unsigned long long ieee754dpf2uint64(const double& f)
  {
    unsigned long long i;
    memcpy(&i, &f, sizeof(f));
    
    // checks if number is NaN or Inf
    if(((i >> 52) & 0x7FF) == 0x7FF){
      
      // checks lowest 52 bits (mantissa)
      if((i & 0xFFFFFFFFFFFFFULL) != 0){ // NaN
	
	// two forms of NaN: plus or minus      
	if((i & 0x8000000000000000ULL) == 0){ // non-negative
	  return 0xFFFFFFFFFFFFFFFFULL; // NaN are deicided to be "biggest" values
	}
	else{ // sign bit on: negative
	  return 0xFFFFFFFFFFFFFFFEULL;
	}
      }
      else{ // infinities
	if((i & 0x8000000000000000ULL) == 0){ // non-negative
	  return 0xFFFFFFFFFFFFFFFDULL; // +Inf is biggest right after NaNs
	}
	else{ // sign bit on: negative
	  return 0ULL; // -Inf is smallest value
	}
      }
    }
    else if(((i >> 52) & 0x7FF) == 0){ // zero exponent
      
      if((i & 0xFFFFFFFFFFFFFULL) == 0){ // zero (mantissa is zero)
	if(i & 0x8000000000000000ULL){ // negative zero
	  
	  return 0x8000000000000000ULL; // sign bit on, zero exponent, zero mantissa
	}
	else{ // positive zero
	  
	  return 0x8000000000000001ULL; // sign bit on, zero exponent, mantissa = 1
	}
      }
    }
    
    // handles 'normal' number representation
    
    // toggles the sign bit
    i = i ^ 0x8000000000000000ULL;
    
    // if f is negative, reverses mantissa, reverses exponent
    if((i & 0x8000000000000000ULL) == 0){
      //   reverse(exp)                      reverses mantissa
      i = (0x7FF0000000000000ULL - (i & 0x7FF0000000000000ULL)) + 
	  (0xFFFFFFFFFFFFFULL - (i & 0xFFFFFFFFFFFFFULL));
    }
    else{ // positive numbers
      // makes room for two zeros
      i += 2;
    }
    
    return i;
  }
  
  
  double uint642ieee754dpf(const unsigned long long& i)  // inverse
  {
    double f;
    unsigned long long j;
    
    // checks for NaNs, Infs and zeros
    
    if(i == 0xFFFFFFFFFFFFFFFFULL){ // positive NaN
      j =   0x7FFFFFFFFFFFFFFFULL;
      
      // slow: optimize with assembly to 
      // few simple&fast assembly commands
      memcpy(&f, &j, sizeof(f));
      return f;
    }
    else if(i == 0xFFFFFFFFFFFFFFFEULL){ // negative NaN
      j = 0xFFFFFFFFFFFFFFFFULL;
      
      memcpy(&f, &j, sizeof(f));
      return f;
    }
    else if(i == 0xFFFFFFFFFFFFFFFDULL){ // +Inf
      j = 0x7FF0000000000000ULL;      
      
      memcpy(&f, &j, sizeof(f));
      return f;
    }
    else if(i == 0){ // -Inf
      j = 0xFFF0000000000000ULL;
      
      memcpy(&f, &j, sizeof(f));
      return f;
    }
    
    else if(i == 0x8000000000000000ULL){ // negative zero
      j = 0x8000000000000000ULL;
      
      memcpy(&f, &j, sizeof(f));
      return f;
    }
    else if(i == 0x8000000000000001ULL){ // positive zero
      j = 0x0ULL;
      
      memcpy(&f, &j, sizeof(f));
      return f;
    }
    
    // 'normal' numbers         
    
    if((i & 0x8000000000000000ULL) == 0){ // negative number
      
      //  reverses exponent                 reverses mantissa
      j = (0x7FF0000000000000ULL - (i & 0x7FF0000000000000ULL )) + 
	  (0xFFFFFFFFFFFFFULL - (i & 0xFFFFFFFFFFFFFULL));
      
      j = j ^ 0x8000000000000000ULL; // toggles signbit
      
      memcpy(&f, &j, sizeof(f));
      return f;
    }
    else{ // positive floating point number
      j = i - 2;
      
      j = j ^ 0x8000000000000000ULL; // toggles signbit
      
      memcpy(&f, &j, sizeof(f));
      return f;
    }
    
  }
  
  
};






