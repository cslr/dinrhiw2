

#include "SHA.h"
#include <iostream>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

//#include "dlib.h"
//#include "global.h"


namespace whiteice
{
  namespace crypto
  {
    
    SHA::SHA(unsigned int bits) 
    {
      if(bits != 160 && bits != 256 && bits != 384 && bits != 512)
	throw std::invalid_argument("SHA is available only for 160,256,384 and 512 bit lengths");
      
      this->shalen = bits;

      // FIXME? buggy?
      {
	whiteice::uint32 i = 1;
	
	if(i & 1) this->LITTLE_ENDIAN = true;
	else{
	  std::cout << "WARNING: SHA class might not work correctly on bigendian machines"
		    << std::endl;
	  
	  this->LITTLE_ENDIAN = false;
	}
      }
      
    }
    
    
    SHA::~SHA()
    {
    }
    
    
    bool SHA::hash(unsigned char** data,
		   unsigned int length, // in data length in bytes
		   unsigned char* sha) const   // memory for SHA
    {
      bool ok = false;
      
      if(shalen == 160)      ok = sha1  (data, length, sha);
      else if(shalen == 256) ok = sha256(data, length, sha);
      else if(shalen == 384) ok = sha384(data, length, sha);
      else if(shalen == 512) ok = sha512(data, length, sha);
      
      // tries to realloc() back to original length
      
      unsigned char *ptr = (unsigned char*)
	realloc((void*)(*data), length);
      
      if(ptr) (*data) = ptr;
      
      return ok;
    }
    
    
    // compares two hashes
    bool SHA::check(const unsigned char* sha1, const unsigned char* sha2) const 
    {
      return (memcmp(sha1, sha2, (shalen/8)) == 0);
    }
    
    
    unsigned int SHA::bits() const    // number of bits  in SHA
    {
      return shalen;
    }
    
    
    unsigned int SHA::bytes() const   // number of bytes in SHA    
    {
      return (shalen/8);
    }
    
    /************************************************************/
    
    // TODO: realloc() and pad data memory
    bool SHA::sha1_pad(unsigned char** data, unsigned int& length) const 
    {
      // calculates number of padding bits
      unsigned int blen = (448 - length*8 - 1) % 512;
      blen += 1; // one bit
      
      if (blen < 0) blen += 512;
      
      unsigned char *ptr = (unsigned char*)
	realloc((void*)(*data), length + (blen)/8 + 8);
      
      if(!ptr) return false;
      
      *data = ptr;
      
      // zeroes extra memory
      memset(&(ptr[length]), 0, (blen/8) + 8);
      
      ptr[length] = 0x80; // sets highest bit on
      
      
      // writes length at the end of final block (in a big endian way)
      {
	unsigned int l = length*8;
	unsigned int address = length + (blen/8) + 7;
	
	while(l != 0){
	  
	  ptr[address] = l % 256;
	  l /= 256;
	  
	  address--;
	}
      }
      
      
      length += ((blen/8) + 8);
      
      return true;
    }
    
    
    bool SHA::sha512_pad(unsigned char** data, unsigned int& length) const 
    {
      // calculates number of padding bits
      unsigned int blen = (896 - length*8 - 1) % 1024;
      blen += 1; // one bit
      
      if (blen < 0) blen += 1024;
      
      unsigned char *ptr = (unsigned char*)
	realloc((void*)(*data), length + (blen/8) + 16);
      
      if(!ptr) return false;
      
      *data = ptr;
      
      // zeroes extra memory
      memset(&(ptr[length]), 0, (blen/8) + 16);
      
      ptr[length] = 0x80; // sets highest bit on
      
      
      // writes length at the end of final block (in a big endian way)
      {
	unsigned int l = length*8;
	unsigned int address = length + (blen/8) + 15;
	
	while(l != 0){
	  
	  ptr[address] = l % 256;
	  l /= 256;
	  
	  address--;
	}
      }
      
      
      length += ((blen/8) + 16);
      
      return true;
    }
    
    
    bool SHA::sha1(unsigned char** data, unsigned int length,
		   unsigned char* sha) const 
    {
      whiteice::uint32* M = 0;
      
      try{
	M = new whiteice::uint32[80];

	sha1_pad(data, length); // pads message

	memcpy(sha, SHA1_IHASH, 4*5); // initializes hash
	
	const unsigned int N = (length/64); // number of 512 bit blocks
	
	whiteice::uint32* HASH =
	  (whiteice::uint32*)(sha);

	whiteice::uint32* W = 
	  (whiteice::uint32*)(*data);
	
	whiteice::uint32 A, B, C, D, E;
	whiteice::uint32 temp;
	
	
	for(unsigned int i=0;i<N;i++){
	  
	  memcpy( M, &(W[(i*16)]), 16*4 );
	  
	  if(LITTLE_ENDIAN){
	    for(unsigned int t=0;t<16;t++){
	      change_endianess(M[t]);
	    }
	  }
	  
	  for(unsigned int t=16;t<80;t++){
	    M[t]  = M[(t - 3)];
	    M[t] ^= M[(t - 8)];
	    M[t] ^= M[(t - 14)];
	    M[t] ^= M[(t - 16)];
	    
	    M[t] = ROTL(M[t], 1);
	  }
	  
	  A = HASH[0]; B = HASH[1]; C = HASH[2];
	  D = HASH[3]; E = HASH[4];
	  
	  for(unsigned int t=0;t<80;t++){
	    temp = ROTL(A,5) + sha1_fun(B,C,D,t) + E + M[t] + sha1_constant(t);
	    E = D;
	    D = C;
	    C = ROTL(B,30);
	    B = A;
	    A = temp;
	  }
	  
	  HASH[0] += A; HASH[1] += B; HASH[2] += C;
	  HASH[3] += D; HASH[4] += E;
	}

	
	// changes endianess to small endian
	// (so disk writes etc. write in same order on small/big endian)
	if(LITTLE_ENDIAN)
	  for(unsigned int t=0;t<5;t++)
	    change_endianess(HASH[t]);
	
	
	delete[] M;
	return true;
      }
      catch(std::exception& e){
	if(M) delete[] M;
	return false;
      }
    }
    
    
    bool SHA::sha256(unsigned char** data, unsigned int length,
		     unsigned char* sha) const 
    {
      whiteice::uint32* M = 0;
      
      try{
	M = new whiteice::uint32[64];
	
	sha1_pad(data, length); // pads message
	memcpy(sha, SHA256_IHASH, 4*8); // initializes hash
	
	const unsigned int N = (length/64); // 512 bit blocks
	
	whiteice::uint32* HASH =
	  (whiteice::uint32*)(sha);
	
	whiteice::uint32* W = 
	  (whiteice::uint32*)(*data);
	
	whiteice::uint32 A, B, C, D, E, F, G, H;
	whiteice::uint32 T1, T2;
	
	for(unsigned int i=0;i<N;i++){
	  
	  memcpy( M, &(W[(i*16)]), 16*4 );
	  
	  if(LITTLE_ENDIAN){
	    for(unsigned int t=0;t<16;t++){
	      change_endianess(M[t]);
	    }
	  }
	  
	  for(unsigned int t=16;t<64;t++){
	    M[t]  = sha256_sigma1( M[(t - 2)] );
	    M[t] += M[(t - 7)];
	    M[t] += sha256_sigma0( M[(t - 15)] );
	    M[t] += M[(t - 16)];
	  }
	  
	  A = HASH[0]; B = HASH[1]; C = HASH[2]; D = HASH[3];
	  E = HASH[4]; F = HASH[5]; G = HASH[6]; H = HASH[7];
	  
	  for(unsigned int t=0;t<64;t++){
	    T1 = H + sha256_bsigma1(E) + sha256_ch(E,F,G) +
	         SHA256_TABLE[t] + M[t];
	    
	    T2 = sha256_bsigma0(A) + sha256_maj(A,B,C);
	    
	    H = G;
	    G = F;
	    F = E;
	    E = D + T1;
	    D = C;
	    C = B;
	    B = A;
	    A = T1 + T2;
	  }
	  
	  HASH[0] += A; HASH[1] += B; HASH[2] += C; HASH[3] += D;
	  HASH[4] += E; HASH[5] += F; HASH[6] += G; HASH[7] += H;
	  
	}
	
	
	// changes endianess to small endian
	// (so disk writes etc. write in same order on small/big endian)
	if(LITTLE_ENDIAN)
	  for(unsigned int t=0;t<8;t++)
	    change_endianess(HASH[t]);
	
	
	delete[] M;
	return true;
      }
      catch(std::exception& e){
	if(M) delete[] M;
	return false;
      }
    }
    
    
    bool SHA::sha384(unsigned char** data, unsigned int length,
		     unsigned char* sha) const 
    {
      whiteice::uint64* M = 0;
      
      try{
	M = new whiteice::uint64[80];
	
	whiteice::uint64 HASH[8];
	
	sha512_pad(data, length);
	memcpy(HASH, SHA384_IHASH, 8*8);
	
	const unsigned int N = length / 128; // 1024 bit blocks
	
	whiteice::uint64* W = 
	  (whiteice::uint64*)(*data);
	
	whiteice::uint64 A, B, C, D, E, F, G, H;
	whiteice::uint64 T1, T2;
	
	for(unsigned int i=0;i<N;i++){
	  
	  memcpy( M, &(W[(i*16)]), 16*8 );
	  
	  if(LITTLE_ENDIAN){
	    for(unsigned int t=0;t<16;t++){
	      change_endianess(M[t]);
	    }
	  }
	  
	  for(unsigned int t=16;t<80;t++){
	    M[t]  = sha512_sigma1( M[(t - 2)] );
	    M[t] += M[(t - 7)];
	    M[t] += sha512_sigma0( M[(t - 15)] );
	    M[t] += M[(t - 16)];
	  }
	  
	  A = HASH[0]; B = HASH[1]; C = HASH[2]; D = HASH[3];
	  E = HASH[4]; F = HASH[5]; G = HASH[6]; H = HASH[7];
	  
	  for(unsigned int t=0;t<80;t++){
	    T1 = H + sha512_bsigma1(E) + sha512_ch(E,F,G) +
	      SHA512_TABLE[t] + M[t];
	    
	    T2 = sha512_bsigma0(A) + sha512_maj(A,B,C);
	    
	    H = G;
	    G = F;
	    F = E;
	    E = D + T1;
	    D = C;
	    C = B;
	    B = A;
	    A = T1 + T2;
	  }
		
	  HASH[0] += A; HASH[1] += B; HASH[2] += C; HASH[3] += D;
	  HASH[4] += E; HASH[5] += F; HASH[6] += G; HASH[7] += H;
	}
	
	
	// changes endianess to small endian
	// (so disk writes etc. write in same order on small/big endian)
	if(LITTLE_ENDIAN)
	  for(unsigned int t=0;t<8;t++)
	    change_endianess(HASH[t]);
	
	
	memcpy(sha, HASH, 6*8);
	
	
	delete[] M;
	return true;
      }
      catch(std::exception& e){
	if(M) delete[] M;
	return false;
      }
    }
    
    
    bool SHA::sha512(unsigned char** data, unsigned int length,
		     unsigned char* sha) const 
    {
      whiteice::uint64* M = 0;
      
      try{
	M = new whiteice::uint64[80];
	
	sha512_pad(data, length);
	memcpy(sha, SHA512_IHASH, 8*8);
	
	const unsigned int N = length / 128; // 1024 bit blocks
	
	whiteice::uint64* HASH = 
	  (whiteice::uint64*)(sha);
	
	whiteice::uint64* W = 
	  (whiteice::uint64*)(*data);
	
	whiteice::uint64 A, B, C, D, E, F, G, H;
	whiteice::uint64 T1, T2;
	
	for(unsigned int i=0;i<N;i++){
	  
	  memcpy( M, &(W[(i*16)]), 16*8 );
	  
	  if(LITTLE_ENDIAN){
	    for(unsigned int t=0;t<16;t++){
	      change_endianess(M[t]);
	    }
	  }
	  
	  for(unsigned int t=16;t<80;t++){
	    M[t]  = sha512_sigma1( M[(t - 2)] );
	    M[t] += M[(t - 7)];
	    M[t] += sha512_sigma0( M[(t - 15)] );
	    M[t] += M[(t - 16)];
	  }
	  
	  A = HASH[0]; B = HASH[1]; C = HASH[2]; D = HASH[3];
	  E = HASH[4]; F = HASH[5]; G = HASH[6]; H = HASH[7];
	  
	  for(unsigned int t=0;t<80;t++){
	    T1 = H + sha512_bsigma1(E) + sha512_ch(E,F,G) +
	      SHA512_TABLE[t] + M[t];
	    
	    T2 = sha512_bsigma0(A) + sha512_maj(A,B,C);
	    
	    H = G;
	    G = F;
	    F = E;
	    E = D + T1;
	    D = C;
	    C = B;
	    B = A;
	    A = T1 + T2;
	  }
		
	  HASH[0] += A; HASH[1] += B; HASH[2] += C; HASH[3] += D;
	  HASH[4] += E; HASH[5] += F; HASH[6] += G; HASH[7] += H;
	}
	
	
	// changes endianess to small endian
	// (so disk writes etc. write in same order on small/big endian)
	if(LITTLE_ENDIAN)
	  for(unsigned int t=0;t<8;t++)
	    change_endianess(HASH[t]);
	
	
	delete[] M;
	return true;
      }
      catch(std::exception& e){
	if(M) delete[] M;
	return false;
      }     
    }
    
    

    whiteice::uint32 SHA::sha1_fun(whiteice::uint32 x,
				      whiteice::uint32 y,
				      whiteice::uint32 z,
				      unsigned int t) const 
    {
      // ch
      if(t < 20){
	return ((x & y) ^ ((~ x) & z));
      }
      // parity
      else if(t < 40){
	return (x ^ y ^ z);
      }
      // maj
      else if(t < 60){
	return ((x & y) ^ (x & z) ^ (y & z));
      }
      // parity
      // else if(t < 80){
      
      return (x ^ y ^ z);
    }
    
    
    
    whiteice::uint32 SHA::sha256_ch(whiteice::uint32 x,
				       whiteice::uint32 y,
				       whiteice::uint32 z) const 
    {
      return ((x & y) ^ ((~ x) & z));
    }
    
    
    whiteice::uint32 SHA::sha256_maj(whiteice::uint32 x,
					whiteice::uint32 y,
					whiteice::uint32 z) const 
    {
      return ((x & y) ^ (x & z) ^ (y & z));
    }
    
    
    
    whiteice::uint32 SHA::sha256_bsigma0(whiteice::uint32 x) const 
    {
      return ( ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22) );
    }
    
    whiteice::uint32 SHA::sha256_bsigma1(whiteice::uint32 x) const 
    {
      return ( ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25) );
    }
    
    whiteice::uint32 SHA::sha256_sigma0(whiteice::uint32 x) const 
    {
      return ( ROTR(x,  7) ^ ROTR(x, 18) ^ SHR(x, 3) );
    }
    
    whiteice::uint32 SHA::sha256_sigma1(whiteice::uint32 x) const 
    {
      return ( ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10) );
    }

    
    
    whiteice::uint64 SHA::sha512_ch(whiteice::uint64 x,
				       whiteice::uint64 y,
				       whiteice::uint64 z) const 
    {
      return ((x & y) ^ ( (~ x) & z));
    }
    
    
    whiteice::uint64 SHA::sha512_maj(whiteice::uint64 x,
					whiteice::uint64 y,
					whiteice::uint64 z) const 
    {
      return ((x & y) ^ ( x & z) ^ (y & z));
    }
    
    
    whiteice::uint64 SHA::sha512_bsigma0(whiteice::uint64 x) const 
    {
      return (ROTR(x,28) ^ ROTR(x,34) ^ ROTR(x,39));
    }
    
    whiteice::uint64 SHA::sha512_bsigma1(whiteice::uint64 x) const 
    {
      return (ROTR(x,14) ^ ROTR(x,18) ^ ROTR(x,41));
    }
    
    whiteice::uint64 SHA::sha512_sigma0(whiteice::uint64 x) const 
    {
      return (ROTR(x, 1) ^ ROTR(x, 8) ^ SHR(x, 7));
    }
    
    whiteice::uint64 SHA::sha512_sigma1(whiteice::uint64 x) const 
    {
      return (ROTR(x,19) ^ ROTR(x,61) ^ SHR(x, 6));
    }
    
    
    
    whiteice::uint32 SHA::sha1_constant(unsigned int t) const 
    {
      if(t < 20)      return SHA1_TABLE[0];
      else if(t < 40) return SHA1_TABLE[1];
      else if(t < 60) return SHA1_TABLE[2];
      
      // else if(t < 80)
      
      return SHA1_TABLE[3];
    }
    
    
    whiteice::uint32 SHA::ROTR(whiteice::uint32 x, unsigned int s) const 
    {
      return ((x >> s) | (x << (32 - s)));
    }

    whiteice::uint32 SHA::ROTL(whiteice::uint32 x, unsigned int s) const 
    {
      return ((x << s) | (x >> (32 - s)));
    }
    
    whiteice::uint32 SHA::SHL(whiteice::uint32 x, unsigned int s) const 
    {
      return (x << s);
    }
    
    whiteice::uint32 SHA::SHR(whiteice::uint32 x, unsigned int s) const 
    {
      return (x >> s);
    }
    
    whiteice::uint64 SHA::ROTR(whiteice::uint64 x, unsigned int s) const 
    {
      return ((x >> s) | (x << (64 - s)));
    }
    
    whiteice::uint64 SHA::ROTL(whiteice::uint64 x, unsigned int s) const 
    {
      return ((x << s) | (x >> (64 - s)));
    }
    
    whiteice::uint64 SHA::SHL(whiteice::uint64 x, unsigned int s) const 
    {
      return (x << s);
    }
    
    whiteice::uint64 SHA::SHR(whiteice::uint64 x, unsigned int s) const 
    {
      return (x >> s);
    }
    
    
    
    
    void SHA::change_endianess(whiteice::uint32& x) const 
    {
      x = ((x >> 24) | ((x >> 8) & 0xFF00) | ((x & 0xFF00) << 8) | (x << 24));
    }
    
    void SHA::change_endianess(whiteice::uint64& x) const 
    {
      x = ((x >> 56) | ((x >> 40) & 0xFF00) | ((x >> 24) & 0xFF0000) |
	   ((x >> 8) & 0xFF000000) | ((x & 0xFF000000) << 8) |
	   ((x & 0xFF0000) << 24) | ((x & 0xFF00) << 40) | ((x & 0xFF) << 56));
    }
    
    
    ////////////////////////////////////////////////////////////////////////////////
    // SHA CONSTANTS
    // (in bigendian format)
    
    const whiteice::uint32 SHA::SHA1_TABLE[4] =
    { 
      0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6
    };
    
    
    const whiteice::uint32 SHA::SHA256_TABLE[64] =
    {
      0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
      0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
      0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
      0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
      0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
      0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
      0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
      0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
    };
    
    
    const whiteice::uint64 SHA::SHA512_TABLE[80] =
    {      
      0x428A2F98D728AE22ULL, 0x7137449123EF65CDULL, 0xB5C0FBCFEC4D3B2FULL, 0xE9B5DBA58189DBBCULL,
      0x3956C25BF348B538ULL, 0x59F111F1B605D019ULL, 0x923F82A4AF194F9BULL, 0xAB1C5ED5DA6D8118ULL,
      0xD807AA98A3030242ULL, 0x12835B0145706FBEULL, 0x243185BE4EE4B28CULL, 0x550C7DC3D5FFB4E2ULL,
      0x72BE5D74F27B896FULL, 0x80DEB1FE3B1696B1ULL, 0x9BDC06A725C71235ULL, 0xC19BF174CF692694ULL,
      0xE49B69C19EF14AD2ULL, 0xEFBE4786384F25E3ULL, 0x0FC19DC68B8CD5B5ULL, 0x240CA1CC77AC9C65ULL,
      0x2DE92C6F592B0275ULL, 0x4A7484AA6EA6E483ULL, 0x5CB0A9DCBD41FBD4ULL, 0x76F988DA831153B5ULL,
      0x983E5152EE66DFABULL, 0xA831C66D2DB43210ULL, 0xB00327C898FB213FULL, 0xBF597FC7BEEF0EE4ULL,
      0xC6E00BF33DA88FC2ULL, 0xD5A79147930AA725ULL, 0x06CA6351E003826FULL, 0x142929670A0E6E70ULL,
      0x27B70A8546D22FFCULL, 0x2E1B21385C26C926ULL, 0x4D2C6DFC5AC42AEDULL, 0x53380D139D95B3DFULL,
      0x650A73548BAF63DEULL, 0x766A0ABB3C77B2A8ULL, 0x81C2C92E47EDAEE6ULL, 0x92722C851482353BULL,
      0xA2BFE8A14CF10364ULL, 0xA81A664BBC423001ULL, 0xC24B8B70D0F89791ULL, 0xC76C51A30654BE30ULL,
      0xD192E819D6EF5218ULL, 0xD69906245565A910ULL, 0xF40E35855771202AULL, 0x106AA07032BBD1B8ULL,
      0x19A4C116B8D2D0C8ULL, 0x1E376C085141AB53ULL, 0x2748774CDF8EEB99ULL, 0x34B0BCB5E19B48A8ULL,
      0x391C0CB3C5C95A63ULL, 0x4ED8AA4AE3418ACBULL, 0x5B9CCA4F7763E373ULL, 0x682E6FF3D6B2B8A3ULL,
      0x748F82EE5DEFB2FCULL, 0x78A5636F43172F60ULL, 0x84C87814A1F0AB72ULL, 0x8CC702081A6439ECULL,
      0x90BEFFFA23631E28ULL, 0xA4506CEBDE82BDE9ULL, 0xBEF9A3F7B2C67915ULL, 0xC67178F2E372532BULL,
      0xCA273ECEEA26619CULL, 0xD186B8C721C0C207ULL, 0xEADA7DD6CDE0EB1EULL, 0xF57D4F7FEE6ED178ULL,
      0x06F067AA72176FBAULL, 0x0A637DC5A2C898A6ULL, 0x113F9804BEF90DAEULL, 0x1B710B35131C471BULL,
      0x28DB77F523047D84ULL, 0x32CAAB7B40C72493ULL, 0x3C9EBE0A15C9BEBCULL, 0x431D67C49C100D4CULL,
      0x4CC5D4BECB3E42B6ULL, 0x597F299CFC657E2AULL, 0x5FCB6FAB3AD6FAECULL, 0x6C44198C4A475817ULL
    };
    
    
    const whiteice::uint32 SHA::SHA1_IHASH[5] = 
    { 
      0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
    };
    
  
    const whiteice::uint32 SHA::SHA256_IHASH[8] =
    {
      0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
    };
  
  
    const whiteice::uint64 SHA::SHA384_IHASH[8] =
    {
      0xCBBB9D5DC1059ED8ULL, 0x629A292A367CD507ULL,
      0x9159015A3070DD17ULL, 0x152FECD8F70E5939ULL,
      0x67332667FFC00B31ULL, 0x8EB44A8768581511ULL,
      0xDB0C2E0D64F98FA7ULL, 0x47B5481DBEFA4FA4ULL
    };
  
    
    const whiteice::uint64 SHA::SHA512_IHASH[8] =
    {
      0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
      0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
      0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
      0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
    };
    
    
  };
};
