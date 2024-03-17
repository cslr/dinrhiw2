/*
 * SHA - secure hash algorithms
 *       (for little endian machines)
 */

#ifndef SHA_h
#define SHA_h

#include <stdexcept>
#include <exception>

#include "global.h"
#include "function.h"
#include "Cryptosystem.h"

#ifdef LITTLE_ENDIAN
#undef LITTLE_ENDIAN
#endif



namespace whiteice
{
  namespace crypto
  {
    
    class SHA : public CryptoHash
    {
    public:
      SHA(unsigned int bits) ;
      virtual ~SHA();
      
      bool hash(unsigned char** data,
		unsigned int length,
		unsigned char* sha) const ;
	
      // compares two hashes
      bool check(const unsigned char* sha1, const unsigned char* sha2) const ;
      
      unsigned int bits() const ;  // number of bits  in SHA
      unsigned int bytes() const ; // number of bytes in SHA
      
    private:
      
      bool sha1_pad(unsigned char** data, unsigned int& length) const ;
      bool sha512_pad(unsigned char** data, unsigned int& length) const ;
      
      
      bool sha1(unsigned char** data, unsigned int length,
		unsigned char* sha) const ;

      bool sha256(unsigned char** data, unsigned int length,
		  unsigned char* sha) const ;

      bool sha384(unsigned char** data, unsigned int length,
		  unsigned char* sha) const ;

      bool sha512(unsigned char** data, unsigned int length,
		  unsigned char* sha) const ;
      
      
      
      whiteice::uint32 sha1_fun(whiteice::uint32 x,
				   whiteice::uint32 y,
				   whiteice::uint32 z,
				   unsigned int t) const  PURE_FUNCTION;
      
      
      whiteice::uint32 sha256_ch(whiteice::uint32 x,
				    whiteice::uint32 y,
				    whiteice::uint32 z) const  PURE_FUNCTION;
      
      whiteice::uint32 sha256_maj(whiteice::uint32 x,
				     whiteice::uint32 y,
				     whiteice::uint32 z) const  PURE_FUNCTION;

      whiteice::uint32 sha256_bsigma0(whiteice::uint32 x)
	const  PURE_FUNCTION;

      whiteice::uint32 sha256_bsigma1(whiteice::uint32 x)
	const  PURE_FUNCTION;

      whiteice::uint32 sha256_sigma0(whiteice::uint32 x)
	const  PURE_FUNCTION;
      
      whiteice::uint32 sha256_sigma1(whiteice::uint32 x)
	const  PURE_FUNCTION;
      
      
      whiteice::uint64 sha512_ch(whiteice::uint64 x,
				    whiteice::uint64 y,
				    whiteice::uint64 z)
	const  PURE_FUNCTION;
      
      
      whiteice::uint64 sha512_maj(whiteice::uint64 x,
				     whiteice::uint64 y,
				     whiteice::uint64 z)
	const  PURE_FUNCTION;
      
      
      whiteice::uint64 sha512_bsigma0(whiteice::uint64 x)
	const  PURE_FUNCTION;
      
      
      whiteice::uint64 sha512_bsigma1(whiteice::uint64 x)
	const  PURE_FUNCTION;
      
      
      whiteice::uint64 sha512_sigma0(whiteice::uint64 x)
	const  PURE_FUNCTION;
      
      
      whiteice::uint64 sha512_sigma1(whiteice::uint64 x)
	const  PURE_FUNCTION;
      
      
      whiteice::uint32 sha1_constant(unsigned int t)
	const  PURE_FUNCTION;
      
      
      whiteice::uint32 ROTR(whiteice::uint32 x, unsigned int s)
	const  PURE_FUNCTION;
      
      whiteice::uint32 ROTL(whiteice::uint32 x, unsigned int s)
	const  PURE_FUNCTION;
      
      whiteice::uint32 SHR (whiteice::uint32 x, unsigned int s)
	const  PURE_FUNCTION;
      
      whiteice::uint32 SHL (whiteice::uint32 x, unsigned int s)
	const  PURE_FUNCTION;
      
      whiteice::uint64 ROTR(whiteice::uint64 x, unsigned int s)
	const  PURE_FUNCTION;
      
      whiteice::uint64 ROTL(whiteice::uint64 x, unsigned int s)
	const  PURE_FUNCTION;
      
      whiteice::uint64 SHR (whiteice::uint64 x, unsigned int s)
	const  PURE_FUNCTION;
      
      whiteice::uint64 SHL (whiteice::uint64 x, unsigned int s)
	const  PURE_FUNCTION;
			      
      
      // conversion between big endian and small endian values
      
      void change_endianess(whiteice::uint32& x) const ;
      void change_endianess(whiteice::uint64& x) const ;
	
      
      
    private:
      // length in bits
      unsigned int shalen;
      
      bool LITTLE_ENDIAN;
      
      
      // SHA CONSTANTS
      
      static const whiteice::uint32 SHA1_TABLE[4];
      static const whiteice::uint32 SHA256_TABLE[64];
      static const whiteice::uint64 SHA512_TABLE[80];
      
      // initial hash values;
      static const whiteice::uint32 SHA1_IHASH[5];
      static const whiteice::uint32 SHA256_IHASH[8];
      static const whiteice::uint64 SHA384_IHASH[8];
      static const whiteice::uint64 SHA512_IHASH[8];
      
      
    };
    
  };
};

#endif


