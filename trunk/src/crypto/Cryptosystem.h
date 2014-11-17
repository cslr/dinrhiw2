/*
 * Cryptosystem interface
 */

#ifndef Cryptosystem_h
#define Cryptosystem_h


#include <stdexcept>
#include <vector>

#include "data_source.h"


namespace whiteice
{
  namespace crypto
  {
    enum ModeOfOperation {
      ECBmode, CFBmode, CBCmode, OFBmode, CTRmode
    };
    
    
    template <typename KEY>
      class Keyschedule;
    
    
    template <typename DATA, typename KEY>
      class SymmetricCryptosystem
      {
      public:
	
	virtual ~SymmetricCryptosystem(){ }
	
	// encrypts / decrypts given data or returns false
	// if encryption failed.

	// uses ECB mode of operation
	virtual bool encrypt(DATA& data, const Keyschedule<KEY>& k) throw() = 0;
	virtual bool decrypt(DATA& data, const Keyschedule<KEY>& k) throw() = 0;
	
	// encrypt/decrypt with given mode of operation
	virtual bool encrypt(data_source<DATA>& data,
			     const Keyschedule<KEY>& k, const DATA& IV,
			     ModeOfOperation mode = ECBmode) throw() = 0;
			     
	virtual bool decrypt(data_source<DATA>& data,
			     const Keyschedule<KEY>& k, const DATA& IV,
			     ModeOfOperation mode = ECBmode) throw() = 0;
			     
      };

    
    template <typename DATA, typename KEY>
      class UnsymmetricCryptosystem
      {
      public:
	
	virtual ~UnsymmetricCryptosystem(){ }
	
	// encrypts / decrypts given data or returns false
	// if encryption failed.

	// uses ECB mode of operation
	virtual bool encrypt(DATA& data, const Keyschedule<KEY>& k) throw() = 0;
	virtual bool decrypt(DATA& data, const Keyschedule<KEY>& k) throw() = 0;
	
	// encrypt/decrypt with given mode of operation
	virtual bool encrypt(data_source<DATA>& data,
			     const Keyschedule<KEY>& k) throw() = 0;
			     
	virtual bool decrypt(data_source<DATA>& data,
			     const Keyschedule<KEY>& k) throw() = 0;
			     
      };
	
	
	
    template <typename KEY>
      class Keyschedule
      {
      public:
	
	virtual ~Keyschedule(){ }
	
	// returns number of keys in a keyschedule
	// or ((unsigned int)(-1)) for infinity
	virtual unsigned int size() const throw() = 0;
	
	// returns number of bits in a single key      
	virtual unsigned int keybits() const throw() = 0;
	
	// gets n:th key from the key schedule
	virtual const KEY& operator[](unsigned int n)
	  const throw(std::out_of_range) = 0;
	
	// Keyschedules can copy itself
	virtual Keyschedule<KEY>* copy() const = 0;
	
      };
    
    
    template <typename KEY>
      class PublicPrivateKeyPair : public Keyschedule<KEY>
      {
      public:
	
	PublicPrivateKeyPair(){ }
	virtual ~PublicPrivateKeyPair(){ }
	
	// returns publickeys
	virtual const std::vector<KEY>& publickey() const throw() = 0;
	
	// returns privatekeys
	virtual const std::vector<KEY>& privatekey() const throw() = 0;
      };
    
    
    /*
     * unkeyed cryptographic hash
     * interface
     */
    class CryptoHash
    {
    public:
      
      virtual ~CryptoHash(){ }
      
      /* calculates cryptographic hash of the malloc'ed data
       * length is data length in *bytes* and
       * shash must be pointer to large enough memory
       * where (secure) hash will be calculated to.
       * length of the data area maybe increased with realloc()
       * because of padding for the hash calculation
       */
      virtual bool hash(unsigned char** data, // pointer to data pointer
			unsigned int length,
			unsigned char* shash) const throw() = 0;
      
      // compares two hashes
      virtual bool check(const unsigned char* sha1,
			 const unsigned char* sha2) const throw() = 0;
      
      // number of bits in hash
      virtual unsigned int bits() const throw() = 0;
      
      // number of bytes in hash
      virtual unsigned int bytes() const throw() = 0;
    };
    
    
  };
};

#endif

