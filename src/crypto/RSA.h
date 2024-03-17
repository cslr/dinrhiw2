/*
 * RSA (Rivest-Shamir-Adleman)
 * public key cryptography
 *
 */

#ifndef RSA_h
#define RSA_h

#include <vector>
#include <stdexcept>
#include <exception>
#include "Cryptosystem.h"
#include "dynamic_bitset.h"
#include "integer.h"


namespace whiteice
{
  namespace crypto
  {
    using whiteice::math::integer;
    
    
    class RSA : public UnsymmetricCryptosystem<integer, integer>
    {
    public:
      
      RSA();
      virtual ~RSA();
      
      // data must be 64 bit and Keyschedule should be DESKey
      bool encrypt(integer& data, const Keyschedule<integer>& k) ;
      bool decrypt(integer& data, const Keyschedule<integer>& k) ;
      
      bool encrypt(data_source<integer>& data, const Keyschedule<integer>& k) ;
      bool decrypt(data_source<integer>& data, const Keyschedule<integer>& k) ;      
      
    private:
      
    };
    

    
    class RSAKey : public PublicPrivateKeyPair<integer>
    {
    public:
      
      // generates new public / private key pair
      // (n = p*q will be 'bits' bits long)
      RSAKey(unsigned int bits);
      
      RSAKey(const std::vector<integer>& public_key); // only public key available
      
      RSAKey(const std::vector<integer>& public_key,
	     const std::vector<integer>& private_key);
      
      virtual ~RSAKey();
      
	     
      // gets n:th key from the key schedule
      const std::vector<integer>& publickey() const ;
      
      // gets n:th key from the key schedule
      const std::vector<integer>& privatekey() const ;
      
      
      // returns number of keys in a keyschedule
      // or negative value for infinity
      unsigned int size() const ;
      
      // returns number of bits in a single key
      unsigned int keybits() const ;
      
      // gets n:th key from the key schedule
      const integer& operator[](unsigned int n) const ;
      
      // Keyschedules can copy itself
      Keyschedule<integer>* copy() const;
      
      
    private:
      
      void generate_prime(integer& a, unsigned int bits) const ;
      bool random_bit() const ;
      void choose_random_mod_invertible_number(integer& x,
					       const integer& phi) const ;
      void reset(integer& a) const ;
      
    private:
      
      unsigned int bits;
      
      std::vector<integer> public_keydata;
      std::vector<integer> private_keydata;
      
    };
    
    
    
  };
};



#endif







