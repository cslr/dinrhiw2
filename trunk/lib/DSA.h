/*
 * 160 bit DSA (with L = 1024)
 */

#ifndef DSA_h
#define DSA_h

#include <stdexcept>
#include <exception>

#include "Cryptosystem.h"
#include "dynamic_bitset.h"
#include "function.h"

namespace whiteice
{
  namespace crypto
  {        
    class DSAKey;
    using whiteice::math::integer;
    
    class DSA
    {
    public:
      
      DSA(); // creates new key
      DSA(const DSA& dsa); // gets key from the DSA
      DSA(const DSAKey& key);
      
      ~DSA();
      
      // allocates memory with malloc() and calculates signature for len bytes long
      // message. (message pointer may be changed and length increased)
      
      bool sign(unsigned char** message, unsigned int len,
		std::vector<integer>& signature) const throw();
      
      // returns true if the signature of a message matches with the given message
      bool verify(unsigned char** message, unsigned int len,
		  const std::vector<integer>& signature) const throw();
      
      bool getKey(DSAKey& dsakey) const throw();
      
    private:
      
      bool calculate_sha(integer& sha, unsigned char **message, unsigned int len)
	const throw();
			 
      
      bool random_bit() const throw();
      
    private:
      
      DSAKey* key;
    };
    
    
    class DSAKey : public PublicPrivateKeyPair<integer>
    {
    public:
      
      DSAKey(); // generates new public/private key pair
      
      // public key only
      DSAKey(const integer& p,     const integer& q,
	     const integer& alpha, const integer& beta);
      
      // public & private key
      DSAKey(const integer& p,     const integer& q,
	     const integer& alpha, const integer& beta,
	     const integer& a);
      
      DSAKey(const DSAKey& dsakey);
      virtual ~DSAKey();
      
      // returns number of keys in a keyschedule
      // or negative value for infinity
      unsigned int size() const throw();
      
      // returns number of bits in a single key
      unsigned int keybits() const throw();
      
      // gets n:th key from the key schedule
      const integer& operator[](unsigned int n)
	const throw(std::out_of_range);
      
      DSAKey& operator=(const DSAKey& dsakey) throw();
      
      // Keyschedules can copy itself
      Keyschedule<integer>* copy() const;
      
      const std::vector<integer>& publickey() const throw();
      
      const std::vector<integer>& privatekey() const throw();
      
    private:
      
      void generate_random_prime(integer& a, unsigned int bits) const throw();
      
      void generate_random_even(integer& a, unsigned int bits) const throw();
      
      void random_number_smaller_than(integer& a, const integer& b) const throw();
      
      bool random_bit() const throw();
      
      bool check_key_values() const throw();
      
    private:
      
      // 0 = q, 1 = q, 2 = alpha, 3 = beta
      std::vector<integer> pk;
      
      // 0 = a
      std::vector<integer> sk;
    };
    
    
  };
  
};




#endif
