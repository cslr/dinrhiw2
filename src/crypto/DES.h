/*
 * DES (data encryption standard)
 * (this is obsolette and quite weak - use AES instead)
 * 
 * NOT OPTIMIZED SO DES IS THIS IMPLEMENTATION IS VERY SLOW
 */

#ifndef DES_h
#define DES_h

#include <stdexcept>
#include <exception>
#include "Cryptosystem.h"
#include "dynamic_bitset.h"


namespace whiteice
{
  namespace crypto
  {
    
    class NDESKey;
    class DESKey;
    
    
    // DES
    class DES : public SymmetricCryptosystem<dynamic_bitset, dynamic_bitset>
    {
    public:
	
      virtual ~DES(){ }
      
      // data must be 64 bit and Keyschedule should be DESKey
      bool encrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) ;
      bool decrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) ;
      
      bool encrypt(data_source<dynamic_bitset>& data,
		   const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		   ModeOfOperation mode = ECBmode) ;
      
      bool decrypt(data_source<dynamic_bitset>& data,
		   const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		   ModeOfOperation mode = ECBmode) ;
      
    public:
      
      // initial permutation
      static const unsigned int IP0[64];
      
      // inverse of initial permutation
      static const unsigned int IP1[64];
      
      // DES f function permutation
      static const unsigned int P0[32];
      
      // DES f function expansion function
      static const unsigned int E[48];
      
      // DES f function S-boxes
      
      static const unsigned int S1[4][16];
      static const unsigned int S2[4][16];
      static const unsigned int S3[4][16];
      static const unsigned int S4[4][16];
      static const unsigned int S5[4][16];
      static const unsigned int S6[4][16];
      static const unsigned int S7[4][16];
      static const unsigned int S8[4][16];
      
      
    private:
      
      // A = data, J = key
      bool f(dynamic_bitset& A, const dynamic_bitset& J) ;
    };
    
    
    // triple DES
    class TDES : public SymmetricCryptosystem<dynamic_bitset, dynamic_bitset>
    {
    public:
	
      virtual ~TDES(){ }
      
      // data must be 64 bit and Keyschedule should be NDESKey with 3 DESKeys
      // (key schedule must provide exactly 48 round keys which are 48 bits long
      bool encrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) ;
      bool decrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) ;
      
      // note: mode of operation is applied separatedly to each single DES operation
      bool encrypt(data_source<dynamic_bitset>& data,
		   const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		   ModeOfOperation mode = ECBmode) ;
      
      // note: mode of operation is applied separatedly to each single DES operation
      bool decrypt(data_source<dynamic_bitset>& data,
		   const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		   ModeOfOperation mode = ECBmode) ;
    };

    
    
    // normal DES key
    class DESKey : public Keyschedule<dynamic_bitset>
    {
    public:
      
      explicit DESKey(const dynamic_bitset& key);
      DESKey(const DESKey& k);
      virtual ~DESKey();
      
      unsigned int size() const ;
            
      unsigned int keybits() const ;
      
      const dynamic_bitset& operator[](unsigned int n)
	const ;
      
      Keyschedule<dynamic_bitset>* copy() const;
      
    public:

      static const unsigned int PC1_C0[28];
      static const unsigned int PC1_D0[28];      
      static const unsigned int PC2[56];
      
      // DES key generation left shift schedule
      static const unsigned int left_shifts[16];
      
    private:
      
      void calculate_roundkeys();
      
      dynamic_bitset key;
      
      dynamic_bitset derived_keys[16]; // 16 round keys calculated from key
      
    };

    
    // N-DES key (especially for triple DES with vector size 3)
    class NDESKey : public Keyschedule<dynamic_bitset>
    {
    public:
      
      explicit NDESKey(const dynamic_bitset& key);
      explicit NDESKey(const std::vector<dynamic_bitset>& key);
      NDESKey(const NDESKey& key);
      virtual ~NDESKey();
      
      unsigned int size() const ;
            
      unsigned int keybits() const ;
      
      const dynamic_bitset& operator[](unsigned int n)
	const ;
      
      Keyschedule<dynamic_bitset>* copy() const;
      
    private:
      
      // n DES keys
      std::vector<DESKey*> keys;
    };
    
    
    // creates shorter partial key schedule
    // from the longer one by wrapping given Keyschedule
    class PartialKeyschedule : public Keyschedule<dynamic_bitset>
    {
    public:
      PartialKeyschedule(const Keyschedule<dynamic_bitset>& k,
			 unsigned int begin, unsigned int len);
      
      PartialKeyschedule(const PartialKeyschedule& k);
      virtual ~PartialKeyschedule();
      
      unsigned int size() const ;
            
      unsigned int keybits() const ;
      
      const dynamic_bitset& operator[](unsigned int n)
	const ;
      
      Keyschedule<dynamic_bitset>* copy() const;
      
    private:
      unsigned int begin;
      unsigned int len;
      
      // copy of wrapped key schedule
      Keyschedule<dynamic_bitset>* ks;
    };
    
  };
};


#endif



