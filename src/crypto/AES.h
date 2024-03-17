/*
 * AES encryption
 */

#ifndef AES_h
#define AES_h


#include <stdexcept>
#include <exception>
#include "Cryptosystem.h"
#include "dynamic_bitset.h"
#include "function.h"


namespace whiteice
{
  namespace crypto
  {
    
    class AESKey;
    
    
    class AES : public SymmetricCryptosystem<dynamic_bitset, dynamic_bitset>
    {
    public:
      AES();      
      virtual ~AES();
      
      // data must be X bit and Keyschedule should be AESKey
      bool encrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) ;
      bool decrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) ;
      
      bool encrypt(data_source<dynamic_bitset>& data,
		   const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		   ModeOfOperation mode = ECBmode) ;
      
      bool decrypt(data_source<dynamic_bitset>& data,
		   const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		   ModeOfOperation mode = ECBmode) ;
       
    private:
      
      void substitute(dynamic_bitset& data) const;      // SubBytes
      void inv_substitute(dynamic_bitset& data) const;  // InvSubBytes
      
      void shift_rows(dynamic_bitset& data) const;      // ShiftRows()
      void inv_shift_rows(dynamic_bitset& data) const;  // InvShiftRows()
            
      void mix_columns(dynamic_bitset& data) const;     // MixColumns()
      void inv_mix_columns(dynamic_bitset& data) const; // InvMixColumns()
      
      void gf_precalculate();
      
      void xmulti(unsigned char& p) const;
      
      inline unsigned int mod(int i, unsigned int modulo) const  PURE_FUNCTION;
      
      void self_test();
      
      
    public:
      
      //      static const unsigned char SBOX1[16][16];
      //      static const unsigned char SBOX2[16][16]; // inverse SBOX1
      
      static const unsigned char SBOX1[256];
      static const unsigned char SBOX2[256]; // inverse SBOX1
      
      static const unsigned int rowshifts[4];
      
    private:
      
      // precalculated GF(2^8) multiplication table
      unsigned char* GFMULTI;
      
      mutable dynamic_bitset s; // global temp variable
    };
    
    
    class AESKey : public Keyschedule<dynamic_bitset>
    {
    public:
      
      explicit AESKey(const dynamic_bitset& key);
      AESKey(const AESKey& k);
      virtual ~AESKey();
      
      unsigned int size() const ;
      
      // resizes keyschedule to be smaller
      bool resize(unsigned int s) ;
      
      unsigned int keybits() const ;
      
      const dynamic_bitset& operator[](unsigned int n)
	const ;
      
      Keyschedule<dynamic_bitset>* copy() const;
      
    public:
      
      static const unsigned int rcon[10]; // (TODO: extend to longer values)
      
    private:
      
      unsigned int substitute_word(unsigned int x) const PURE_FUNCTION;
      unsigned int rotate_word(unsigned int x) const PURE_FUNCTION;
      
      std::vector<dynamic_bitset> keys;
            
    };
    
  };
};




#endif
