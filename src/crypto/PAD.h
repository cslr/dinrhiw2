/*
 * trivial but sometimes useful
 * one time pad symmetric cryptosystem
 */

#ifndef PAD_h
#define PAD_h

#include <stdexcept>
#include <exception>
#include "Cryptosystem.h"
#include "dynamic_bitset.h"

namespace whiteice
{
  namespace crypto
  {
    
    class PADKey;
    
    
    class PAD : public SymmetricCryptosystem<dynamic_bitset, dynamic_bitset>
    {
    public:
      
      virtual ~PAD(){ }
      
      bool encrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) ;
      bool decrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) ;
      
      // note: ModeOfOperation is ignored
      bool encrypt(data_source<dynamic_bitset>& data,
		   const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		   ModeOfOperation mode = ECBmode) ;
      
      // note: ModeOfOperation is ignored
      bool decrypt(data_source<dynamic_bitset>& data,
		   const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		   ModeOfOperation mode = ECBmode) ;
    };
    
    
    class PADKey : public Keyschedule<dynamic_bitset>
    {
    public:
      explicit PADKey(const dynamic_bitset& key);
      PADKey(const PADKey& k);
      
      virtual ~PADKey();
      
      unsigned int size() const ;
      
      unsigned int keybits() const ;
      
      const dynamic_bitset& operator[](unsigned int n)
	const ;
      
      Keyschedule<dynamic_bitset>* copy() const;
      
    private:
      
      dynamic_bitset pad;
    };
    
  };
};


#endif

