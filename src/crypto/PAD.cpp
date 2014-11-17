

#include "PAD.h"
#include "dynamic_bitset.h"


namespace whiteice
{
  namespace crypto
  {
    
    
    bool PAD::encrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) throw()
    {
      try{	
	if(k.size() <= 0) return false;
	if(k.keybits() != data.size()) return false;
	
	data ^= k[0];
	
	return true;
      }
      catch(std::exception& e){ return false; }
    }
    
    
    bool PAD::decrypt(dynamic_bitset& data, const Keyschedule<dynamic_bitset>& k) throw()
    {
      try{
	if(k.size() <= 0) return false;
	if(k.keybits() != data.size()) return false;
      
	data ^= k[0];
	
	return true;
      }
      catch(std::exception& e){
	return false;
      }
    }
    
    
    bool PAD::encrypt(data_source<dynamic_bitset>& data,
		      const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		      ModeOfOperation mode) throw()
    {
      for(unsigned int i=0;i<data.size();i++)
	if(encrypt(data[i], k) == false) return false;
      
      return true;
    }
    
    

      
    bool PAD::decrypt(data_source<dynamic_bitset>& data,
		      const Keyschedule<dynamic_bitset>& k, const dynamic_bitset& IV,
		      ModeOfOperation mode) throw()
    {
      for(unsigned int i=0;i<data.size();i++)
	if(decrypt(data[i], k) == false) return false;
      
      return true;
    }
    
    ////////////////////////////////////////////////////////////
    
    
    PADKey::PADKey(const dynamic_bitset& key)
    {
      this->pad.resize(key.size());
      
      this->pad = key;
    }
    
    
    PADKey::PADKey(const PADKey& k)
    {
      this->pad.resize(k.pad.size());
      
      this->pad = k.pad;
    }
    
    
    PADKey::~PADKey()
    {
      // zeroes the pad
      pad.reset();
    }
    
    
    unsigned int PADKey::size() const throw(){ return 1; }
    
    unsigned int PADKey::keybits() const throw(){ return pad.size(); }
    
    const dynamic_bitset& PADKey::operator[](unsigned int n)
      const throw(std::out_of_range)
    {
      if(n == 0) return pad;
      else throw 
      std::out_of_range("Key index too big. PADKey has only one round key");
    }
    
    
    Keyschedule<dynamic_bitset>* PADKey::copy() const{ return new PADKey(*this); }
    
  };
};

