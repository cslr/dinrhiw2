/*
 * File datasource for crypting.
 * 
 */

#include <string>
#include <dinrhiw/dinrhiw.h>


template <unsigned int B>
class CryptoFileSource : public whiteice::data_source<whiteice::dynamic_bitset>
{
 public:
  CryptoFileSource(const std::string& filename,
		   bool encrypting, bool aarmour = false);
  
  ~CryptoFileSource();
  
  whiteice::dynamic_bitset& operator[](unsigned int index)
    throw(std::out_of_range){ return (*DC)[index]; }
  
  const whiteice::dynamic_bitset& operator[](unsigned int index)
    const throw(std::out_of_range){ return (*DC)[index]; }
  
  unsigned int size() const throw();
  
  bool good() const throw();
  
  void flush() const;
  
  // writes data to file with given filename
  bool write(const std::string& filename) const throw();
  
  const whiteice::dynamic_bitset& getIV(){ return IV; }
  
 private:
  
  FILE* file;
  bool encrypting;   // true if we are encrypting data, 
                     // false otherwise
  bool asciiarmour;
  whiteice::dynamic_bitset IV; 
  
  mutable unsigned char* buffer;
  mutable whiteice::crypto::DataConversion<B>* DC;
  mutable unsigned long long filesize;
  
};


extern template class CryptoFileSource<128u>;



