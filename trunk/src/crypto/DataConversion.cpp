
#ifndef DataConversion_cpp
#define DataConversion_cpp


#include "DataConversion.h"
#include <stdexcept>

namespace whiteice
{
  namespace crypto
  {

    template <unsigned int B>
    DataConversion<B>::DataConversion(unsigned char* data, unsigned int bytes)
    {
      if(bytes == 0)
	throw std::invalid_argument("DataConversion: must have at least one byte of memory");
      else if((B % 8) != 0)
	throw std::invalid_argument("template parameter B is not divisible by 8");
      
      this->data = data;
      this->numbytes = bytes;
      
      this->cached.resize(B);
      load(0);
    }
    
    
    template <unsigned int B>
    DataConversion<B>::DataConversion(const DataConversion& dc)
    {
      this->data = dc.data;
      this->numbytes = dc.numbytes;
      
      this->cached = dc.cached;
      this->cached_location = cached_location;
    }
    
    
    template <unsigned int B>
    DataConversion<B>::~DataConversion()
    {
      flush();
    }
    
    
    template <unsigned int B>
    dynamic_bitset& DataConversion<B>::operator[](unsigned int index) throw(std::out_of_range)
    {
#ifdef DEBUG
      if(index >= size())
	throw std::out_of_range("index out of range");
#endif
      
      if(cached_location != index){
	flush(); // flushes old data back to memory
	load(index);
      }
      
      return cached;
    }
    
    
    template <unsigned int B>
    const dynamic_bitset& DataConversion<B>::operator[](unsigned int index) const throw(std::out_of_range)
    {
#ifdef DEBUG
      if(index >= size())
	throw std::out_of_range("index out of range");
#endif
      
      if(cached_location != index){
	flush(); // flushes old data back to memory
	load(index);
      }
      
      return cached;      
    }
    
    
    template <unsigned int B>
    unsigned int DataConversion<B>::size() const throw()
    {
      return ( (numbytes + (B/8) - 1) / (B/8) );
    }
    
    
    template <unsigned int B>
    bool DataConversion<B>::good() const throw()
    {
      return true;
    }
    
    template <unsigned int B>
    void DataConversion<B>::load(unsigned int index) const
    {
      unsigned int ibytes = (index * (B/8)); // index in bytes      
      const unsigned int len = numbytes - ibytes;
      
      cached.reset(); // zeroes values
      
      // copies available data into 'cache'      
      
      if(len < (B/8)){  
	for(unsigned int i=0;i<len;i++)
	  cached.value(i) = data[ibytes + i];
      }
      else{
	for(unsigned int i=0;i<(B/8);i++)
	  cached.value(i) = data[ibytes + i];
      }
      
      cached_location = index;
    }
    
    
    // flushes all transformed (cached / dirty) dynamic_bitset data
    // back to provided data memory
    template <unsigned int B>
    void DataConversion<B>::flush() const
    {
      const unsigned int beg = cached_location*(B/8);
      const unsigned int len = numbytes - beg;
      
      if(len < (B/8)){
	for(unsigned int i=0;i<len;i++)
	  data[beg + i] = cached.value(i);
      }
      else{
	for(unsigned int i=0;i<(B/8);i++)
	  data[beg + i] = cached.value(i);
      }
    }
    
    
    template <unsigned int B>
    bool DataConversion<B>::resize_memory(unsigned int bytes)
    {
      // no real resizing done by DataConversion
      
      if(bytes <= 0) return false;
      
      if(cached_location >= bytes)
	load(0);
      
      numbytes = bytes;
      
      return true;
    }
    
    
  };
};


#endif
