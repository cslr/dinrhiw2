/*
 * DataConversion (data_source) provides as needed conversion
 * of table of char data to B bit dynamic_bitset:s
 */

#ifndef DataConversion_h
#define DataConversion_h

#include "dynamic_bitset.h"
#include "data_source.h"

namespace whiteice
{
  namespace crypto
  {
    // number of bits in a dynamic_bitset, must be multitiple of 8 bits !
    template <unsigned int B>
      class DataConversion : public data_source<dynamic_bitset>
      {
      public:
	DataConversion(unsigned char* data, unsigned int bytes);
	DataConversion(const DataConversion& dc);
	virtual ~DataConversion();
	
	dynamic_bitset& operator[](unsigned int index) throw(std::out_of_range);
	const dynamic_bitset& operator[](unsigned int index) const throw(std::out_of_range);
	
	// number of dynamic_bitsets available
	unsigned int size() const throw();
	
	// tells if the data source access is working correctly (-> can read more data)
	bool good() const throw();
	
	// flushes all transformed (cached & dirty) dynamic_bitset data
	// back to provided data memory
	void flush() const;
	
	// resizes accessible data memory (after realloc()/reuse etc.)
	bool resize_memory(unsigned int bytes);
	
      private:
	
	// brings given location into 'cache'
	void load(unsigned int index) const;
	
	unsigned char* data; // + cached data
	
	unsigned int numbytes;
	
	mutable dynamic_bitset cached;
	mutable unsigned int cached_location;
      };
  };
};

#include "DataConversion.cpp"

#endif
