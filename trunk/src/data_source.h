/*
 * data_source interface
 *
 * this can be used as a wrapper for example to
 * provide conversion of data to cryptographic
 * algorithms when it's needed and reconverting
 * data back to "natural" format
 */

#ifndef data_source_h
#define data_source_h

#include <stdexcept>


namespace whiteice
{
  
  template <typename datum>
    class data_source
    {
    public:
      virtual ~data_source(){ }
      
      virtual datum& operator[](unsigned int index) throw(std::out_of_range) = 0;
      virtual const datum& operator[](unsigned int index) const throw(std::out_of_range) = 0;      
      
      virtual unsigned int size() const throw() = 0;
      
      // returns if reading data can success
      virtual bool good() const throw() = 0;
      
      // flushes all data to source if changed and cached
      // throws exception if something goes wrong
      virtual void flush() const = 0;
  };
  
};

#endif

