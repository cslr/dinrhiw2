/*
 * list_source implementation of data_source
 *
 */

#ifndef list_source_h
#define list_source_h

#include <vector>
#include <stdexcept>
#include <exception>

#include "data_source.h"


namespace whiteice
{
  template <typename datum>
    class list_source : public data_source<datum>
    {
    public:
      list_source(std::vector<datum>& list);
      virtual ~list_source();
      
      datum& operator[](unsigned int index) throw(std::out_of_range);
      const datum& operator[](unsigned int index) const throw(std::out_of_range);
      
      unsigned int size() const throw();
      
      bool good() const throw();
      
      void flush() const;
      
    private:
      std::vector<datum>& list;
      
  };
  
  
}

#include "list_source.cpp"

#endif
