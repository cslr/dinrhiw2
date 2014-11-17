/*
 * array interface
 * Tomas Ukkonen <tomas.ukkonen@hut.fi>
 */
#ifndef array_h
#define array_h

#include <exception>
#include <stdlib.h>
#include <stdexcept>

#include "container.h"

namespace whiteice
{

  template <typename D, typename T=int>
    class array : public container<D,T>
  {
    public:
    
    virtual ~array(){ }
    
    /* resizes array to given size and returns true if size is now n
     * returning false means size is unchanged */
    virtual bool resize(const T& n) throw() = 0;
    
    /* returns nth element from array or throws out_of_range if
     * n is out of range (too large) */
    virtual D& operator[](const T& n) throw(std::out_of_range) = 0;

    /* returns nth element from array or throws out_of_range if
     * n is out of range (too large) */
    virtual const D& operator[](const T& n) const throw(std::out_of_range) = 0;
  };
  
}


#endif


