/*
 * queue interface
 * Tomas Ukkonen <tomas.ukkonen@iki.fi>
 */
#ifndef queue_h
#define queue_h

#include <exception>
#include <stdexcept>
 
#include "container.h"

namespace whiteice
{
  template <typename D, typename T=int>
    class queue : public container<D,T>
  {
    public:
    
    virtual ~queue(){ }
    
    /* returns true if data has been succesfully put the the queue,
     * otherwise returns false */
    virtual bool enqueue(const D& data)  = 0;
    
    /* gets oldest data from stack or throws logic_error
     * exception if queue is empty */
    virtual D dequeue()  = 0;
  };

}
  
#endif

 
 
