/*
 * Additional interface for sending informative messages to caller of 
 * function(params, LoggingInterface* li)
 *
 */

#ifndef __LoggingInterface_h
#define __LoggingInterface_h

namespace whiteice {


  class LoggingInterface {
  public:

    // prints message. strbuf must be null stopped string
    virtual bool printMessage(const char* strbuf) = 0;
    
  };
  
  
};


#endif
