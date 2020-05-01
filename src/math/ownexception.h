
/*
 * all exceptions in one file, kind of bad.. 
 */

#ifndef ownexception_h
#define ownexception_h

#include <exception>
#include <string>

namespace whiteice
{
  class exception : public std::exception
  {
  public:
    
    ~exception()  { }
    
    bool operator==(const whiteice::exception& e) ;
    const std::string& message() const ;
    
    const char* what() const throw()  { return msg.c_str(); }
    
  protected:
    
    std::string msg;
  };
    
    
  class uncomparable : public whiteice::exception
  {
  public:
    
    uncomparable() ;
    uncomparable(const std::string& s) ;
    uncomparable(const uncomparable& u) ;
    ~uncomparable()  { }
    
  };
  
  
  class illegal_operation : public whiteice::exception
  {
  public:
      
    illegal_operation() ;
    illegal_operation(const std::string& s) ;
    illegal_operation(const illegal_operation& u) ;
    ~illegal_operation()  { }
  };
  
  
  /*
   * thrown by access control mechanisms 
   */
  class noaccess : public whiteice::exception
  {
  public:
    noaccess() ;
    noaccess(const std::string& s) ;
    noaccess(const noaccess& a) ;
    ~noaccess() { }
  };
  
}
  

#endif

