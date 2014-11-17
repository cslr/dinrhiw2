
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
    
    ~exception() throw() { }
    
    bool operator==(const whiteice::exception& e) throw();
    const std::string& message() const throw();
    
    const char* what() const throw() { return msg.c_str(); }
    
  protected:
    
    std::string msg;
  };
    
    
  class uncomparable : public whiteice::exception
  {
  public:
    
    uncomparable() throw();
    uncomparable(const std::string& s) throw();
    uncomparable(const uncomparable& u) throw();
    ~uncomparable() throw() { }
    
  };
  
  
  class illegal_operation : public whiteice::exception
  {
  public:
      
    illegal_operation() throw();
    illegal_operation(const std::string& s) throw();
    illegal_operation(const illegal_operation& u) throw();
    ~illegal_operation() throw() { }
  };
  
  
  /*
   * thrown by access control mechanisms 
   */
  class noaccess : public whiteice::exception
  {
  public:
    noaccess() throw();
    noaccess(const std::string& s) throw();
    noaccess(const noaccess& a) throw();
    ~noaccess() throw(){ }
  };
  
}
  

#endif

