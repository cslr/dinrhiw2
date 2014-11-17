/*
 * exception interface
 */

#ifndef ownexception_h
#define ownexception_h

#include <exception>
#include <string>

namespace whiteice
{
  class ownexception_base : public std::exception
    {
    public:
      
      ~ownexception_base() throw() { }
      
      bool operator==(const ownexception_base& e) throw();
      const std::string& message() const throw();
      
      const char* what() const throw() { return msg.c_str(); }
      
    protected:
      
      std::string msg;
    };
  
  
  class uncomparable : public ownexception_base
    {
    public:
      
      uncomparable() throw();
      uncomparable(const std::string& s) throw();
      uncomparable(const uncomparable& u) throw();
      ~uncomparable() throw() { }
      
    };
  
  
  class illegal_operation : public ownexception_base
    {
    public:
      
      illegal_operation() throw();
      illegal_operation(const std::string& s) throw();
      illegal_operation(const illegal_operation& u) throw();
      ~illegal_operation() throw() { }
    };
  
  
  class noaccess : ownexception_base
    {
    public:
      
      noaccess() throw();
      noaccess(const std::string& s) throw();
      noaccess(const noaccess& u) throw();
      
      ~noaccess() throw(){ }
    };
  
}


#endif

