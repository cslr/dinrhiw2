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
    
    ~ownexception_base()  { }
    
    bool operator==(const ownexception_base& e) ;
    const std::string& message() const ;
    
    const char* what() const throw() { return msg.c_str(); }
    
  protected:
    
    std::string msg;
  };
  
  
  class uncomparable : public ownexception_base
  {
  public:
    
    uncomparable() ;
    uncomparable(const std::string& s) ;
    uncomparable(const uncomparable& u) ;
    ~uncomparable()  { }
    
  };
  
  
  class illegal_operation : public ownexception_base
  {
  public:
    
    illegal_operation() ;
    illegal_operation(const std::string& s) ;
    illegal_operation(const illegal_operation& u) ;
    ~illegal_operation()  { }
  };
  
  
  class noaccess : ownexception_base
  {
  public:
    
    noaccess() ;
    noaccess(const std::string& s) ;
    noaccess(const noaccess& u) ;
      
    ~noaccess() { }
  };

  
  // for raising NVIDIA CUDA/cuBLAS Exceptions
  class CUDAException
  {
  public:
    CUDAException();
    CUDAException(const std::string& s);
    CUDAException(const CUDAException& e);
    
    bool operator==(const CUDAException& e);
    const std::string& message() const ;
    
    const char* what() const throw() { return msg.c_str(); }
    
  protected:
    
    std::string msg;
  };
  
}


#endif

