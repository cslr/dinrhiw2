

#include "ownexception.h"

using namespace std;

namespace whiteice
{
  
  uncomparable::uncomparable() 
  {
    this->msg = "";
  }
  

  uncomparable::uncomparable(const string& s) 
  {
    this->msg = s;
  }
  
  
  uncomparable::uncomparable(const uncomparable& u) 
  {
    this->msg = u.msg;
  }
  
  //////////////////////////////////////////////////
  
  illegal_operation::illegal_operation() 
  {
    this->msg = "";
  }
  
  illegal_operation::illegal_operation(const std::string& s) 
  {
    this->msg = s;
  }
  
  illegal_operation::illegal_operation(const illegal_operation& e) 
  {
    this->msg = e.msg;
  }

  //////////////////////////////////////////////////
  
  
  noaccess::noaccess() {
    this->msg = "";
  }
  
  noaccess::noaccess(const std::string& s) {
    this->msg = s;
  }
  
  noaccess::noaccess(const noaccess& u) {
    this->msg = u.msg;
  }
  
  //////////////////////////////////////////////////
  
  bool ownexception_base::operator==(const ownexception_base& e1) 
  {
    const ownexception_base& e2 = *this;
    
    return (e1.msg == e2.msg);
  }

  
  
  const string& ownexception_base::message() const 
  {
    return msg;
  }


  //////////////////////////////////////////////////

  
  CUDAException::CUDAException(){ }
  
  CUDAException::CUDAException(const std::string& s){
    this->msg = s;
  }

  
  CUDAException::CUDAException(const CUDAException& e){
    this->msg = e.msg;
  }
  
  bool CUDAException::operator==(const CUDAException& e){
    if(this->msg == e.msg) return true;
    else return false;
  }
  
  const std::string& CUDAException::message() const {
    return this->msg;
  }
  

}
  




