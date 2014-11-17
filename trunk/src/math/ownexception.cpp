

#include "ownexception.h"

using namespace std;

namespace whiteice
{
  
  uncomparable::uncomparable() throw(){ this->msg = ""; }
  uncomparable::uncomparable(const string& s) throw(){ this->msg = s; }
  uncomparable::uncomparable(const uncomparable& u) throw(){
    this->msg = u.msg;
  }
  
  
  illegal_operation::illegal_operation() throw(){ this->msg = ""; }
  illegal_operation::illegal_operation(const std::string& s) throw(){ this->msg = s; }
  illegal_operation::illegal_operation(const illegal_operation& e) throw(){
    this->msg = e.msg;
  }
  
  
  noaccess::noaccess() throw(){ this->msg = ""; }
  noaccess::noaccess(const std::string& s) throw(){ this->msg = s; }
  noaccess::noaccess(const noaccess& a) throw(){ this->msg = a.msg; }
  
  
  bool exception::operator==(const exception& e1) throw(){
    const exception& e2 = *this;
    return (e1.msg == e2.msg);
  }
  
  const string& exception::message() const throw(){ return msg; }
  
}
  




