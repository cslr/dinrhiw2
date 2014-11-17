/*
 * tests 'modifier' classes,
 * general frameworks etc.
 *
 */

#include <iostream>
#include <string>
#include <exception>
#include <stdexcept>
#include <vector>

#include "singleton.h"
#include "function_access_control.h"


class single : public whiteice::singleton<single>
{
public:
  
  single()
  {
    std::cout << "single created" << std::endl;
  }
  
  
  ~single() throw()
  {
    std::cout << "single destroyed" << std::endl;
  }
  
  const std::string& get() throw(){
    return name;
  }
  
  
  bool set(const char* str) throw(){
    this->name = str;
    std::cout << "name changed - new name: "
	      << str << std::endl;
    
    return true;    
  }
  
  bool set(std::string& name) throw(){
    this->name = name;
    std::cout << "name changed - new name: "
	      << name << std::endl;
    
    return true;
  }
  
private:
  std::string name;
};



class server : public whiteice::fac_granter<server>
{  
public:
  server()
  {
    std::cout << "server created\n";
    this->fac_expiration_time = 15; // 15 secs
  }
    
  virtual ~server(){
    std::cout << "server destroyed\n";
  }
  
  bool request(const std::vector<int>& params,
	       const whiteice::fac<server>& auth) throw()
  {
    if(!auth.valid()) return false;
    
    std::cout << "REQUEST AUTHORIZATION IS VALID"
	      << std::endl;
    
    for(unsigned int i=0;i<params.size();i++){
      std::cout << params[i] << std::endl;
    }
    
    return true;
  }
  
private:
  bool has_access(const std::string& fid,
		  const whiteice::fac_applicant<server>& a) 
    const throw()
  {
    
    if(fid != "request") return false;
    else{
      // inspect applicant a
      std::cout << "server grants access to authorization\n";
      return true;
    }
  }
};



class client : public whiteice::fac_applicant<server>
{
public:
  client(){ std::cout << "client created\n"; }
  ~client(){ std::cout << "client destroyed\n"; }
  
  bool do_request(const std::vector<int>& params,
		  server& s)
  {
    try{
      unsigned int id = get_access("request", s);
      
      s.request(params, auth(id));
      
      clear_access(id);
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "ERROR: unexpected exception: "
		<< e.what() << std::endl;
      return false;
    }
  }

private:
};


void singleton_tests();
void fac_tests();


int main()
{
  using namespace whiteice;
  
  std::cout << "MOD TESTS" << std::endl;
  
  singleton_tests();
  
  fac_tests();    
  
}



void fac_tests()
{
  std::cout << "FUNCTION ACCESS CONTROL TESTS BEGIN" << std::endl;
  
  try{
    
    server corba_server;
    client corba_client;
    
    std::vector<int> params;
    params.push_back(100);
    params.push_back(10);
    params.push_back(1);
    
    if(corba_client.do_request(params,corba_server) == false){
      std::cout << "ERROR: authorized call denied" << std::endl;
    }
  }
  catch(std::exception& e){
    std::cout << "ERROR unexpected exception thrown: "
	      << e.what() << std::endl;
  }
  
  
  std::cout << "FUNCTION ACCESS CONTROL TESTS END" << std::endl;
}



void singleton_tests()
{
  std::cout << "SINGLETON TESTS BEGIN" << std::endl;

  // create and destroy test
  try{
    single::instance().set("ripley");
    single::destroy();
  }
  catch(std::exception& e){
    std::cout << "ERROR: bad singleton create attempt"
	      << std::endl;
  }
  
  // create + 'new' create test
  try{
    single::instance().set("marcus");
    new single;
    
    std::cout << 
      "ERROR: creation of singleton via new method failed"
	      << std::endl;
      
  }
  catch(std::logic_error& e){
    // ok
  }
  catch(std::exception& e){
    std::cout << "ERROR unexpected exception thrown: "
	      << e.what() << std::endl;
  }
  
  single::destroy();
  
  
  // 'new create' test
  try{
    new single;
    
    std::cout << 
      "ERROR: creation of singleton via new method failed"
	      << std::endl;
      
  }
  catch(std::logic_error& e){
    // ok
  }
  catch(std::exception& e){
    std::cout << "ERROR unexpected exception thrown: "
	      << e.what() << std::endl;
  }
  
  single::destroy();
  
  std::cout << "SINGLETON TESTS END" << std::endl;
}






