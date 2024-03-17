
#ifndef function_access_control_cpp
#define function_access_control_cpp

#include "function_access_control.h"


namespace whiteice
{
  
  // default ctor only for map< fac<T> >
  template <typename T>  
  fac<T>::fac()
  {
    time(&timestamp);
    
    this->expiration = 
      timestamp - 1; // -> expired
  }
  
  
  template <typename T>
  fac<T>::fac(const char* s, time_t secs)
  {
    time(&timestamp);
    this->expiration = timestamp + secs;
    this->fid = s;
  }
  
  
  template <typename T>
  fac<T>::fac(const std::string& s, time_t secs)
  {
    time(&timestamp);
    this->expiration = timestamp + secs;
    this->fid = s;
  }
  

  // private copy constructors
  template <typename T>
  fac<T>::fac(const fac<T>& f)
  {
    this->timestamp  = f.timestamp;
    this->expiration = f.expiration;
    this->fid = f.fid;
  } 
  
  
  template <typename T>
  fac<T>& fac<T>::operator=(const fac<T>& f)
  {
    timestamp = f.timestamp;
    this->expiration = f.expiration;
    this->fid = f.fid;
    
    return (*this);
  }
  
  
  template <typename T>
  bool fac<T>::valid() const 
  {
    return (expiration < time(0));
  }

  //////////////////////////////////////////////////
  
  template <typename T>
  fac_granter<T>::fac_granter() 
  {
    fac_expiration_time = 10; // safe default
    
    if(typeid(*this).before( typeid(T) )){
      std::cout << "typecheck mismatch" << std::endl;
      std::cout << "*this: "
		<< typeid(*this).name() << std::endl
		<< "given type T: "
		<< typeid(T).name() << std::endl;
      
      throw noaccess("typecheck mismatch detected");
    }
    
  }

  
  template <typename T>
  fac<T> fac_granter<T>::get_access(const std::string& fid,
				    const fac_applicant<T>& a)
    const 
  {
    try{
      // creates new fac to last up to expiration_time
      if(has_access(fid, a))
	return whiteice::fac<T>(fid, fac_expiration_time);
      else
	throw noaccess("access denied");
    }
    catch(std::exception& e){
      throw noaccess("access denied: unknown exception");
    }
  }

  
  //////////////////////////////////////////////////
  
  template <typename T>
  fac_applicant<T>::fac_applicant(){
    
    // gets id number domain: max practical limit of
    // max 1024 authorizations
    
    idlist = "authorizations";
    
    create(idlist, 1024);
  }
  
  template <typename T>
  fac_applicant<T>::~fac_applicant(){
    
    free(idlist);
  }


  template <typename T>
  unsigned int fac_applicant<T>::get_access(const char* fid,
					    const fac_granter<T>& g)
    const 
  {
    unsigned int id = get(idlist); // tries to get id number
    
    if(id == 0) 
      throw noaccess("authorization list full");
    
    authorizations[id] = 
      g.get_access(std::string(fid), *this);
    
    return id;
  }
  
  
  template <typename T>
  unsigned int fac_applicant<T>::get_access(const std::string& fid,
					    const fac_granter<T>& g)
    const 
  {
    unsigned int id = get(idlist); // tries to get id number
    
    if(id == 0) 
      throw noaccess("authorization list full");
    
    authorizations[id] = g.get_access(fid, *this);
    
    return id;
  }
  
  
  template <typename T>
  fac<T>& fac_applicant<T>::auth(unsigned int id) const
    
  {
    if(authorizations.find(id) != authorizations.end())
      return authorizations[id];
    else
      throw std::logic_error("no such authorization id");
  }
  
  
  template <typename T>
  void fac_applicant<T>::clear_access(unsigned int id) const
  {
    authorizations.erase(id);
    free(idlist, id);
  }

  //////////////////////////////////////////////////    
  
};


#endif












