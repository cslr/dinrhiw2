/*
 * implements totally
 * general 'dynamic/runtime function access control' system
 */

#ifndef function_access_control_h
#define function_access_control_h

#include <exception>
#include <stdexcept>
#include <iostream>
#include <typeinfo>
#include <string>
#include <map>
#include <time.h>

#include "ownexception.h"
#include "unique_id.h"


namespace whiteice
{
  
  template <typename T> class fac_granter;
  template <typename T> class fac_applicant;
  
  
  
  template <typename T>
    class fac
    { 
    public:
      
      fac(); // default ctor only for map etc.
             // creates invalid fac's
      
      // private: // FIXME should be private?
    public: 
      
      
      // seconds function access is valid      
      fac(const char* str, time_t secs);
      fac(const std::string& s, time_t secs);
      
      fac(const fac<T>& f); // private copy constructors
      fac<T>& operator=(const fac<T>& f);
      
      friend class fac_granter<T>;
      friend class fac_applicant<T>;
      friend class std::map< unsigned int, whiteice::fac<T> >;
      
    private:
      time_t timestamp;  // creation time
      time_t expiration; // expiration time
      
      std::string fid;  // function id
      
    public:
      bool valid() const ;
    };
  
  
  // T is class you want to get access to
  // I is inheritating class
  template <typename T>
    class fac_applicant : public whiteice::unique_id
    {
    public:
      fac_applicant();
      virtual ~fac_applicant();
      
    protected:
      
      unsigned int get_access(const char* fid,
			      const fac_granter<T>& g)
	const ;
      
      
      unsigned int get_access(const std::string& fid,
			      const fac_granter<T>& g)
	const ;
      
      
      fac<T>& auth(unsigned int id) const
	; 
      
      
      void clear_access(unsigned int id) const;
      
    private:
      
      std::string idlist;
      
      // authorizations
      mutable std::map< unsigned int, whiteice::fac<T> > authorizations;
    };
  
  
  template <typename T>
    class fac_granter
    {
    public:
      fac_granter() ;
      
    protected:
      // inheritator implements / sets
      
      // params: function id, applicant class
      // (implementator may need to use mutable
      //  to work'a'round const)
      virtual bool has_access(const std::string& fid,
			      const fac_applicant<T>& a) const
	 = 0;
      
      time_t fac_expiration_time;
      
    private:
      
      friend class fac_applicant<T>;
      
      // function id, applicant class
      fac<T> get_access(const std::string& fid,
			const fac_applicant<T>& a)
	const ;
      
    };
  
};


#include "function_access_control.cpp"

#endif

