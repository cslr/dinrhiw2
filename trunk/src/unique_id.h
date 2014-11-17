/*
 * class provides unique identification number service
 * inherit class to get service access point
 * 
 * there's two modes of operation, either "limited" mode
 * when create()'s number > 0 and when all free numbers are listed and
 * "unlimited" mode when number == 0 and all reserved numbers are listed.
 */

#ifndef unique_id_h
#define unique_id_h

#include <set>
#include <map>
#include <string>

namespace whiteice
{
  // TODO: generalize with template to general
  // (unsigned) numbers
  
  class unique_id
  {
  protected:

    // allocates and frees id numbers, zero cannot be id
    // number -> zero means error
    virtual unsigned int get(const std::string& name) const throw();
    virtual bool free(const std::string& name, unsigned int id) const throw();
    
    // creates and frees id number domains
    // variable number is number of id numbers or zero if
    // number of free IDs are unlimited.
    // 
    // memory required is O('number of id numbers') for 1st mode
    // and O('reserved numbers') for unlimited mode
    // 
    virtual bool create(const std::string& name,
			unsigned int number) const throw();
    
    virtual bool free(const std::string& name) const throw();
    
    // unlimited (limited by memory/int size) number of id values
    virtual bool create(const std::string& name) const throw();
    
    
    // returns true if domain exists
    virtual bool exists(const std::string& name) const;

  public:

    // in order to remove warnings
    virtual ~unique_id(){ }

  private:

    // name -> set of free integers

    struct id_domain{
      unsigned int maxvalue;
      std::set<unsigned int> ids;
    };
    
    mutable std::map< std::string, id_domain> idlist;
    
  };
  
};



#endif // unique_id.h


