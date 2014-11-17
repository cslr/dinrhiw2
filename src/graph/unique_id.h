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

namespace blade
{
  class unique_id
  {
  protected:

    // allocates and frees id numbers
    virtual unsigned int get(const std::string& name) throw();
    virtual bool free(const std::string& name, unsigned int id) throw();
    
    // creates and frees id number domains
    virtual bool create(const std::string& name, unsigned int number) throw();
    virtual bool free(const std::string& name) throw();
    
    // unlimited (limited by memory/int size) number of id values
    virtual bool create(const std::string& name) throw();
    
    
    // returns true if domain exists
    virtual bool exists(const std::string& name);

  public:

    // in order to remove warnings
    virtual ~unique_id(){ }

  private:

    // name -> set of free integers

    struct id_domain{
      unsigned int maxvalue;
      std::set<unsigned int> ids;
    };
    
    std::map< std::string, id_domain> idlist;
    
  };
  
};



#endif // unique_id.h


