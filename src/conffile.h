/*
 * simple configuration file saving/loading
 *
 */

#ifndef conffile_h
#define conffile_h

#include <string>
#include <vector>
#include <map>


namespace whiteice
{

  class conffile
  {
  public:
    conffile() ;
    conffile(const std::string& file) ;
    ~conffile() ;
    
    // loads and/or saves settigns to file
    bool load(const std::string& file) ;
    bool save(const std::string& file) ;
    
    // checks if configuration has a named variable
    bool exists(const std::string& name) const ;
    
    // removes named variable
    bool remove(const std::string& name) ;
    
    // removes all variables
    bool clear() ;
    
    // gets list symbols containing data
    bool get(std::vector<std::string>& vnames) const ;
    
    // gets value(s) of variable
    bool get(const std::string& name, std::vector<int>& value) const ;
    bool get(const std::string& name, std::vector<float>& value) const ;
    bool get(const std::string& name, std::vector<std::string>& value) const ;
    
    // sets value(s) for variable - setting value over old variable
    // is possible *only* if old variables had a same type (vector size can be different)
    bool set(const std::string& name, const std::vector<int>& value) ;
    bool set(const std::string& name, const std::vector<float>& value) ;
    bool set(const std::string& name, const std::vector<std::string>& value) ;              
    
  private:        
    
    bool encode(std::string& s) const ;
    bool decode(std::string& s) const ;
    bool trim(std::string& s) const ;
    
    bool is_good_variable_name(const std::string& name) const ;
    bool is_good_string_vector(const std::vector<std::string>& name) const ;
    
    bool parse(std::string& line,
	       std::string& name,
	       std::vector<int>& i,
	       std::vector<float>& f,
	       std::vector<std::string>& s) ;
    
    // variables    
    std::map< std::string, std::vector<int> >   integers;
    std::map< std::string, std::vector<float> > floats;
    std::map< std::string, std::vector<std::string> > strings;
    
    bool verbose;
    
  };
  
};

#endif
