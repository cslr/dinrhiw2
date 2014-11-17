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
    conffile() throw();
    conffile(const std::string& file) throw();
    ~conffile() throw();
    
    // loads and/or saves settigns to file
    bool load(const std::string& file) throw();
    bool save(const std::string& file) throw();
    
    // checks if configuration has a named variable
    bool exists(const std::string& name) const throw();
    
    // removes named variable
    bool remove(const std::string& name) throw();
    
    // removes all variables
    bool clear() throw();
    
    // gets list symbols containing data
    bool get(std::vector<std::string>& vnames) const throw();
    
    // gets value(s) of variable
    bool get(const std::string& name, std::vector<int>& value) const throw();
    bool get(const std::string& name, std::vector<float>& value) const throw();
    bool get(const std::string& name, std::vector<std::string>& value) const throw();
    
    // sets value(s) for variable - setting value over old variable
    // is possible *only* if old variables had a same type (vector size can be different)
    bool set(const std::string& name, const std::vector<int>& value) throw();
    bool set(const std::string& name, const std::vector<float>& value) throw();
    bool set(const std::string& name, const std::vector<std::string>& value) throw();              
    
  private:        
    
    bool encode(std::string& s) const throw();
    bool decode(std::string& s) const throw();
    bool trim(std::string& s) const throw();
    
    bool is_good_variable_name(const std::string& name) const throw();
    bool is_good_string_vector(const std::vector<std::string>& name) const throw();
    
    bool parse(std::string& line,
	       std::string& name,
	       std::vector<int>& i,
	       std::vector<float>& f,
	       std::vector<std::string>& s) throw();
    
    // variables    
    std::map< std::string, std::vector<int> >   integers;
    std::map< std::string, std::vector<float> > floats;
    std::map< std::string, std::vector<std::string> > strings;
    
    bool verbose;
    
  };
  
};

#endif
