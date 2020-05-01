
#include <cstdio>
#include <cstdlib>
#include <ctype.h>
#include "config.h"

#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <math.h>

#include "conffile.h"


using namespace std;


namespace whiteice
{

#ifndef HAVE_ISFINITE
  bool isfinite(float w);
#endif
  
  
  
  conffile::conffile() 
  {
    verbose = true;
  }
  
  
  conffile::conffile(const std::string& file) 
  {
    if(!load(file)){
      integers.clear();
      floats.clear();
      strings.clear();
    }
      
    verbose = true;
  }
  
  
  conffile::~conffile() 
  {
    verbose = true;
  }
  
  
  // loads and/or saves settings to file
  bool conffile::load(const std::string& filename) 
  {
    try{
      // 20*1024*1024 is a limitation of configuration file line length.
      // conffile is not meant for massive data structures
      // (aprox. more than 2 000 000 elements per line)
      // [10 char per item]
      
      ifstream file;
      //const unsigned int BUFLEN=20*1024*1024;
      //char* buffer = new char[BUFLEN];
      string line;
      
      std::vector<int> i;
      std::vector<float> f;
      std::vector<std::string> s;
      std::string str;
      
      file.open(filename.c_str());
      if(!file.is_open()) return false;
      if(!file.good()) return false;
      
      integers.clear();
      strings.clear();
      floats.clear();

      std::getline(file, line);
      
      do{
	i.clear();
	f.clear();
	s.clear();
	str = "";
	
	if(parse(line, str, i, f, s)){
	  if(!i.empty()){
	    integers[str] = i;
	  }
	  else if(!f.empty()){
	    floats[str] = f;
	  }
	  else if(!s.empty()){
	    strings[str] = s;
	  }
	}
	
	std::getline(file, line);
      }
      while(!file.eof() && file.good());
      
      file.close();

      // delete[] buffer;
      
      return true;
    }
    catch(std::exception& e){
      std::cout << "Unexpected exception: " << e.what() << std::endl;
      return false;
    }
  }
  
  
  bool conffile::save(const std::string& filename) 
  {
    try{
      ofstream file;
      string line;
      
      file.open(filename.c_str());    
      if(!file.good()) return false;

      // printf("CONFSAVE A\n"); fflush(stdout);
      
      std::map< std::string, std::vector<int> >::iterator i;
      std::map< std::string, std::vector<float> >::iterator j;
      std::map< std::string, std::vector<std::string> >::iterator k;    
      i = integers.begin();
      j = floats.begin();
      k = strings.begin();
      
      while(i != integers.end() && file.good()){
	line = i->first;
	line += " = ";
	
	std::vector<int>::iterator w =
	  i->second.begin();
	
	while(w != i->second.end()){
	  {
	    char buf[40];
	    sprintf(buf, "%d",*w);
	    line += buf;
	  }
	  
	  w++;
	  if(w != i->second.end())
	    line += ",";
	}
	
	line += ";\n";
	i++;
	
	file << line;
      }
      
      
      if(!file.good()){
	file.close();
	return false;
      }
      
      file << "\n\n";
      line = "";

      // printf("CONFSAVE B\n"); fflush(stdout);
      
      
      while(j != floats.end() && file.good()){
	line = j->first;
	line += " = ";
	
	std::vector<float>::iterator w =
	  j->second.begin();
	
	while(w != j->second.end()){
	  {
	    char buf[40];
	    if(isfinite(*w))
	      sprintf(buf, "%.16f",*w);
	    else
	      sprintf(buf, "%.16f", 0.0);
	    line += buf;
	  }
	
	  w++;
	  if(w != j->second.end())
	    line += ",";
	}      
	
	line += ";\n";
	j++;
	
	file << line;
      }
      
      if(!file.good()){
	file.close();
	return false;
      }
      
      file << "\n\n";
      line = "";

      // printf("CONFSAVE C\n"); fflush(stdout);
    
      while(k != strings.end() && file.good()){
	line = k->first;
	line += " = ";
	
	std::vector<std::string>::iterator w =
	  k->second.begin();
	
	
	
	while(w != k->second.end()){
	  
	  std::string v(*w);
	  encode(v);
	  
	  line += "\"";
	  line += v;
	  line += "\"";
	  
	  w++;
	  if(w != k->second.end())
	    line += ",";
	}
	
	line += ";\n";
	k++;
	
	file << line;      
      }
      
      if(!file.good()){
	file.close();
	return false;
      }

      // printf("CONFSAVE D\n"); fflush(stdout);
      
      file << "\n\n";
      
      file.close();
      return true;
      
    }
    catch(std::exception& e){
      std::cout << "Unexpected failure: " << e.what() << std::endl;
      return false;
    }
  }
  
  
  // checks if configuration has a named variable
  bool conffile::exists(const std::string& name) const 
  {
    std::map< std::string, std::vector<int> >::const_iterator i;
    std::map< std::string, std::vector<float> >::const_iterator j;
    std::map< std::string, std::vector<std::string> >::const_iterator k;
    
    i = integers.find(name);
    j = floats.find(name);
    k = strings.find(name);
    
    if(i != integers.end()) return true;
    if(j != floats.end()) return true;
    if(k != strings.end()) return true;
    
    return false;
  }
  
  
  // removes named variable
  bool conffile::remove(const std::string& name) 
  {
    std::map< std::string, std::vector<int> >::iterator i;
    std::map< std::string, std::vector<float> >::iterator j;
    std::map< std::string, std::vector<std::string> >::iterator k;
    
    i = integers.find(name);
    j = floats.find(name);
    k = strings.find(name);
    
    if(i != integers.end()){
      integers.erase(i);
      return true;
    }
    
    if(j != floats.end()){
      floats.erase(j);
      return true;
    }
    
    if(k != strings.end()){
      strings.erase(k);
      return true;
    }
    
    return false;    
  }
  
  
  // removes all variables
  bool conffile::clear() 
  {
    integers.clear();
    floats.clear();
    strings.clear();
    return true;
  }
  
  
  bool conffile::get(std::vector<std::string>& vnames) const 
  {
    try{
      std::map< std::string, std::vector<int> >::const_iterator ii;
      std::map< std::string, std::vector<float> >::const_iterator fi;
      std::map< std::string, std::vector<std::string> >::const_iterator si;
      
      vnames.clear();
      ii = integers.begin();
      fi = floats.begin();
      si = strings.begin();
      
      while(ii != integers.end()){
	vnames.push_back(ii->first);
	ii++;
      }
      
      while(fi != floats.end()){
	vnames.push_back(fi->first);
	fi++;
      }
      
      while(si != strings.end()){
	vnames.push_back(si->first);
	si++;
      }
      
      return true;
    }
    catch(std::exception& e){
      return false;
    }
  }
  
  
  // gets value(s) of variable
  bool conffile::get(const std::string& name,
		     std::vector<int>& value) const 
  {
    std::map< std::string, std::vector<int> >::const_iterator i;
    
    i = integers.find(name);
    if(i == integers.end()) return false;
    
    value = i->second;
    
    return true;
  }
  
  
  bool conffile::get(const std::string& name,
		     std::vector<float>& value) const 
  {
    std::map< std::string, std::vector<float> >::const_iterator j;
    
    j = floats.find(name);
    if(j == floats.end()) return false;
    
    value = j->second;
    return true;
  }
  
  
  bool conffile::get(const std::string& name,
		     std::vector<std::string>& value) const 
  {
    std::map< std::string, std::vector<std::string> >::const_iterator k;
    
    k = strings.find(name);
    if(k == strings.end()) return false;
    
    value = k->second;
    return true;
  }
  
  
  // sets value(s) for variable - setting value over old variable
  // is possible *only* if old variables had a same type (vector size can be different)
  bool conffile::set(const std::string& name,
		     const std::vector<int>& value) 
  {
    if(name.size() <= 0 || value.size() <= 0 || is_good_variable_name(name) == false)
      return false;
    
    std::map< std::string, std::vector<int> >::iterator i;
    std::map< std::string, std::vector<float> >::iterator j;
    std::map< std::string, std::vector<std::string> >::iterator k;
    
    i = integers.find(name);
    
    if(i != integers.end()){
      i->second = value;
      return true;
    }
    
    j = floats.find(name);
    k = strings.find(name);
    
    if(j != floats.end()) return false;
    if(k != strings.end()) return false;
    
    integers[name] = value;
    return true;
  }
  
  
  bool conffile::set(const std::string& name,
		     const std::vector<float>& value) 
  {
    if(name.size() <= 0 || value.size() <= 0 || is_good_variable_name(name) == false)
      return false;
    
    std::map< std::string, std::vector<int> >::iterator i;
    std::map< std::string, std::vector<float> >::iterator j;
    std::map< std::string, std::vector<std::string> >::iterator k;
    
    j = floats.find(name);
    
    if(j != floats.end()){
      j->second = value;
      return true;
    }
    
    
    i = integers.find(name);
    k = strings.find(name);
    
    if(i != integers.end()) return false;
    if(k != strings.end()) return false;
    
    floats[name] = value;
    return true;    
  }
  
  
  bool conffile::set(const std::string& name,
		     const std::vector<std::string>& value) 
  {
    if(name.size() <= 0 || value.size() <= 0 || is_good_variable_name(name) == false)
      return false;
    
    if(is_good_string_vector(value) == false)
      return false;
    
    std::map< std::string, std::vector<int> >::iterator i;
    std::map< std::string, std::vector<float> >::iterator j;
    std::map< std::string, std::vector<std::string> >::iterator k;
    
    k = strings.find(name);
    
    if(k != strings.end()){
      k->second = value;
      return true;
    }
    
    i = integers.find(name);
    j = floats.find(name);
    
    if(i != integers.end()) return false;
    if(j != floats.end()) return false;
  
    strings[name] = value;
    return true;    
  }
  
  
  
  bool conffile::parse(std::string& line,
		       std::string& name,
		       std::vector<int>& i,
		       std::vector<float>& f,
		       std::vector<std::string>& s) 
  {
    int index;
    int variable_start;
    
    std::string parse_name, parse_value;
    
    // PARSE
    //
    // goes through line
    //  either: spaces* end of line -> return false (empty line)
    //          spaces* # -> return false (comment)
    //          characters -> variable names starts
    //
    //          variable names must not containt '='
    //          or spaces at the beginning or the end
    
    for(index=0;index<(signed)line.length();index++)
    {
      if(line[index] == '#') return false;
      if(line[index] != ' ') break;
    }
    
    if(index >= (signed)line.length()) return false;
    
    // reads forward till '=' OR if end of line return false
    // (print error)
    
    for(;index<(signed)line.length();index++){
      if(line[index] == '=') break;
    }
    
    if(index >= (signed)line.length()) return false;
    
    parse_name = line.substr(0, index);
    index++;
    variable_start = index;
    
    // reads forward till ';' which isn't between '"' OR
    // if meets end of line before it return false
    {
      bool ignore_special_chars = false; // inside '"' marks
      
      for(;index<(signed)line.length();index++){
	
	
	if(line[index] == '\\'){
	  if(index + 1 < (signed)line.length()){
	    
	    if(line[index+1] == '\\') index++;
	    else if(line[index+1] == '"') index++;
	    
	    // => cannot never run into '"' when '\' is before it
	  }
	}
	else if(line[index] == '"'){ // this isn't part of the string
	  ignore_special_chars = !ignore_special_chars;
	}	
	
	if(ignore_special_chars == false && line[index] == ';')
	  break;
      }
    }
    
    if(index >= (signed)line.length()) return false;
    
    parse_value = 
      line.substr(variable_start, index - variable_start);
    
    // ANALYZE
    // remove spaces before and after
    
    trim(parse_name);
    trim(parse_value);
    
    // if variable name length <= 1 -> NOT GOOD
    
    if(parse_name.length() < 1 &&
       parse_value.length() <= 0)
    {
      if(verbose)
	cout 
	  << "conffile: " << line << endl
	  << "Bad name/value: '" 
	  << parse_name << "' '" 
	  << parse_value << "'" << endl;
      
      return false;
    }
    
    // if first and last of value = " -> string
    // if contains only numbers and , and numbers between "," -> int
    // if contains only numbers dots and , and numbers between "," -> float
    // if
    //    else return false (print error)
    
    bool isString = false, isFloat = false, isInteger = false;
    
    if(parse_value[0] == '"' &&
       parse_value[parse_value.length()-1] == '"' &&
       parse_value.length() > 1) isString = true;
    
    if(!isString){
      
      isInteger = true;
      
      for(unsigned int i=0;i<parse_value.length();i++)
      {
	if(parse_value[i] == '0' || parse_value[i] == '1' ||
	   parse_value[i] == '2' || parse_value[i] == '3' ||
	   parse_value[i] == '4' || parse_value[i] == '5' ||
	   parse_value[i] == '6' || parse_value[i] == '7' ||
	   parse_value[i] == '8' || parse_value[i] == '9')
	{
	  // ok
	}
	else if(parse_value[i] == ' '){
	  // ok
	}
	else if(parse_value[i] == '+'){
	  // ok
	}
	else if(parse_value[i] == '-'){
	  // ok
	}
	else if(parse_value[i] == ','){
	  // ok
	}
	else if(parse_value[i] == '.'){
	  isFloat = true;
	}
	else{
	  isInteger = false;
	}
      }
      
      if(isInteger){
	if(isFloat) isInteger = false;
      }
      else isFloat = false;
      
    }
    
    
    if(!isInteger && !isFloat && !isString){
      if(verbose){
	cout << "Cannot parse value '" << parse_value << "'" << endl;
	cout << line << endl;
      }
      return false;
    }
    
    // checks if variable is already defined
    {
      std::map< std::string, std::vector<int> >::iterator i;
      std::map< std::string, std::vector<float> >::iterator j;
      std::map< std::string, std::vector<std::string> >::iterator k;
      
      i = integers.find(name);
      j = floats.find(name);
      k = strings.find(name);
      
      {
	bool ok = true;
	
	if(i != integers.end() && !isInteger) ok = false;
	if(j != floats.end() && !isFloat) ok = false;
	if(k != strings.end() && !isString) ok = false;
	
	if(!ok)
	{
	  if(verbose){
	    cout << "Variable: '" + parse_name + "' already defined to be different type.";
	    return false;
	  }
	}
      }
      
    }
    
    
    // tokenizing item list
    // divide strings between "," unless they are between
    // '\"' marks. '\"' are coded as a \" within strings 
    
    int a, b;
    bool ignore_special_chars = false; // inside '"' marks
    std::vector<std::string> items;
    std::string item;
    
    a = 0;
    b = -1;
    
    {
      unsigned int i = 0;
      
      if(parse_value[0] == ',') i = 1;
      
      
      for(;i<parse_value.length();i++)
      {
	if(parse_value[i] == '\\'){
	  if(i + 1 < parse_value.length()){
	    
	    if(parse_value[i+1] == '\\') i++;	    	    
	    else if(parse_value[i+1] == '"') i++;
	    
	    // => cannot never run into '"' when '\' is before it
	  }
	}
	else if(parse_value[i] == '"'){ // this isn't part of the string
	  ignore_special_chars = !ignore_special_chars;
	}
	
	
	// if not in a string
	if(ignore_special_chars == false){
	  
	  if(parse_value[i] == ','){
	    
	    if(a >= 0){
	      b = i-1;
	      item = parse_value.substr(a, (b-a) + 1);
	      decode(item);
	      trim(item);
	      items.push_back(item);
	      a = i+1;
	    }
	  }
	  
	  if(parse_value[i] == ';') break; // done
	}
	
      }
    }
    
    
    if(ignore_special_chars == false){ // last one was mostly ok
      // adds the last element
      
      item = parse_value.substr(a);
      decode(item);
      trim(item);
      items.push_back(item);
      
    }
    
    
    // TODO strict regexp form checking for each case
    // for each case 
    // if int check there's only numbers (and maybe +/-) before any numbers and spaces -> atoi
    //    ( [ ]*[+-]?[0123456789]*[ ]* )
    // if float check there's maybe (+/-) and maybe spaces before and after it and
    //    atmost one "." and number before and after it -> atof
    //    ( [ ]*[+-]+[0123456789]+[.[0123456789]+]?[ ]* )
    // if fails print error and return false
    
    {
      std::vector<std::string>::iterator m = items.begin();
      
      while(m != items.end())
      {
	if(isInteger){
	  integers[parse_name].push_back(atoi((*m).c_str()));
	}
	else if(isFloat){
	  floats[parse_name].push_back(atof((*m).c_str()));
	}
	else if(isString)
	{
	  int end = m->length();
	  if(end < 2){
	    if(verbose)
	      std::cout << "Bad string item: '" << *m << "'\n";
	    m++; continue; }
	  
	  if((*m)[0] == '"' && (*m)[end-1] == '"'){
	    strings[parse_name].push_back( m->substr(1, end - 2) );
	  }
	}
	
	m++;
      }
    }
    
    return true;
  }
  
  
  // encodes string into a format, where '"' and 
  // '\' are coded with two characters
  bool conffile::encode(std::string& s) const 
  {
    std::string u = s;
    
    unsigned int j = 0;
    unsigned int s_size = s.size();
    
    for(unsigned int i=0;i<u.size();i++,j++){
      
      if(u[i] == '\\'){
	s_size++;
	s.resize(s_size);
	s[j] = '\\';
	j++;
      }
      else if(u[i] == '"'){
	s_size++;
	s.resize(s_size);
	s[j] = '\\';
	j++;
      }
      
      s[j] = u[i];
    }
    
    return true;
  }


  // encodes string from a format, where '"' and 
  // '\' are handled specially (inverse of encode)
  bool conffile::decode(std::string& s) const 
  {
    // replaces \\ -> \ and \" -> "
    
    if(s.size() > 1){
      unsigned int j = 0;
      unsigned int i = 0;
      
      for(i=0;i<s.size();i++,j++){
	if(s[i] == '\\'){
	  if(i + 1 < s.size())
	    if(s[i+1] == '\\' || s[i+1] == '"')
	      i++;
	}
	
	s[j] = s[i];
      }
      
      s.resize(j);
      
      
    }
    
    return true;
  }
  
  
  // removes extra spaces at the beginning and end of the line
  bool conffile::trim(std::string& s) const 
  {
    if(s.length() <= 0) return true;
    
    {
      int a = 0, b = 0;
      
      for(int i=0;i<(signed)s.length();i++){
	if(s[i] == ' ')
	  a = i+1;
	else
	  break;
      }
      
      b = s.length();
      int i = (signed int)s.length() - 1;
      
      for(;i>=0;i--)
      {
	if(s[i] != ' ')
	  break;
	
	b = i;
      }
      
      s = s.substr(a, b-a);
    }

    return true;
  }
  
  
  bool conffile::is_good_variable_name(const std::string& name) const 
  {
    for(unsigned int i=0;i<name.size();i++){
      if(!isalpha(name[i]) && !isdigit(name[i]) && name[i] != '_')
	return false;
    }
    
    return true;
  }
  
  
  bool conffile::is_good_string_vector(const std::vector<std::string>& name) const 
  {
    std::vector<std::string>::const_iterator i = name.begin();
    
    while(i != name.end()){
      
      for(unsigned int j=0;j<i->size();j++){
	if( !isprint( (*i)[j] ) ) return false;
      }
      
      i++;
    }
    
    return true;
  }
  
  
  
#ifndef HAVE_ISFINITE
  bool isfinite(float w){
    return ( (((*((unsigned int*)&w)) >> 23) & 0xFF) == 0xFF );
  }
#endif
  
  
}

