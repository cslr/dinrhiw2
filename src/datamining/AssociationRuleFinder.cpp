

#include <vector>
#include <stdexcept>

#include "dynamic_bitset.h"
#include "data_source.h"
#include "fast_radixv.h"

#include "AssociationRuleFinder.h"
#include "timed_boolean.h"


namespace whiteice
{
  namespace datamining
  {
    
    
    AssociationRuleFinder::
    AssociationRuleFinder(const whiteice::data_source<whiteice::dynamic_bitset>& _data,
			  std::vector<whiteice::datamining::rule>& _rules,
			  float _freq_limit, float _conf_limit) :
      data(_data), rules(_rules), freq_limit(_freq_limit), conf_limit(_conf_limit)
    {
      rules.clear(); // clears rules at the initialization
      if(freq_limit < 0 || freq_limit > 1 ||  conf_limit < 0 || conf_limit > 1)
      {
	std::string s;
	s = "AssociationRuleFinder Ctor: ";
	s = s + "freq_limit and conf_limit must be within [0,1] interval";
	
	throw std::invalid_argument(s);
      }
      
      // current state of analysis/search
      level = 0; elem = 0;
      
      
      // finished is needed to tell that search was finished (finished()-call)
      // after clean() has removed internal data structures
      search_finished = false;
    }
    
    
    AssociationRuleFinder::~AssociationRuleFinder(){ }
    
    
    
    // finds (more) associative rules that can be found in TIME_LIMIT seconds
    // if TIME_LIMIT <= 0 amount of time used is unlimited.
    // returns true if any new rule could be found or false if no new rules
    // were added. Note that returning false may complite the search
    // (finished() returns true).
    bool AssociationRuleFinder::find(float TIME_LIMIT)
    {
      if(data.size() < 0) return false;
      
      timed_boolean timeout(TIME_LIMIT, false);
      unsigned int numrules = 0; // number of new rules found
      unsigned int numgroups = 0; // number of good sets found
      
      // converts frequency limit to number of occurances limit
      unsigned int nflimit = (unsigned int)(data.size() * freq_limit);
      
      const unsigned int M = data[0].size(); // number of bits in data
      
      
      // frequent set calculation (algo 2.14)
      // creating candidate set and checking for freq_limit (database pass)
      // has been combined here
      {
	fast_radixv<dynamic_bitset> sorter;
	dynamic_bitset zero, mask, z;
	zero.resize(data[0].size()); zero.reset();
	mask.resize(data[0].size());

	// fset = frequent sets
	// fcset = frequencies of frequent relations in fset
      
	unsigned int l = fset.size();
	      	
	while(timeout == false){
	  
	  if(fset.empty()){
	    // creates F1 set
	    
	    fset.resize(1);
	    fcset.resize(1);
	    
	    dynamic_bitset db; 
	    db.resize(M); db.reset();
	    
	    
	    // puts candidates into set from
	    // the smallest to the biggest
	    db.set(0);	  
	    
	    {
	      unsigned int N = calculate_counts(db);
	      if(N > nflimit){
		fset[0].push_back(db);
		fcset[0].push_back(N);
	      }
	    }
	    
	      
	    for(unsigned int i=1;i<M;i++){
	      db.reset(i-1);
	      db.set(i);
	      
	      unsigned int N = calculate_counts(db);
	      if(N > nflimit){
		fset[0].push_back(db);
		fcset[0].push_back(N);
	      }
	    }
	    
	    l++;
	  }
	  else{
	    if(fset[l-1].empty())
	      break; // all frequent sets have been generated
	    
	    fset.resize(l+1);
	    fcset.resize(l+1);
	    
	    mask.reset(); mask.set(M - 1); // mask
	    // sorts previous set (fast_radix)
	    sorter.sort(fset[l-1], mask, zero);
	    
	    // algo 2.18, previous (candidate set) should be in sorted order.
	    
	    for(unsigned int i=0;i<fset[l-1].size();i++){
	      for(unsigned int j=i+1;j<fset[l-1].size();j++){
		
		
		// check i and j share their l-1:th lowest on bits
		bool ok = true;
		
		{
		  unsigned int m = 0, k = l-1;
		  while(k > 0 && ok){
		    if(fset[l-1][i][m] != fset[l-1][j][m]){
		      ok = false; // ... don't share
		      break;
		    }
		    
		    
		    if(fset[l-1][i][m]) k--;
		    
		    m++;
		    
		    if(m == M && k > 0)
		      ok = false; // ... don't share		    		    
		  }
		}				
		
		if(ok){
		  dynamic_bitset w;
		  
		  z = fset[l-1][i] | fset[l-1][j];
		  
		  // checks if all z's subsets of size 'l' are
		  // in previous set
		  
		  unsigned int counter = 0;
		  
		  for(unsigned int k=0;k<fset[l-1].size();k++){
		    
		    w = z & fset[l-1][k];
		    
		    // ('l' common bits -> subset with size 'l' is in previous set)
		    if(w.count() >= l)
		      counter++;
		    
		  }
		  
		  
		  if(counter != l+1) // didn't find all subsets
		    ok = false;
		  
		  
		  // adds it to next set if set freq is bigger that fr_limit
		  // todo: optimize to calculate frequencies in a single pass
		  if(ok){
		    unsigned int N = calculate_counts(z);
		    if(N > nflimit){
		      fset[l].push_back(z);
		      fcset[l].push_back(N);
		      numgroups++;
		    }
		  }
		  
		}
	      }
	    }
	    
	    l++;
	  }
	  
	}
      }
      
      
      
      // checks if any new relations were found at all
      if(level < fset.size())
	if(numgroups == 0)
	  level = fset.size(); // no new groups -> no data to preprocess
      
      
      // rule generation (latter part of algo 2.9)
      {
	dynamic_bitset y, w;	
	y.resize(data[0].size());	
	whiteice::datamining::rule r;
	
	
	while(timeout == false && level < fset.size() - 1){
	  
	  // generate all subsets of fset[level][elem]	  
	  
	  // preparation ..
	  const unsigned int MAX = fset[level][elem].count();
	  const unsigned int MAXVALUE = 1 << MAX; //E = (2**MAX)
	  
	  // tells where the bits on in the fset[level][elem] is
	  std::vector<unsigned int> on_bits;
	  
	  on_bits.resize(MAX);
	  
	  {
	    unsigned int n = 0;
	    for(unsigned int i=0;i<MAX;i++){
	      while(fset[level][elem][n] == false)
		n++;
	      
	      on_bits[i] = n;
	      n++;
	    }
	  }
	  
	  
	  // calculates up to MAXVALUE with 'the on bits of fset[level][elem]'
	  for(unsigned int i=1;i<MAXVALUE;i++){
	    unsigned int j = i, k = 0;
	    y.reset();
	    
	    while(j != 0){
	      if(j & 1) y.set(on_bits[k]);
	      k++;
	      j >>= 1;
	    }
	    
	    // calculate frequency of subset -> confidence
	    
	    w = fset[level][elem] ^ y; // xors with subset -> w = x \ y => y
	    
	    if(w.count() == 0) continue; // don't add {} -> {list..} rules
	    
	    float fwy = (float)calculate_counts(fset[level][elem]);
	    float fw = (float)calculate_counts(w);
	    float conf = fwy/fw; // P(Y,X)/P(X) = P(Y|X) = f(Y,X)/f(X)

	    if(conf >= conf_limit){ // P(Y|X) >= conf_limit
	      r.x = w;
	      r.y = y;
	      r.frequency = ((float)fcset[level][elem])/((float)data.size());
	      r.confidence = conf;
	      rules.push_back(r);
	      numrules++;
	    }
	  }
	  
	  	  
	  elem++;
	  /*
	  if(elem >= 1)
	    while(fset[level][elem] == fset[level][elem-1])
	      elem++; // skips dupes
	  */
	  
	  if(elem >= fset[level].size()){
	    elem = 0;
	    level++;
	  }
	}
	
      }
      
      
      if(internal_isfinished())
	search_finished = true;
	
      
      return (numrules > 0);
    }
    
    
    bool AssociationRuleFinder::finished() const 
    {
      return search_finished;
    }
    
    
    void AssociationRuleFinder::clean()
    {
      fset.clear();
      fcset.clear();
    }
    
    
    /************************************************************/
    
    
    // calculates number of times given pattern (subset) exists in data
    unsigned int AssociationRuleFinder::calculate_counts(const dynamic_bitset& bset) const 
    {
      unsigned int c = 0;
      unsigned int bsetc = bset.count();

      for(unsigned int i=0;i<data.size();i++)
	if((bset & data[i]).count() == bsetc) c++;
	  
      
      return c;
    }
    
    
    // returns true if last call of find() finished before time limit
    // and managed to find all possible rules
    bool AssociationRuleFinder::internal_isfinished() const 
    {
      // fset generation:
      if(fset.size() == 0) return false;
      if(fset[fset.size() - 1].size() != 0) return false;
      
      return (level+1 >= fset.size());
    }
    
    
  }
}



