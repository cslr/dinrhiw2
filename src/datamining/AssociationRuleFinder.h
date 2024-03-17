/*
 * finds association rules from the data
 * provided by data_source
 *
 *
 * note: due implementation limitations only bitsizes up to 32 bits works now
 * todo: after tests with 32bits work change everything to use math::integer code -> no limits
 */

#ifndef AssociationRuleFinder_h
#define AssociationRuleFinder_h

#include <vector>
#include "dynamic_bitset.h"
#include "data_source.h"


namespace whiteice
{
  namespace datamining
  {
    
    struct rule {
      dynamic_bitset x, y; // X => Y
      float frequency, confidence;      
    };
    
    
    class AssociationRuleFinder
    {
    public:
      AssociationRuleFinder(const whiteice::data_source<whiteice::dynamic_bitset>& data,
			    std::vector<whiteice::datamining::rule>& rules,
			    float freq_limit, float conf_limit);
      
      ~AssociationRuleFinder();
      
      // finds (more) associative rules that can be find in TIME_LIMIT seconds
      // if TIME_LIMIT <= 0 amount of time used is unlimited.
      // returns true if any new rule could be found
      bool find(float TIME_LIMIT = -1.0f);
      
      // returns true if last call of find() finished before time limit
      // and managed to find all possible rules
      bool finished() const ;
      
      // frees all data structures not needed
      // after all rules have been found
      // (clean everything except rules)
      void clean();
      
    private:
      
      unsigned int calculate_counts(const dynamic_bitset& set) const ;
      bool internal_isfinished() const ;
      
      // input data
      
      const whiteice::data_source<whiteice::dynamic_bitset>& data;
      std::vector<whiteice::datamining::rule>& rules;
      
      const float freq_limit;
      const float conf_limit;
      
      
    private:
      // data structures used by search
      
      // frequent sets
      std::vector< std::vector<dynamic_bitset> > fset;
      
      // frequencies of frequent relations in fset
      std::vector< std::vector<unsigned int> > fcset; 
      
      
      // where the search is going: fset[level][elem]
      unsigned int level;
      unsigned int elem;
      
      bool search_finished;
    };
    
    
    
  }
}

#endif
