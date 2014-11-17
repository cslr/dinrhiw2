/*
 * tests for datamining code
 *
 */

#include <string>
#include <exception>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "AssociationRuleFinder.h"
#include "list_source.h"


void test_associationrulefinder();


using namespace whiteice;


int main()
{
  printf("DATAMINING CODE TESTS\n");
  
  srand(time(0));
  
  test_associationrulefinder();
  
  return 0;
}



void test_associationrulefinder()
{
  try{
    printf("ASSOCIATION RULE FINDER TESTS\n");
    
    
    // tests with logic operations
    // 3 input bits, 4 output bits, output bits are calculated with and/or/xor of input bits
    {
      std::vector<std::string> colnames; // names for columns of row vectors
      std::vector<dynamic_bitset> data;
      std::vector<datamining::rule> rules;
      
      list_source<dynamic_bitset>* source;
      datamining::AssociationRuleFinder* rulefinder;
      
      
      colnames.push_back("x");     // 0
      colnames.push_back("y");     // 1
      colnames.push_back("z");     // 2
      colnames.push_back("x * y"); // 3 (and)
      colnames.push_back("y * z"); // 4 (and)
      colnames.push_back("x + z"); // 5 (or)
      colnames.push_back("z");     // 6 (identity)
      
      dynamic_bitset x;
      x.resize(7);
      
      for(unsigned int i=0;i<5000;i++){
	x.reset();
	x.set(0, (bool)(rand() & 1) ); // x      50%
	x.set(1, (bool)(rand() & 1) ); // y      50%
	x.set(2, (bool)(rand() & 1) ); // z      50%
	x.set(3, x[0] & x[1] );        // x * y  25%
	x.set(4, x[1] & x[2] );        // y * z  25%
	x.set(5, x[0] | x[2] );        // x + z  25%
	x.set(6, x[2]);
	
	data.push_back(x);
      }
      
      
      printf("data size: %d\n", (int)data.size());
      
      source = new list_source<dynamic_bitset>(data);
      rulefinder = new datamining::AssociationRuleFinder(*source, rules, 0.2, 0.5);
      
      while(rulefinder->finished() == false){
	rulefinder->find(-1.0); // no time limits
	printf("total number of rules: %d\n", (int)rules.size());
      }
      
      rulefinder->clean();
      delete rulefinder;
      delete source;
      
      // shows found rules (in no specific order)
      for(unsigned int i=0;i<rules.size();i++){
	printf("RULE %3d (%3.2f , %3.2f)  ", i+1,
	       rules[i].frequency, rules[i].confidence);
	
	unsigned int k = 0;
	printf("(");
	for(unsigned int j=0;j<rules[i].x.size();j++){
	  if(rules[i].x[j]){
	    printf("%s", colnames[j].c_str());
	    
	    k++;
	  
	    if(k != rules[i].x.count())
	      printf(",");
	  }
	}
	printf(") => (");
	
	k = 0;
	for(unsigned int j=0;j<rules[i].y.size();j++){
	  if(rules[i].y[j]){
	    printf("%s", colnames[j].c_str());
	    
	    k++;
	    
	    if(k != rules[i].y.count())
	      printf(",");
	  }
	}
	
	printf(")\n");
      }
            
    }
    
    
    
    std::cout << "ASSOCIATION RULE FINDER TESTS PASSED" << std::endl;
  }
  catch(std::exception& e){
    std::cout << "Unexpected exception: " << e.what() << std::endl;
  }
}
