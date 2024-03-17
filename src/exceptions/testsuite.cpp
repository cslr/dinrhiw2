
#include <iostream>
#include <string>
#include <new>
#include <exception>
#include <stdexcept>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "uncomparable.h"

void do_uncomparable_tests();

int main()
{
  std::cout << "UNCOMPARABLE TESTS STARTS\n";
  do_uncomparable_tests();
  std::cout << "UNCOMPARABLE TESTS ENDS\n";
  
  std::cout << "WARN: OTHER TESTS NOT DONE\n";
  
  return 0;
}


void do_uncomparable_tests()
{
  // blackbox tests: tests *most* of the cases - not all.
  
  using namespace whiteice;
  
  try{
    std::string str = "thedoris";
    std::string str2 = "ABC";
    
    uncomparable u("ABC");
    uncomparable v(str);
    
    if(u.what() == 0)
      std::cout << "ERROR0: null pointer\n";

    if(v.what() == 0)
      std::cout << "ERROR0a: null pointer\n";
    
    if(strcmp(u.what(), v.what()) == 0)
      std::cout << "ERROR0b: uncomparable objects have same message\n";
    
    v = u;

    if(strcmp(u.what(), v.what()) != 0)
      std::cout << "ERROR1: uncomparable objects have different message after '=' operator \n";

    if(strcmp(v.what(), str2.c_str()) != 0)
      std::cout << "ERROR2: string with same characters doesn't match message value \n";
    
    ////////////////////////////////
    // dynamic allocated cases

    try{
      uncomparable* w = new uncomparable(str2.c_str());
      delete w;
    }
    catch(std::bad_alloc& b){
      std::cout << "ERROR3: new/delete pair failed - CTOR0\n";
    }
    
    try{
      uncomparable* w = new uncomparable(u);
      delete w;
    }
    catch(std::bad_alloc& b){
      std::cout << "ERROR4: new/delete pair failed - CTOR1\n";
    }
    
    uncomparable* w[2];
    w[0] = new uncomparable(str.c_str());
    w[1] = new uncomparable(u);
    
    if(w[0]->what() == 0)
      std::cout << "ERROR5: null pointer\n";

    if(w[1]->what() == 0)
      std::cout << "ERROR6: null pointer\n";
    
    if(strcmp(w[0]->what(), w[1]->what()) == 0)
      std::cout << "ERROR0: uncomparable objects have same message\n";
    
    *(w[1]) = *(w[0]);

    if(strcmp(w[0]->what(), w[1]->what()) != 0)
      std::cout << "ERROR6: uncomparable objects have different message after '=' operator \n";

    if(strcmp(w[1]->what(), str.c_str()) != 0)
      std::cout << "ERROR7: string with same characters doesn't match message value \n";
    
    
    delete w[0];
    delete w[1];
  }
  catch(std::exception& e){
    std::cout << "UNCOMPARABLE: ERROR8 - UNCAUGHT EXCEPTION: " << e.what() << std::endl;
  }
  
}



