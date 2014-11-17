/*
 * code for edit distance measurement
 */

#include <new>
#include <string>
#include <iostream>

template <typename T> // T = string, T = vector are ok.
int editdistance(const T& s1, const T& s2);


int main(int argc, char **argv)
{
  if(argc < 3 || argc > 4){
    std::cout << "Usage: ";
    std::cout << argv[0] << " [-v] string1 string2" << std::endl;
    return -1;
  }
  
  bool verbose = false;
  std::string s[2];
  
  if(argc == 4){
    if(strcmp("-v", argv[1]) == 0)
      verbose = true;
    else{
      std::cout << "Usage: ";
      std::cout << argv[0] << " [-v] string1 string2" << std::endl;      
      return -1;
    }
      
    
    s[0] = argv[2];
    s[1] = argv[3];
  }
  else{
    s[0] = argv[1];
    s[1] = argv[2];
  }
  
  int d = editdistance<std::string>(s[0], s[1]);
  
  if(verbose)
    std::cout << "distance: " << d << std::endl;
  
  return d;
}








/*
 * calculates edit distance between
 * strings with dynamic programming
 */
template <typename T> // T = string, T = vector are ok.
int editdistance(const T& s1, const T& s2)
{
  int* table;  
  table = new int[(s1.length() + 1)*(s2.length() + 1)];

  /* initial values */
  table[0] = 0;
  
  for(unsigned int i=0;i<s1.length();i++)
    table[1 + i] = 1;
  
  for(unsigned int i=0;i<s2.length();i++)
    table[(i+1)*(s1.length()+1)] = 1;

  /* calculates distance table */
  
  for(unsigned int i=1;i<(s1.length()+1);i++){
    for(unsigned int j=1;j<(s2.length()+1);j++)
    {
      int c1 = table[(i-1) + (j-1)*(s1.length()+1)];      
      if(s1[i-1] != s2[j-1]) c1++;
      
      int c2 = table[(i-1) + j*(s1.length()+1)] + 1;
      int c3 = table[i + (j-1)*(s1.length()+1)] + 1;
      
      // chooses minimum
      if(c2 < c1) c1 = c2;
      if(c3 < c1) c1 = c3;
            
      table[i + j*(s1.length()+1)] = c1;
    }
  }
  
  int distance = table[(s1.length()+1)*(s2.length()+1) - 1];
  delete[] table;
  
  return distance;
}




