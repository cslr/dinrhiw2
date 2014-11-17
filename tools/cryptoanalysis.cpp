/*
 * cryptoanalysis for simple & naive substitution 
 * cryptography
 *
 * DOESN'T SEEM TO WORK VERY WELL
 * 
 * solves naive cryptoanalysis problem
 * (assuming direct alphabet mapping (substitution),
 *  (or almost direct when handling sequencies))
 *
 * calculates distribution of single
 * character data from ascii
 * 
 * in practice one should use datamining
 * algorithms(*) to find out sequencies
 * which probability > F and calculate
 * then probabilities of these common
 * sequencies and their subsequencies.
 * probability of words can be then calculated
 * by using always longest possible sequence
 * available.
 * 
 * (*) this has been studied extensively and 
 *     there are fast algorithms for this 
 * 
 * uses genetic algorithm based optimization
 * to solve combinational problem for finding
 * ordering which minimizes error/difference
 * between vectors.
 */



#include <dinrhiw.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string>
#include <vector>
#include <assert.h>
#include <map>


using namespace whiteice;


bool text_analyze(std::string& filename,
		  std::map<std::string, float>& p);

void print_file(std::string& cryptedfile,
		std::map<std::string, std::string>& mapping);


class GAReorderProblem : public function<dynamic_bitset, float>
{
public:
  GAReorderProblem(std::vector<float>* target,
		   std::vector<float>* source)
  {
    this->target = target;
    this->source = source;
    
    assert(this->target->size() == this->source->size());

    numberBits = (unsigned int)ceil(logf((float)this->target->size()));
    numBits = numberBits*(this->target->size());
  }
  
  
  GAReorderProblem(const GAReorderProblem& p)
  {
    this->target = p.target;
    this->source = p.source;
    this->numBits = p.numBits;
    this->numberBits = p.numberBits;
  }
  
  
  virtual ~GAReorderProblem()
  {
    this->target = 0;
    this->source = 0;
  }
  
  
  // calculates value of function
  float operator() (const dynamic_bitset& x) const{
    float y;
    calculate(x, y);
    return y;
  }
  
  
  // calculates value
  float calculate(const dynamic_bitset& x) const {
    float y;
    calculate(x, y);
    return y;
  }
  
  
  void calculate(const dynamic_bitset& x, float& y) const {
    std::vector<unsigned int> reordering;
    
    if(!toOrdering(x, reordering)){
      y = 1000.0f;
      return;
    }
    
    y = 0.0f;
    
    for(unsigned int i=0;i<source->size();i++)
      y += math::abs((*target)[reordering[i]] - (*source)[i]);
    
  }
  
      
  // creates copy of object
  function<dynamic_bitset, float>* clone() const{
    return new GAReorderProblem(*this);
  }
  
  
  // returns number of bits required by representing
  // all possible reorderings
  unsigned int bitSize(){ return numBits; }
  
  
  // converts reordering to bitset
  bool toBitset(const std::vector<unsigned int>& reordering,
		dynamic_bitset& bits) const 
  {
    if(reordering.size() != target->size())
      return false;
    
    bits.resize(numBits);
    
    for(unsigned int i=0;i<reordering.size();i++){
      for(unsigned int j=0;j<numberBits;j++)
	bits.set(j+i*numberBits, (bool)( (reordering[i] >> j) & 1 ) );
    }
    
    return true;
  }
  
  
  // converts bitset to reordering
  bool toOrdering(const dynamic_bitset& bits,
		  std::vector<unsigned int>& reordering) const {
    if(bits.size() != numBits)
      return false;
    
    reordering.resize(target->size());
    
    for(unsigned int i=0;i<reordering.size();i++){
      reordering[i] = 0;
      for(unsigned int j=0;j<numberBits;j++){
	if(bits[j+i*numberBits])
	  reordering[i] |= (1 << j);
      }
      
      if(reordering[i] >= target->size())
	reordering[i] = target->size() - 1;
      
    }
    
    return true;
  }
  
private:
  std::vector<float>* target;
  std::vector<float>* source;
  
  unsigned int numBits;
  unsigned int numberBits; // bits per single number
};




int main(int argc, char ** argv)
{
  
  if(argc != 3) return -1;
  
  std::string cryptedfile = argv[1];
  
  // correct text statistics
  std::string samplefile  = argv[2];
  
  std::map<std::string, float> pc;
  std::map<std::string, float> ps;
  
  if(!text_analyze(cryptedfile, pc))
    return -1;
  
  if(!text_analyze(samplefile, ps))
    return -1;
  
  std::cout << "crypted: " << pc.size() << std::endl;
  std::cout << "target : " << ps.size() << std::endl;
  
  {
    // calculates number of mappings
    // 
    // number of mappings:
    // if PS >= PC:   PS! / (PS - PC)!
    // if PC >= PS:   PC! / (PC - PS)!
    // 
    
    if(pc.size() < 1000000.0 && ps.size() < 1000000.0){
      // with modern cryptography problem sizes are something like 2^64 ...
      std::cout << "weak problem" << std::endl;
    }
    
    if(ps.size() > pc.size()){
      float e = 0.0f;
      
      for(unsigned int i=(ps.size() - pc.size() + 1);i<ps.size();i++)
	e += logf((float)i);
      
      e /= logf(10.0f);
      
      std::cout << "mappings: 10e" << ceil(e) << std::endl;
    }
    else{
      float e = 0.0f;
      
      for(unsigned int i=(pc.size() - ps.size() + 1);i<pc.size();i++)
	e += logf((float)i);
      
      e /= logf(10.0f);
      
      std::cout << "mappings: 10e" << ceil(e) << std::endl;
    }
  }
  
  std::cout << std::endl;
  
  // calculates variance of probabilities
  float ps_var = 0.0f, pc_var = 0.0f;
  {
    float ps_mean = 1.0f/((float)ps.size());
    float pc_mean = 1.0f/((float)pc.size());
    
    std::map<std::string, float>::iterator i;
    
    i = pc.begin();
    while(i != pc.end()){
      pc_var += (i->second - pc_mean)*(i->second - pc_mean);
      i++;
    }
    pc_var /= (float)pc.size();
    
    i = ps.begin();
    while(i != ps.end()){
      ps_var += (i->second - ps_mean)*(i->second - ps_mean);
      i++;
    }
    ps_var /= (float)ps.size();
  }
  
  std::cout << "pc var: " << pc_var << std::endl;
  std::cout << "ps var: " << ps_var << std::endl;
  
  
  std::vector<float> pcv, psv;
  
  {
    std::map<std::string, float>::iterator iter;
    iter = ps.begin();
    
    for(unsigned int i=0;i<ps.size();i++, iter++)
      psv.push_back(iter->second);
    
    iter = pc.begin();
  
    for(unsigned int i=0;i<pc.size();i++, iter++)
      pcv.push_back(iter->second);
  }
  
  
  // probability model/error is too simple to optimization
  // gives wrong results
  
  GAReorderProblem reorderError(&psv, &pcv);
  GA optimizer(reorderError.bitSize(), reorderError);
  optimizer.verbosity(true);
  
  if(!optimizer.minimize(100, 100))
    std::cout << "optimization failed" << std::endl;

  
  // mapping from crypted text to non-crypted text
  std::map<std::string, std::string> mapping;
  
  {
    dynamic_bitset bits;
    std::cout << "error: " << optimizer.getBest(bits) << std::endl;
    
    std::vector<unsigned int> result;
    std::map<std::string, float>::iterator iter1, iter2;
    
    if(reorderError.toOrdering(bits, result)){
      iter1 = pc.begin();
      
      for(unsigned int i=0;i<result.size();i++,iter1++, iter2++){
	iter2 = ps.begin();
	for(unsigned int j=0;j<result[i];j++) iter2++;	  
	
	mapping[iter1->first] = iter2->first;
	
	std::cout << iter1->first << " => " << iter2->first << std::endl;
      }
    }
    
    
    iter1 = pc.begin();
    iter2 = ps.begin();
    while(iter1 != pc.end()){
      std::cout << iter1->first << " = " << iter1->second << "   | ";
      std::cout << iter2->first << " = " << iter2->second << std::endl;
      iter1++;
      iter2++;
    }
    
  }
  
  
  
  print_file(cryptedfile, mapping);
  
  
  return 0;
}










class Boolean
{
public:
  bool value;
  
  Boolean(){ value = false; }
  Boolean(const Boolean& b){ value = b.value; }
  Boolean(bool v){ value = v; }
  
  Boolean operator=(const Boolean& b){
    value = b.value;
    return *this;
  }
  
  Boolean operator=(bool q){
    value = q;
    return *this;
  }
};


/* calculates probability distribution of
 * single characters (for now)
 */
bool text_analyze(std::string& filename,
		  std::map<std::string, float>& p)
{
  FILE* fp = fopen(filename.c_str(), "rt");
  
  if(fp == 0) return false;
  
  char buf[80];
  std::map<std::string, Boolean> bmap;
  p.clear();
  
  float total = 0.0f; // total number of characters
  
  while(!feof(fp) && !ferror(fp)){
    fgets(buf, 80, fp);
    char* ptr = buf;
    
    while(*ptr != '\0'){
      
      // problem specific hack. uppercase it.
      *ptr = toupper(*ptr);
      
      if(isprint(*ptr) && isalpha(*ptr)){
	char tmp[2] = { *ptr, '\0' };
	total++;
	
	if(bmap[tmp].value)
	  p[tmp]++;
	else{
	  bmap[tmp] = true;
	  p[tmp] = 1.0f;
	}
      }
      
      ptr++;
    }
  }
  
  fclose(fp);
  
  std::map<std::string, float>::iterator iter;
  iter = p.begin();
  
  // converts counts to frequencies
  while(iter != p.end()){
    iter->second /= total;
    iter++;
  }
  
  return true;
}



void print_file(std::string& cryptedfile,
		std::map<std::string, std::string>& mapping)
{
  FILE* fp = fopen(cryptedfile.c_str(), "rt");
  
  if(fp == 0) return;
  
  char buf[80];
  
  while(!feof(fp) && !ferror(fp)){
    fgets(buf, 80, fp);
    char* ptr = buf;
    
    while(*ptr != '\0'){
      // problem specific hack. uppercase it.
      *ptr = toupper(*ptr);
      
      std::map<std::string,std::string>::iterator iter;
      char tmp[2] = { *ptr, '\0' };
      
      iter = mapping.find(tmp);
      
      if(iter != mapping.end())
	std::cout << iter->second;
      else
	std::cout << *ptr;
      
      ptr++;
    }
  }
  
  fclose(fp);
  
}
