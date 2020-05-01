/*
 * chained dynamically resizing hash table with universal hashing
 */
#ifndef chainhash_h
#define chainhash_h

#include "hash_table.h"
#include "array.h"

namespace whiteice
{

  template <typename D, typename T>
  class chainhash : public hash_table<D,T>
  {
  public:
    
    chainhash(const T initial_size = 10037, /* should be prime! */
	      const float alpha = 1.5);
    
    virtual ~chainhash();
    
    bool insert(const T& key, D& data) ;
    bool remove(const T& key) ;
    
    D&   search(const T& key) ;
    D& operator[](const T& key) ;
    
    bool rehash();
    
    /* information / parameter changing */
    T size() const { return data_size; }
    T get_table_size() const{ return table_size; }
    float get_alpha() const { return alpha; }
    float set_alpha(float a){ this->alpha = a; return this->ALPHA; }
    
  private:
    search_array<D, T> *table; // hash table
    unsigned char* bytetable;  // coefficients of universal hash function
    
    const T hash(const T& key) const;
    
    T  data_size;
    T  coef_size;
    T table_size;
    
    float alpha;
    
    
    class hash_node
    {
    public:
      D key;
      T data;
    };
    
  };
  
}
  
#include "chainhash.cpp"

#endif
