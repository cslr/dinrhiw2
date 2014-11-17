

#ifndef dynamic_bitset_h
#define dynamic_bitset_h

#include <iostream>
#include <string>
#include <stdexcept>
#include <exception>
#include <vector>
#include <iostream>
#include "integer.h"


namespace whiteice
{
  class dynamic_bitset
  {
  public:
    
    dynamic_bitset();
    dynamic_bitset(const dynamic_bitset& bitset);
    explicit dynamic_bitset(unsigned int val);
    explicit dynamic_bitset(unsigned long val);
    dynamic_bitset(const whiteice::math::integer& i);
    
    virtual ~dynamic_bitset();
    
    dynamic_bitset& operator=(const dynamic_bitset& bitset);
    
    dynamic_bitset& operator&=(const dynamic_bitset& r);
    dynamic_bitset& operator|=(const dynamic_bitset& r);
    dynamic_bitset& operator^=(const dynamic_bitset& r);
    dynamic_bitset& operator<<=(int pos) throw();
    dynamic_bitset& operator>>=(int pos) throw();
    
    // adds bitset at the beginning of the current one
    dynamic_bitset& operator+=(const dynamic_bitset& r);
    
    dynamic_bitset operator&(const dynamic_bitset& r) const;
    dynamic_bitset operator|(const dynamic_bitset& r) const;
    dynamic_bitset operator^(const dynamic_bitset& r) const;
    dynamic_bitset operator<<(const int pos) const throw();
    dynamic_bitset operator>>(const int pos) const throw();
    
    // ORs bit sequencies
    dynamic_bitset operator+(const dynamic_bitset& r) const;
    
    dynamic_bitset& set() throw(); // sets all bits
    dynamic_bitset& set(unsigned int pos, bool val = true) throw(std::out_of_range);
    
    dynamic_bitset& reset() throw(); // clears all bits
    dynamic_bitset& reset(unsigned int pos) throw(std::out_of_range);
    
    dynamic_bitset& flip() throw();
    dynamic_bitset& flip(unsigned int pos) throw(std::out_of_range);
    
    
    inline bool operator[](unsigned int pos)
      const throw(std::out_of_range)
    {
#ifndef FAST_CODE
      if(pos >= seqlen)
	throw std::out_of_range("index bigger than bitset length");
#endif
      
      unsigned int rpos = pos % (sizeof(unsigned long)*8);
      pos /= (sizeof(unsigned long)*8);
      
      return ((bool)((this->memory[pos] >> rpos) & 1));
    }
    
    std::string to_string() const;
    std::string to_hexstring() const;
    whiteice::math::integer to_integer() const;
    
    unsigned int count() const throw(); // number of bits set
    unsigned int size() const throw();  // length of the bit sequence
        
    unsigned int blocks() const throw(); // number of 8bit blocks
    
    void resize(unsigned int len) throw(std::bad_alloc);  // sets length of sequence
    
    bool operator==(const dynamic_bitset& r) const throw();
    bool operator!=(const dynamic_bitset& r) const throw();
    
    bool any() const throw();  // is any bit set?
    bool none() const throw(); // no bits are set?
    
    dynamic_bitset operator~() const; // returns flipped bitset
    bool operator!() const; // returns true if no bits are set (iszero)
    
    // shifts bitset cyclically (pos > 0: left shift, pos < 0: right shift)
    void cyclicshift(int pos) throw(std::bad_alloc);
    
    // accesses 8 bit blocks
    // WARNING: when accessing the last block's unused bits
    // must be set to zero or dynamic_bitset may stop working correctly
    inline unsigned char& value(unsigned int p) throw(std::out_of_range){
#ifndef FAST_CODE
      // remove checks to increase speed and reduce stability
      if(p >= (sizeof(unsigned long)*numblocks)/sizeof(unsigned char))
	throw std::out_of_range("index bigger than bitset length");
#endif
    
      return ((unsigned char*)(this->memory))[p];
    }
    
    inline const unsigned char& value(unsigned int p) const throw(std::out_of_range){
#ifndef FAST_CODE
      // remove checks to increase speed and reduce stability
      if(p >= (sizeof(unsigned long)*numblocks)/sizeof(unsigned char))
	throw std::out_of_range("index bigger than bitset length");
#endif
      
      return ((unsigned char*)memory)[p];
    }
    
    
    // increments unsigned integer by one 
    // (uses carry flag to optimize this one)
    inline void inc() throw()
    {      
#if defined IA32
      // assumes sizeof(unsigned long) = 4
      
      asm("1:             \n\t" // loop start
	  "incl (%%eax)   \n\t"
	  "jnz 2f         \n\t"
	  "addl $4, %%eax \n\t"
	  "decl %%ecx     \n\t"
	  "jnz 1b         \n\t"
	  "2:             \n\t" // loop end
	  : /* no output registers */
	  : "c" (numblocks), "a" (memory));
#elif defined AMD64
      // assumes sizeof(unsigned long) = 8
      
      asm("1:             \n\t" // loop start
	  "incq (%%rax)   \n\t"
	  "jnz 2f         \n\t"
	  "addq $8, %%rax \n\t"
	  "decl %%ecx     \n\t"
	  "jnz 1b         \n\t"
	  "2:             \n\t" // loop end
	  : /* no output registers */
	  : "c" (numblocks), "a" (memory));
#else
#error "Unsupported machine architecture"
#endif
      
      clean_dirty_bits();
    }
    
    
    // decrements unsigned integer by one
    inline void dec() throw()
    {

#if defined IA32
      // assumes sizeof(unsigned long) = 4
      
      asm("1:               \n\t" // loop start
	  "subl $1, (%%eax) \n\t" // no way to get this done with dec?
	  "jnc 2f           \n\t" // (dec doesn't update carry flag)
	  "addl $4, %%eax   \n\t"
	  "decl %%ecx       \n\t"
	  "jnz 1b           \n\t"
	  "2:               \n\t" // loop end
	  : /* no output registers */
	  : "c" (numblocks), "a" (memory));
#elif defined AMD64
      // assumes sizeof(unsigned long) = 8
      
      asm("1:               \n\t" // loop start
	  "subq $1, (%%rax) \n\t" // no way to get this done with dec?
	  "jnc 2f           \n\t" // (dec doesn't update carry flag)
	  "addq $8, %%rax   \n\t"
	  "decl %%ecx       \n\t"
	  "jnz 1b           \n\t"
	  "2:               \n\t" // loop end
	  : /* no output registers */
	  : "c" (numblocks), "a" (memory));      
#else
#error "Unsupported machine architecture"
#endif
      
      clean_dirty_bits();
    }
    


    
  private:
    
    inline void clean_dirty_bits(){ // cleans unused bits
      // all zero mask isn't possible/it codes: no cleaning needed case
      if(cleanmask) 
	memory[numblocks - 1] &= cleanmask;
    }
    
    // needs to be done every time seqlen changes
    void calculate_cleanmask();
    
    
    unsigned long* memory; // nbits worth memory (+ round up to sizeof(char)*8 bits)
    
    unsigned int seqlen; // lengts of bit sequence (in bits)
    unsigned int numblocks; // lenght of sequence (in unsigned longs)
    
    unsigned long cleanmask; // used to clean possible dirty bits
    
  };
  
  
  
  
  std::ostream& operator<<(std::ostream& os, const dynamic_bitset& b);
  
  
};


#endif
