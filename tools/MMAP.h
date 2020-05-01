/*
 * large area memory mapping for 32bit platforms
 * uses gcc "unsigned long long" extension for
 * 64bit offsets and sizes
 * 
 * single access is atomic
 * - memory accesses during resizes have undefined behaviour
 * - changing larger memory areas atomically is responsibility of user
 */

#ifndef MMAP_h
#define MMAP_h

#include <stdexcept>
#include <exception>
#include <assert.h>

// 1 GB page size
#define PAGESIZE 0x20000000ULL

// 64bit address space limit (biggest possible index)
#define __LIMIT__  0xFFFFFFFFFFFFFFFFULL

// for calculating correct page and offset
#define PAGESHIFT 29
#define PAGEMASK  0x1FFFFFFFULL

// ASSUMES sizeof(float) is 4

#define FP_PAGESHIFT 27
#define FP_PAGEMASK  0x7FFFFFFULL


namespace whiteice
{
  
  class MMAP // memory mapping
  {
  public:
    MMAP(int fd, unsigned long long bytes);
    MMAP(const MMAP& mmap);
    ~MMAP();
    
    bool resize(unsigned long long bytes) ;
    
    unsigned long long size() const {
      return bytes;
    }

    unsigned long long pagesize() const {
      return ((bytes + PAGESIZE - 1) / PAGESIZE);
    }

    inline const unsigned char& operator[](unsigned long long index) const {
      const unsigned long long ipage = (index >> PAGESHIFT);
      if(ipage != activePage)
	assert(changePage(ipage) == true);
      
      return page_ptr[index & PAGEMASK];
    }
    
    inline unsigned char& operator[](unsigned long long index) {
      const unsigned long long ipage = (index >> PAGESHIFT);
      if(ipage != activePage)
	assert(changePage(ipage) == true);
      
      return page_ptr[index & PAGEMASK];
    }
    
    // returns pointer to PAGESIZE continuous piece of memory
    // (or less if last page)
    inline const unsigned char* page(unsigned long long page) const {
      if(activePage != page)
	assert(changePage(page) == true);
      
      return page_ptr;
    }
    
    // returns pointer to PAGESIZE continuous piece of memory
    // (or less if last page)
    inline unsigned char* page(unsigned long long page) {
      if(activePage != page)
	assert(changePage(page) == true);
      
      return page_ptr;
    }
    
  private:
    bool changePage(unsigned long long newpage) const ;
    
    mutable unsigned long long activePage;
    mutable unsigned char* page_ptr;
    
    unsigned long long bytes;
    int fd;
  };
  
  
  class FMMAP // floating point memory map
  {
  public:
    
    FMMAP(int fd, unsigned long long floats);
    FMMAP(const FMMAP& fmmap);
    ~FMMAP();
    
    bool resize(unsigned long long floats) ;

    unsigned long long size() const {
      return floats;
    }

    unsigned long long pagesize() const {
      return ((floats*sizeof(float) + PAGESIZE - 1) / PAGESIZE);
    }
    
    inline const float& operator[](unsigned long long index) const {
      return pages[index >> FP_PAGESHIFT][index & FP_PAGEMASK];
    }
    
    inline float& operator[](unsigned long long index) {
      return pages[index >> FP_PAGESHIFT][index & FP_PAGEMASK];
    }
    
    // returns pointer to 1 GB continuous piece of memory
    // this means 1 GB / sizeof(float) number of floating point
    // numbers (or less if last page)
    inline const float* page(unsigned int page) const {
      return pages[page];
    }
    
    // returns pointer to 1 GB continuous piece of memory
    // this means 1 GB / sizeof(float) number of floating point
    // numbers (or less if last page)
    inline float* page(unsigned int page) {
      return pages[page];
    }
    
  private:
    float** pages;
    
    unsigned long long floats;
    int fd;
  };
  
};


#undef PAGESHIFT
#undef PAGEMASK
#undef FP_PAGESHIFT
#undef FP_PAGEMASK


#endif
