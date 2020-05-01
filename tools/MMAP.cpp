
#include "MMAP.h"

#include <exception>
#include <stdexcept>

#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>


namespace whiteice
{
  
  MMAP::MMAP(int fd, unsigned long long bytes)
  {
    if(bytes > (__LIMIT__ - PAGESIZE))
      throw std::out_of_range("memory area too large");
    
    {
      // physical page size
      const unsigned long long PPS = sysconf(_SC_PAGESIZE);
      bytes = ((bytes + PPS - 1) / PPS) * PPS;
    }
    
        
    if(bytes > (__LIMIT__ - PAGESIZE))
      throw std::out_of_range("memory area too large");
    
    this->activePage = 0;
    this->page_ptr = 0;
    this->fd = fd;
    this->bytes = bytes;
    
    if(ftruncate64(fd, bytes) == -1){
      switch(errno){
      case EBADF:
	throw std::out_of_range("resizing file failed [corrupted file descriptor].");
      case EACCES:
	throw std::out_of_range("resizing file failed [file is directory or not writable].");
      case EINVAL:
	throw std::out_of_range("resizing file failed [ftruncate length parameter is negative].");
      case EFBIG:
	throw std::out_of_range("resizing file failed [file size beyond OS or file system limits].");
      case EIO:
	throw std::out_of_range("resizing file failed [hardware I/O error].");
      case EPERM:
	throw std::out_of_range("resizing file failed [file is append only or immutable].");
      case EINTR:
	throw std::out_of_range("resizing file failed [operation interrupted by signal].");
      default:
	throw std::out_of_range("resizing file failed.");
      };
      
    }
    
    if(bytes > 0){
      // tries to create initial mapping
      
      if(bytes < PAGESIZE)
	page_ptr = (unsigned char*)
	  mmap(0, bytes, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE, fd, 0);
      else
	page_ptr = (unsigned char*)
	  mmap(0, PAGESIZE, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE, fd, 0);
      
      if(page_ptr == MAP_FAILED){
	if(bytes < PAGESIZE)
	  munmap(page_ptr, bytes);
	else
	  munmap(page_ptr, PAGESIZE);
	
	page_ptr = 0;
	throw std::length_error("Creation of memory mapping failed.");
      }
    }
  }
  
  
  MMAP::MMAP(const MMAP& mmap)
  {
    throw std::invalid_argument("Cannot create copy of MMAP.");
  }
  
  
  MMAP::~MMAP()
  {
    // frees mapping
    if(page_ptr){
      
      const unsigned long long numpages = (bytes + PAGESIZE - 1) / PAGESIZE;
      
      if(activePage == numpages - 1){ // the last page
	if(bytes % PAGESIZE)
	  munmap(page_ptr, bytes % PAGESIZE);
	else
	  munmap(page_ptr, PAGESIZE);
      }
      else
	munmap(page_ptr, PAGESIZE);
      
      page_ptr = 0;
      activePage = 0;
    }
    
    bytes = 0;
  }
  
  bool MMAP::resize(unsigned long long newbytes) 
  {
    if(newbytes > (__LIMIT__ - PAGESIZE))
      return false; // too large memory area
    
    {
      // physical page size
      const unsigned long long PPS = sysconf(_SC_PAGESIZE);
      newbytes = ((newbytes + PPS - 1) / PPS)*PPS;
    }
    
    if(newbytes > (__LIMIT__ - PAGESIZE))
      return false; // too large memory area
    
    const unsigned long long oldpages = (bytes + PAGESIZE - 1) / PAGESIZE;
    const unsigned long long newpages = (newbytes + PAGESIZE - 1) / PAGESIZE;
    
    if(ftruncate64(fd, newbytes) == -1)
      return false;
    
    if(activePage >= newpages){
      // page doesn't exist anymore
      if(activePage == oldpages - 1){
	if(bytes % PAGESIZE)
	  munmap(page_ptr, bytes);
	else
	  munmap(page_ptr, PAGESIZE);
      }
      
      activePage = 0;
      if(newbytes > PAGESIZE)
	page_ptr = (unsigned char*)
	  mmap(0, PAGESIZE, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE, fd, 0);
      else
	page_ptr = (unsigned char*)
	  mmap(0, newbytes, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE, fd, 0);
      
      if(page_ptr == MAP_FAILED)
	return false;
    }
    else if(activePage == newpages - 1){
      // the new last page

      // page needs to be remapped
      if(activePage == oldpages - 1){
	if(bytes % PAGESIZE)
	  munmap(page_ptr, bytes);
	else
	  munmap(page_ptr, PAGESIZE);
      }
      
      if(newbytes % PAGESIZE)
	page_ptr = (unsigned char*)
	  mmap(0, newbytes % PAGESIZE, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE, fd, newbytes - (newbytes % PAGESIZE));
      else
	page_ptr = (unsigned char*)
	  mmap(0, PAGESIZE, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE, fd, newbytes - PAGESIZE);
      
      if(page_ptr == MAP_FAILED)
	return false;
    }
    else if(activePage == oldpages - 1 && oldpages < newpages){
      // was the last page and is full size page
      
      if(bytes % PAGESIZE)
	munmap(page_ptr, bytes);
      else
	munmap(page_ptr, PAGESIZE);
      
      page_ptr = (unsigned char*)
	mmap(0, PAGESIZE, PROT_READ | PROT_WRITE,
	     MAP_PRIVATE, fd, activePage*PAGESIZE);
      
      if(page_ptr == MAP_FAILED)
	return false;
    }
    
    return true;
  }
  
  
  
  bool MMAP::changePage(unsigned long long newpage) const 
  {
    const unsigned long long numpages = (bytes + PAGESIZE - 1) / PAGESIZE;
    
    if(activePage == numpages - 1){
      if(bytes % PAGESIZE)
	munmap(page_ptr, bytes % PAGESIZE);
      else
	munmap(page_ptr, PAGESIZE);
    }
    else
      munmap(page_ptr, PAGESIZE);
    
    if(newpage == numpages - 1){
      if(bytes % PAGESIZE)
	page_ptr = (unsigned char*)
	  mmap(0, bytes % PAGESIZE, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE, fd, newpage*PAGESIZE);
      else
	page_ptr = (unsigned char*)
	  mmap(0, PAGESIZE, PROT_READ | PROT_WRITE,
	       MAP_PRIVATE, fd, newpage*PAGESIZE);
    }
    else{
      page_ptr = (unsigned char*)
	mmap(0, PAGESIZE, PROT_READ | PROT_WRITE,
	     MAP_PRIVATE, fd, newpage*PAGESIZE);
    }
    
    activePage = newpage;
    
    if(page_ptr == MAP_FAILED)
      return false;
    
    return true;
  }
  
  
  ////////////////////////////////////////////////////////////////////////////////
  
  FMMAP::FMMAP(int fd, unsigned long long floats)
  {
    pages = 0;
    this->floats = floats;
    this->fd = fd;    
    
    if(floats > (__LIMIT__ - PAGESIZE)/sizeof(float))
      throw std::out_of_range("memory area too large");
    
    const unsigned long long bytes = floats*sizeof(float);
    
    if(bytes > 0){
      // tries to create mapping
      const unsigned long long numpages = (bytes + PAGESIZE - 1) / PAGESIZE;
      
      pages = (float**)malloc(sizeof(void*)*numpages);
      
      if(pages == 0) throw std::bad_alloc();
      
      if(numpages > 1)
	for(unsigned long long i=0;i+1<numpages;i++){
	  pages[i] = (float*)
	    mmap(0, PAGESIZE, PROT_READ | PROT_WRITE,
		 MAP_PRIVATE, fd, i*PAGESIZE);
	  
	  if(pages[i] == MAP_FAILED){
	    for(unsigned long long j=0;j<i;j++)
	      munmap(pages[i], PAGESIZE);
	    
	    free(pages);
	    pages = 0;
	    
	    throw std::length_error("Cannot create memory mapping.");
	  }
	}
      
      // maps the last page
      pages[numpages-1] = (float*)
	mmap(0, bytes % PAGESIZE, PROT_READ | PROT_WRITE,
	     MAP_PRIVATE, fd, bytes - (bytes % PAGESIZE));
      
      if(pages[numpages-1] == MAP_FAILED){
	for(unsigned long long j=0;j<numpages-1;j++)
	  munmap(pages[j], PAGESIZE);
	
	free(pages);
	pages = 0;
	
	throw std::length_error("Cannot create memory mapping.");
      }
    }
  }
  
  
  FMMAP::FMMAP(const FMMAP& fmmap)
  {
    throw std::invalid_argument("Cannot create copy of FMMAP.");
  }
  
  
  FMMAP::~FMMAP()
  {
    // frees mapping
    if(pages){
      const unsigned long long numpages = 
	(floats*sizeof(float) + PAGESIZE - 1) / PAGESIZE;
      
      for(unsigned long long i=0;i+1<numpages;i++)
	munmap(pages[i], PAGESIZE);
      
      munmap(pages[numpages - 1], (floats*sizeof(float)) % PAGESIZE);
    }
    
    pages = 0;
    floats = 0;    
  }
  
  
  bool FMMAP::resize(unsigned long long floats) 
  {
    // implement me!
    return false;
  }
  
  ////////////////////////////////////////////////////////////////////////////////
};
