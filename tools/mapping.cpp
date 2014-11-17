/*
 * memory mapping test for HD-NN implementation
 *
 */


#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <iostream>
#include <exception>

#include "MMAP.h"

using namespace whiteice;

int main(int argc, char** argv)
{
  try{
    int fd = open("largefile", O_RDWR | O_CREAT | O_LARGEFILE,
		  S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP |
		  S_IROTH | S_IWOTH);
  
    if(fd == -1){
      printf("Opening file failed.\n");
      return -1;
    }
    
    // creating 1.5 GB memory mapped file
    
    MMAP mmap(fd, 0x10000000000ULL);
    
    printf("size: 0x%x%x\n", (unsigned int)(mmap.size() >> 32),
	   (unsigned int)(mmap.size() & 0xFFFFFFFFULL));
    
    printf("pages: 0x%x%x\n", (unsigned int)(mmap.pagesize() >> 32),
	   (unsigned int)(mmap.pagesize() & 0xFFFFFFFFULL));
    
    for(unsigned long long int i=2;
	i<0x10000000000ULL;
	i += 0x10000000000ULL/1000ULL)
      mmap[i] = mmap[i-1] + 13;
    
    return 0;
  }
  catch(std::exception& e){
    std::cout << "E: " << e.what() << std::endl;
    return -1;
  }
}



