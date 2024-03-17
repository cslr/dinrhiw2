

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
// #include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include "FileSource.h"


template <unsigned int B>
FileSource<B>::FileSource(const std::string& filename)
{
  file = (FILE*)fopen(filename.c_str(), "rb");
  
  buffer = new unsigned char[(size()*B)/8];  // size also calculates filesize
  memset(buffer, 0, size()*B/8);
  
  if(fread(buffer, sizeof(char), filesize, file) != filesize)
    throw std::runtime_error("file i/o error");
  
  this->DC = new whiteice::crypto::DataConversion<B>(buffer, size()*B/8);  
}


template <unsigned int B>
FileSource<B>::~FileSource()
{
  if(DC != 0){ delete DC; DC = 0; }
  if(file != 0){ fclose(file); file = 0; }
  if(buffer != 0){ free(buffer); buffer = 0; }
}


template <unsigned int B>
whiteice::dynamic_bitset& FileSource<B>::operator[](unsigned int index)
   // not thread safe
{
  return (*DC)[index];
}


template <unsigned int B>
const whiteice::dynamic_bitset& FileSource<B>::operator[](unsigned int index)
  const  // not thread safe
{
  return (*DC)[index];
}


/* returns number of blocks */
template <unsigned int B>
unsigned int FileSource<B>::size() const 
{
  if(filesize == 0){
    struct stat buf;
    
    if(fstat(fileno(file), &buf) != 0)
      return 0;
    
    filesize = buf.st_size;
  }
  
  return ((filesize*8+(B-1))/B); // converts to block size
}


template <unsigned int B>
bool FileSource<B>::good() const 
{
  if(ferror(file) != 0) return false;
  
  // checks file is regular file (non-special)
  struct stat buf;
  if(fstat(fileno(file), &buf) != 0)
    return false;
  
  // shoud be regular non-directory file
  if( (S_IFREG & buf.st_mode) == 0 ||
      (S_IFDIR & buf.st_mode) != 0) return false;
  
  // block size must be multiple of 8
  if((B % 8) != 0) return false;
  
  return true;
}


template <unsigned int B>
void FileSource<B>::flush() const
{
  if(DC) DC->flush();
}


template <unsigned int B>
bool FileSource<B>::write(const std::string& filename) const 
{
  flush();
  
  FILE* out = (FILE*)fopen(filename.c_str(), "wb");
  if(out == 0) return false;
  
  if(fwrite(buffer, sizeof(char), size()*B/8, out) != size()*B/8)
    return false;
  
  fclose(out);
  
  return true;
}



/**************************************************/

template class FileSource<128>;

/**************************************************/
