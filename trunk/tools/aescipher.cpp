/*
 * basic aes128 cipher program with given seed (key)
 *
 * this is somewhat insecure because (some reasons):
 *   - memory etc. is used carelessy: 
 *     swapping, cleaning used memories before freeing data etc. should be
 *     taken into account (memory locking, zeroing)
 *   - not extensively tested
 * 
 * aescipher returns non-zero if there was errors.
 * if file were crypted/decrypted correctly it returns zero.
 * 
 * missing features:
 *  - delete files '-D' flag (deletes&shreds original/encrypted file)
 *  - '-p <passphrase>' based encryption (SHA-1 given passphrase)
 *  - write test scripts to best binary (all cases) so that no
 *    data is lost (if key is known)
 *  - add compression to CryptoFileSource (gzip/libz is easy to use)
 *  - keyfiles
 *  - create symmetric keyfile
 *
 * improvements:
 *  - change fileformat to be:
 *    ID, IV, CONTENT, PAD. add always random PAD 
 *    even when CONTENT SIZE % PAD = 0
 *    save number of bytes in final block to the last byte
 *    (can be zero). encrypt content.
 *    this way it is possible to encrypt/decrypt data streams
 *    where you handle input data as an endless stream.
 *
 * encrypted tars: 
 *  tar cf test.tar *; gzip test.tar; aescipher 3279323892 test.tar.gz
 * 
 */


#include <iostream>
#include <dinrhiw/dinrhiw.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include "FileSource.h"
#include "CryptoFileSource.h"


using namespace whiteice;

bool read_arguments(unsigned int argc, char** argv, std::string& filename,
		    dynamic_bitset& key, bool& opflag,
		    unsigned int& numrounds,
		    bool& verbose, bool& asciiarmour, 
		    whiteice::crypto::ModeOfOperation& mode);

void create_encrypted_filename(std::string& filename);
void create_decrypted_filename(std::string& filename);



int main(int argc, char** argv)
{
  try{
    srand(time(0));
    
    std::string filename;
    dynamic_bitset key, IV;
    crypto::SymmetricCryptosystem<dynamic_bitset, dynamic_bitset>* cipher;
    crypto::Keyschedule<dynamic_bitset>* keyschedule;
    crypto::ModeOfOperation mode = crypto::CTRmode;
    
    CryptoFileSource<128>* filesource;
    bool encryptData = true;
    unsigned int numrounds = 11;
    
    bool verbose = false;
    bool aarmour = false;
    
    
    IV.resize(128);
    IV.reset();
    
    if(read_arguments(argc, argv, filename,
		      key, encryptData,
		      numrounds, aarmour,
		      verbose, mode) == false)
    {
      std::cout << "AES-128 cipher <dinrhiw2.sourceforge.net>" << std::endl;
      std::cout << "usage: aescipher [-rN] [-d] [-v] [-p] [-ascii] [-mode] hexkey128bits filename" << std::endl;
      std::cout << "       mode = ecb, cfb, cbc, ofb, ctr. (default: ctr)" << std::endl;
      return -2;
    }
    
    int rv = 0; // return value
    
    
    if(verbose){
      std::cout << "AES-128 ";
      if(mode == crypto::ECBmode) std::cout << "ECB";
      else if(mode == crypto::CFBmode) std::cout << "CFB";
      else if(mode == crypto::CBCmode) std::cout << "CBC";
      else if(mode == crypto::OFBmode) std::cout << "OFB";
      else if(mode == crypto::CTRmode) std::cout << "CTR";
      std::cout << " cipher <dinrhiw2.sourceforge.net>" << std::endl;
    }
    
    if(verbose)
      std::cout << key.size() << " bit key: 0x"
		<< key.to_hexstring() << std::endl;
    
    cipher = new whiteice::crypto::AES();
    keyschedule = new whiteice::crypto::AESKey(key);
    filesource = new CryptoFileSource<128>(filename, encryptData, false);


    if(filesource->good() == false){
      std::cerr << "ERROR: bad file '" << filename << "'" << std::endl;
      delete cipher;
      delete keyschedule;
      delete filesource;
      
      return -1;
    }
    
    if(((whiteice::crypto::AESKey*)keyschedule)->resize(numrounds) == false){
      std::cerr << "ERROR: keyschedule cannot be modified to provide requested number of rounds" << std::endl;
      delete cipher;
      delete keyschedule;
      delete filesource;
      
      return -1;
    }
    
    if(verbose)
      std::cout << "NR: " << ((whiteice::crypto::AESKey*)keyschedule)->size() << " "
		<< "FS: " << filesource->size() << std::endl;
    
    if(encryptData == true){
      if(verbose){
	std::cout << "encrypting... ";
	std::cout.flush();
      }
    
      if(cipher->encrypt(*filesource, *keyschedule, filesource->getIV(), mode) == false){
	std::cerr << "encrypt() failed." << std::endl;
	rv = - 1;
      }
      else{
	create_encrypted_filename(filename);
	
	if(filesource->write(filename)){
	  if(verbose)
	    std::cout << "ok." << std::endl;
	}
	else{
	  std::cerr << "file writing (i/o) failure." << std::endl;
	  rv = -1;
	}
      }            
    }
    else{
      if(verbose){
	std::cout << "decrypting... ";
	std::cout.flush();
      }
      
      if(cipher->decrypt(*filesource, *keyschedule, filesource->getIV(), mode) == false){
	std::cerr << "decrypt() failed." << std::endl;
	rv = -1;
      }
      else{
	create_decrypted_filename(filename);
	
	if(filesource->write(filename)){
	  if(verbose)
	    std::cout << "ok." << std::endl;
	}
	else{
	  std::cerr << "file writing (i/o) failure." << std::endl;
	  rv = -1;
	}	
      }
    }
    
    
    delete cipher;
    delete keyschedule;
    delete filesource;
    
    return rv;
  }
  catch(std::exception& e){
    std::cerr << "ERROR: Unexpected exception: " << e.what() << std::endl;
    return -1;
  }
}


void create_encrypted_filename(std::string& filename)
{
  const std::string tail = ".encrypted";
  
  filename = filename + tail;
}


void create_decrypted_filename(std::string& filename)
{
  const std::string tail  = ".encrypted";
  const std::string tail2 = ".decrypted";
  
  if(filename.size() > tail.size()){
    if(filename.compare(filename.size() - tail.size(), tail.size(), tail) == 0){
      filename.resize(filename.size() - tail.size());
    }
    else{
      filename = filename + tail2;
    }
  }
  else{
    filename = filename + tail2;
  }
}


bool read_arguments(unsigned int argc, char** argv,
		    std::string& filename, dynamic_bitset& key,
		    bool& opflag, unsigned int& numrounds,
		    bool& verbose, bool& asciiarmour,
		    whiteice::crypto::ModeOfOperation& mode)
{
  if(argc < 3) return false;
  
  bool passphrase = false;
  
  for(unsigned int i=1;i<(argc-2);i++){
    if(strncmp("-r", argv[i], 2) == 0){
      unsigned long long nr;
      char* endptr = &(argv[i][2]);
      
      nr = strtoul(&(argv[i][2]), &endptr, 10);
      
      if(strlen(endptr) > 0)
	return false; // cannot inteprete data
      
      if(nr < 3)
	return false; // at least 3 round keys needed
      
      numrounds = (unsigned int)(nr);
    }
    else if(strcmp("-d", argv[i]) == 0){
      opflag = false;  // decrypt data
    }
    else if(strcmp("-v", argv[i]) == 0){
      verbose = true;
    }
    else if(strcmp("-p", argv[i]) == 0){
      passphrase = true;
    }
    else if(strcmp("-ascii", argv[i]) == 0){
      asciiarmour = true;
    }
    else if(strcmp("-ecb", argv[i]) == 0){
      mode = crypto::ECBmode;
    }
    else if(strcmp("-cfb", argv[i]) == 0){
      mode = crypto::CFBmode;
    }
    else if(strcmp("-cbc", argv[i]) == 0){
      mode = crypto::CBCmode;
    }
    else if(strcmp("-ofb", argv[i]) == 0){
      mode = crypto::OFBmode;
    }
    else if(strcmp("-ctr", argv[i]) == 0){
      mode = crypto::CTRmode;
    }
    else{ // unrecognized option
      return false;
    }
  }
  
  if(passphrase == false){
    
    if(argv[argc-2][0] == '0' && argv[argc-2][1] == 'x')
      key = whiteice::math::integer( std::string(&(argv[argc-2][2])), 16 );
    else
      key = whiteice::math::integer( std::string(argv[argc-2]), 16 );
    
    key.resize(128);
  }
  else{
    // intepretes argv[argc-2] as passphrase
    
    whiteice::crypto::SHA sha(256); // uses only lowest 128bits
    unsigned char hash[(256/8)];
    
    key.resize(128);
    unsigned char* ptr = 
      (unsigned char*)malloc(strlen(argv[argc-2])+1);
    
    memcpy(ptr, argv[argc-2], strlen(argv[argc-2])+1);
    
    if(!sha.hash(&ptr, strlen((char*)ptr), hash))
      return false;
    
    for(unsigned int i=0;i<(256/8)/2;i++)
      key.value(i) = hash[i];
    
    if(verbose)
      std::cout << "Passphrase is: " << argv[argc-2] << std::endl;
  }
  
  filename = argv[(argc - 1)];
  
  // checks existence/accessability of filename
  {
    struct stat buf;
    
    if(stat(filename.c_str(), &buf) != 0){
      std::cerr << "Cannot access file: " << filename << std::endl;
      return false;
    }
  }
  
  // checks if key is zero
  // TRY IF "key == int" comparision works after ALGOS recompile
  if(key.to_integer() == whiteice::math::integer(0L)){
    std::cout << "Warning: encryption key is zero. " << std::endl
	      << "Key might not be correctly formed hexadecimal number."
	      << std::endl;
  }
  
  return true;
}

