
#include "cpuid_threads.h"

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#endif


void cpuID(unsigned i, unsigned regs[4]) {
//#ifdef _WIN32
//  __cpuid((int *)regs, (int)i);
//
//#else
  asm volatile
    ("cpuid" : "=a" (regs[0]), "=b" (regs[1]), "=c" (regs[2]), "=d" (regs[3])
     : "a" (i), "c" (0));
  // ECX is set to zero for CPUID function 4
//#endif
}


int cpuinfoThreads(){
  // calculates number of cpu cores from /proc/cpuinfo
  
  FILE* f = fopen("/proc/cpuinfo", "rt");
  if(f == NULL) return 0;
  else if(ferror(f) != 0){ fclose(f); return 0; }
  
  unsigned int cpus = 0;
  unsigned int cores = 0;
  
  char buffer[256];
  while(!feof(f)){
    fgets(buffer, 256, f);
    buffer[255] = '\0';

    if(strncmp(buffer, "processor", 9) == 0)
      cpus++;
    
    if(strncmp(buffer, "cpu cores", 9) == 0){
      char *ptr = buffer;
      while(*ptr != ':' && *ptr != '\0') ptr++;
      
      if(*ptr == ':'){
	ptr++;
	int c = atoi(ptr);
	if(c > 0)
	  cores += c;
      }
    }
  }
  
  fclose(f);

#ifdef WINOS
  // TODO write code to get number cpus/cores from Windows (OS)
#endif

  if(cores > 0) return cores;
  else return cpus;
}




int numberOfCPUThreads()
{
  // uses CPUID instruction

	  unsigned int regs[4];

	  // Get vendor
	  char vendor[12];
	  cpuID(0, regs);
	  ((unsigned *)vendor)[0] = regs[1]; // EBX
	  ((unsigned *)vendor)[1] = regs[3]; // EDX
	  ((unsigned *)vendor)[2] = regs[2]; // ECX
	  std::string cpuVendor = std::string(vendor, 12);

	  // Get CPU features
	  // cpuID(1, regs);
	  // unsigned int cpuFeatures = regs[3]; // EDX

	  // Logical core count per CPU(?)
	  cpuID(1, regs);
	  unsigned int logical = (regs[1] >> 16) & 0xff; // EBX[23:16]
	  // std::cout << " logical cpus: " << logical << std::endl;
	  unsigned int cores = logical;

	  if (cpuVendor == "GenuineIntel") {
    // Get DCP cache info
    cpuID(4, regs);
    cores = ((regs[0] >> 26) & 0x3f) + 1; // EAX[31:26] + 1
  }
  else if (cpuVendor == "AuthenticAMD") {
    // Get NC: Number of CPU cores - 1
    cpuID(0x80000008, regs);
    cores = ((unsigned)(regs[2] & 0xff)) + 1; // ECX[7:0] + 1
  }
  // std::cout << "    cpu cores: " << cores << std::endl;
  
  // Detect hyper-threads
  // bool hyperThreads = cpuFeatures & (1 << 28) && cores < logical;

  return (int)cores;
}


