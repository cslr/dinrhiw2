
#include "cpuid_threads.h"


void cpuID(unsigned i, unsigned regs[4]) {
#ifdef _WIN32
  __cpuid((int *)regs, (int)i);
  
#else
  asm volatile
    ("cpuid" : "=a" (regs[0]), "=b" (regs[1]), "=c" (regs[2]), "=d" (regs[3])
     : "a" (i), "c" (0));
  // ECX is set to zero for CPUID function 4
#endif
}




int numberOfCPUThreads()
{
  unsigned int regs[4];

#if 0
  // Get vendor
  char vendor[12];
  cpuID(0, regs);
  ((unsigned *)vendor)[0] = regs[1]; // EBX
  ((unsigned *)vendor)[1] = regs[3]; // EDX
  ((unsigned *)vendor)[2] = regs[2]; // ECX
  std::string cpuVendor = std::string(vendor, 12);
#endif
  
  // Get CPU features
  // cpuID(1, regs);
  // unsigned int cpuFeatures = regs[3]; // EDX
  
  // Logical core count per CPU(?)
  cpuID(1, regs);
  unsigned int logical = (regs[1] >> 16) & 0xff; // EBX[23:16]
  // std::cout << " logical cpus: " << logical << std::endl;
  // unsigned int cores = logical;

#if 0
  if (cpuVendor == "GenuineIntel") {
    // Get DCP cache info
    cpuID(4, regs);
    // cores = ((regs[0] >> 26) & 0x3f) + 1; // EAX[31:26] + 1
    
  } else if (cpuVendor == "AuthenticAMD") {
    // Get NC: Number of CPU cores - 1
    cpuID(0x80000008, regs);
    // cores = ((unsigned)(regs[2] & 0xff)) + 1; // ECX[7:0] + 1
  }
#endif  
  // std::cout << "    cpu cores: " << cores << std::endl;
  
  // Detect hyper-threads
  // bool hyperThreads = cpuFeatures & (1 << 28) && cores < logical;
  
  return (int)logical;
}


