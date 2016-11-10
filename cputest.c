
#include <stdio.h>
#include <string.h>

void CPUID(unsigned int input, unsigned int *output)
{
  __asm__ __volatile__("cpuid"
		       : "=a" (output[0]), "=b" (output[1]), "=c" (output[2]),
		         "=d" (output[3])
		       : "a" (input));
}


int main()
{
  // CPU instruction set detection routines:
  // detects: SSE2, SSE3
  
  unsigned int i;
  unsigned int result[4];
  char buf[20];
  
  CPUID(0, (unsigned int*)buf);
  for(i=0;i<4;i++){
    char tmp  = buf[8+i];
    buf[8+i]  = buf[12+i];
    buf[12+i] = tmp;
  }
  
  buf[16] = '\0';
  
  char is_amd = 0, is_intel = 0;
  if(!strcmp(buf+4, "AuthenticAMD"))
    is_amd   = 1;
  else if(!strcmp(buf+4, "GenuineIntel"))
    is_intel = 1;
  
  CPUID(1, result);
  if(result[3] & 1) // FPU bit
    printf("fpu ");
  if(result[3] & (1 << 15))
    printf("cmov ");  // has cmovs
  if(result[3] & (1 << 4))
    printf("rdtsc "); // has rdtsc
  if(result[3] & (1 << 23))
    printf("mmx "); // mmx
  if(result[3] & (1 << 25))
    printf("sse "); // sse
  
  if(is_amd){
    // checks for SSE2, SSE3 support
    CPUID(1, result);
    
    if(result[3] & (1 << 26))
      printf("sse2 "); // sse2
    
    if(result[2] & 1) // ECX Bit 0
      printf("ss3 ");
    
    if(result[3] & (1 << 28))
      printf("htt "); // hyperthreading    
    
    CPUID(0x80000000, result);
    
    if(result[0] >= 0x80000001){
      // has 0x80000001
      CPUID(0x80000001, result);
      
      if(result[3] & (1 << 31))
	printf("3dnow ");
      
      if(result[3] & (1 << 30))
	printf("3dnowext ");
    }
    
  }
  else if(is_intel){
    // checks for SSE2, SSE3 support
    CPUID(1, result);
    
    if(result[3] & (1 << 26))
      printf("sse2 "); // sse2

    if(result[2] & 1)
      printf("sse3 "); // sse3
    
    if(result[3] & (1 << 28))
      printf("htt "); // hyperthreading
  }
  
  printf("\n");
  
  return 0;
}
