
#ifdef OPENBLAS

#include <cblas.h>
#include "openblas_config.h"

int main(void){ 
  openblas_complex_double* ocd = NULL;
  return 0; 
}
#endif

#ifdef INTELMKL

#include "mkl_cblas.h"

int main(void){
  cblas_scopy(10, NULL, 1, NULL, 1);
  return 0;
}

#endif

#ifdef AMDBLIS

#include "cblas.h"

int main(void){
  cblas_scopy(10, NULL, 1, NULL, 1);
  return 0;
}

#endif

#ifdef NVBLAS

// hopefully NVIDIA will later provide full C BLAS interface functions
// that Intel, OpenBLAS and AMD(?) provide. Until then, compilation
// of this function fails.

#include "nvblas.h"

int main(void){
  cblas_scopy(10, NULL, 1, NULL, 1);
}

#endif

