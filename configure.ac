dnl Process this file with autoconf to produce a configure script.

AC_INIT([whiteice::dinrhiw],[<tomas.ukkonen@iki.fi>],[libdinrhiw])

dnl finds the basic compiling tools
AC_PROG_CC
AC_PROG_CXX
AC_PROG_CPP

dnl needed by autoconf itself
AC_PROG_FGREP

AC_CHECK_SIZEOF(char, 1)
AC_CHECK_SIZEOF(short, 2)
AC_CHECK_SIZEOF(int, 4)
AC_CHECK_SIZEOF(long, 4)
AC_CHECK_SIZEOF(long long int, 8)

dnl does source files seem to exist?
AC_CONFIG_SRCDIR([src/dinrhiw.h])

dnl ----------------------------------------------------------
dnl do all the magick stuff here


dnl checks if -I/usr/local/include is needed
AC_MSG_CHECKING([is -I/usr/local/include command line parameter needed])

dnl not checked right now, just added
AC_MSG_RESULT([yes (not checked, write real check)])
EXTRA_INCLUDEPATHS="-I/usr/local/include"

dnl checks if -L/usr/local/lib is needed to
dnl link with libraries in /usr/local/lib
AC_MSG_CHECKING([is -L/usr/local/lib command line parameter needed])

dnl not checked right now, just added
AC_MSG_RESULT([yes (not checked, write real check)])
EXTRA_LIBPATHS="-L/usr/local/lib"

dnl check for Windows (WINNT)
AC_MSG_CHECKING([Windows OS])
EXTRA_DEFINES=""
SHAREDLIBSUFFIX="so"
`g++ -o windowstest.exe windowstest.cpp > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
   EXTRA_DEFINES="-DWINOS"
   SHAREDLIBSUFFIX="dll"
dnl    AC_MSG_RESULT([yes, $conf_flags])
    AC_MSG_RESULT([yes])
else
    AC_MSG_RESULT([no])
fi


dnl checks if isfinite() is defined
AC_MSG_CHECKING([for isfinite()])
has_isfinite="no"

AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
[#include <math.h>]
[char test(void){ return isfinite(0.0f); }]])],
[has_isfinite="yes"])

if [test $has_isfinite = "yes"]; then
  AC_DEFINE(HAVE_ISFINITE,[1],[Defined if isfinite() is in <math.h>])
fi

AC_MSG_RESULT([$has_isfinite])

dnl checks if gettimeofday() is either in time.h or sys/time.h
AC_MSG_CHECKING([if gettimeofday() is available])


AC_LINK_IFELSE([AC_LANG_PROGRAM([
[#include <time.h>]
[#include <sys/time.h>]],
[struct timeval tv; gettimeofday(&tv,0);])],
[has_gettimeofday="yes"],
[has_gettimeofday="no"])

if [test $has_gettimeofday = "yes" ]; then
  AC_DEFINE(HAVE_GETTIMEOFDAY,[1],[Defined if gettimeofday() can be found.])
fi

AC_MSG_RESULT([$has_gettimeofday])


AC_MSG_CHECKING(['uname' command])
cpufamily=`uname -m`

if [[ $? -eq 0 ]]; then
    AC_MSG_RESULT([ok])
else
    AC_MSG_ERROR([DOESN'T EXIST!])
fi

AC_MSG_CHECKING([CPU family])
AC_MSG_RESULT($cpufamily)

dnl enables CPU specific optimization flags

AC_ARG_ENABLE([optimize],
	AS_HELP_STRING([--disable-optimize],[Disable optimization]))

optimization_flags="-fPIC -O3 -ffast-math -mtune=native -ftree-vectorize"

AS_IF([test "x$enable_optimize" == "xno"],[
  optimization_flags="-fPIC -g"
  CFLAGS=""
  CXXFLAGS=""
])

conf_flags=""

dnl -fPIC is needed on Linux to build shared libraries

if [[ $cpufamily = "x86_64" ]]; then
    optimization_flags="-fPIC -O3 -g -DAMD64 -ffast-math -mtune=native -ftree-vectorize"

    AS_IF([test "x$enable_optimize" == "xno"],[
      optimization_flags="-fPIC -g -DAMD64"
      CFLAGS=""
      CXXFLAGS=""
    ])

    conf_flags="-DAMD64"
elif [[ "$cpufamily" = "i386" -o "$cpufamily" = "i486" -o "$cpufamily=i586" -o "$cpufamily" = "i686" -o "$cpufamily" = "i786" -o "$cpufamily" = "i886" -o "$cpufamily" = "i986" ]]; then

    optimization_flags="-fPIC -O3 -g -DIA32 -ffast-math -mtune=native -ftree-vectorize"

    AS_IF([test "x$enable_optimize" == "xno"],[
      optimization_flags="-fPIC -g -DIA32"
      CFLAGS=""
      CXXFLAGS=""
    ])

    conf_flags="-DIA32"
else
    AC_MSG_ERROR($cpufamily [, unrecognized/unsupported machine architecture.])
fi


AC_ARG_ENABLE([openmp],
	AS_HELP_STRING([--disable-openmp],[Disable OpenMP]))

AS_IF([test "x$enable_openmp" != "xno"],[
  conf_flags="$conf_flags -fopenmp"
  optimization_flags="$optimization_flags -fopenmp"
],[
  conf_flags="$conf_flags -Wno-unknown-pragmas"
  optimization_flags="$optimization_flags -Wno-unknown-pragmas"
])



AC_MSG_CHECKING([CPUID identification])
CPUINFO_FILE="cpuinfo"
`rm -rf cputest cpuinfo &> /dev/null`
`gcc cputest.c -o cputest && ./cputest > cpuinfo`

if [[ $? -eq 0 ]]; then
    AC_MSG_RESULT([ok])
else
    AC_MSG_RESULT([failure])
    `$FGREP flags /proc/cpuinfo > cpuinfo`
fi

dnl check for GMP integer library
AC_MSG_CHECKING([for GMP library])
GMP_LIBRARY="no"

`gcc -o gmp gmptest.c -I/usr/include/ -L/usr/lib/ -I/usr/local/include/ -L/usr/local/lib/ -lgmp > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  GMP_LIBRARY="yes"
  AC_MSG_RESULT([yes])
else
  AC_MSG_ERROR([no])
fi


dnl checking for CBLAS (BLAS) header files and (and library)
AC_MSG_CHECKING([for BLAS C headers (1)])
CBLAS_HEADERS="no"

`gcc -c blastest.c -I/usr/local/include/ -L/usr/local/lib/ > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
    CBLAS_HEADERS="yes"
    BLAS_CFLAGS="-I/usr/local/include/"
    BLAS_LIBS="-lblas -L/usr/local/lib/"
    AC_MSG_RESULT([yes])
else
    AC_MSG_RESULT([no cblas.h detected])
fi


AC_MSG_CHECKING([for BLAS C headers (2)])

`gcc -c blastest.c -I/opt/OpenBLAS/include/ -L/usr/local/lib/ > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
    CBLAS_HEADERS="yes"
    BLAS_CFLAGS="-I/opt/OpenBLAS/include/"
    BLAS_LIBS="-lblas -L/usr/local/lib/"
    AC_MSG_RESULT([yes])
else
    AC_MSG_RESULT([no cblas.h detected])
fi

AC_MSG_CHECKING([for BLAS C headers (3)])

`gcc -c blastest.c -I/opt/include/ -L/opt/lib/ > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
    CBLAS_HEADERS="yes"
    BLAS_CFLAGS="-I/opt/include/"
    BLAS_LIBS="-lblas -L/opt/lib/"
    AC_MSG_RESULT([yes])
else
    AC_MSG_RESULT([no cblas.h detected])
fi


if [[ "$CBLAS_HEADERS" = "no" ]]; then
   AC_MSG_ERROR([Cannot find cblas.h])
fi

AC_ARG_ENABLE([openblas],
	AS_HELP_STRING([--disable-openblas],[Disable OpenBLAS library]))
 
AC_MSG_CHECKING([for OpenBLAS library (1)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/usr/local/include/ -L/usr/local/lib/ -lopenblas > /dev/null 2> /dev/null`
 
if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/usr/local/include/"
     BLAS_LIBS="-lopenblas -L/usr/local/lib/"
     AC_MSG_RESULT([yes, headers in /usr/local/include/])
  ])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi

AC_MSG_CHECKING([for OpenBLAS library (2)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/include/ -L/lib/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/include/"
     BLAS_LIBS="-L/lib/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /include/])
  ])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi

AC_MSG_CHECKING([for OpenBLAS library (3)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/usr/include/openblas -L/lib/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/usr/include/openblas/"
     BLAS_LIBS="-L/lib/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /usr/include/openblas/])
   ])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi   
fi

AC_MSG_CHECKING([for OpenBLAS library (4)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/usr/local/include/openblas -L/lib/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/usr/local/include/openblas/"
     BLAS_LIBS="-L/lib/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /usr/local/include/openblas/])
   ])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi

AC_MSG_CHECKING([for OpenBLAS library (5)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/opt/OpenBLAS/include/ -L/opt/OpenBLAS/bin/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/opt/OpenBLAS/include/"
     BLAS_LIBS="-L/opt/OpenBLAS/bin/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /opt/OpenBLAS/include/])
   ])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi

AC_MSG_CHECKING([for OpenBLAS library (6)])
`gcc -o openblastest openblastest.c -DOPENBLAS -I/opt/include/ -L/opt/lib/ -lopenblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS -I/opt/include/"
     BLAS_LIBS="-L/opt/lib/ -lopenblas"
     AC_MSG_RESULT([yes, headers in /opt/include/])
   ])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi


AC_MSG_CHECKING([for OpenBLAS library (7)])
obcflags=`pkg-config openblas --cflags 2> /dev/null`
oblibs=`pkg-config openblas --libs 2> /dev/null`
`gcc -o openblastest openblastest.c -DOPENBLAS $obcflags $oblibs > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_openblas" == "xno"],[
     OPENBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     OPENBLAS="yes"
     BLAS_CFLAGS="-DOPENBLAS `pkg-config openblas --cflags`"
     BLAS_LIBS="`pkg-config openblas --libs`"
     AC_MSG_RESULT([yes, pkg-config conf file])
   ])
else
   AC_MSG_RESULT([no OpenBLAS detected])
   if [[ "x$OPENBLAS" != "xyes" ]]; then
      OPENBLAS="no"
   fi
fi


AC_ARG_ENABLE([mkl],
	AS_HELP_STRING([--disable-mkl],[Disable Intel MKL library]))

MKLROOT="/opt/intel/compilers_and_libraries/linux/mkl"
AC_MSG_CHECKING([for Intel MKL library (1) (Linux)])
`g++ -o openblastest openblastest.c -DINTELMKL -I$MKLROOT/include/ -L$MKLROOT/lib/intel64_lin -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_mkl" == "xno"],[
     INTELMKL="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     INTELMKL="yes"
     BLAS_CFLAGS="-m64 -I$MKLROOT/include -DINTELMKL"
     BLAS_LIBS="-L$MKLROOT/lib/intel64_lin -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
     AC_MSG_RESULT([yes, installed in $MKLROOT])
  ])
else
   AC_MSG_RESULT([no Intel MKL detected])
   INTELMKL="no"
fi


MKLROOT="/opt/intel/psxe_runtime/linux/mkl"
AC_MSG_CHECKING([for Intel MKL library (2) (Linux)])
`g++ -o openblastest openblastest.c -DINTELMKL -I$MKLROOT/include/ -L$MKLROOT/lib/intel64_lin -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_mkl" == "xno"],[
     INTELMKL="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     INTELMKL="yes"
     BLAS_CFLAGS="-m64 -I$MKLROOT/include -DINTELMKL"
     BLAS_LIBS="-L$MKLROOT/lib/intel64_lin -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
     AC_MSG_RESULT([yes, installed in $MKLROOT])
  ])
else
   AC_MSG_RESULT([no Intel MKL detected])
   INTELMKL="no"
fi

dnl Intel MKL in Windows
MKLROOT=/c/Program\ Files\ \(x86\)/IntelSWTools/compilers_and_libraries/windows/mkl

AC_MSG_CHECKING([for Intel MKL library (Windows)])
`g++ -o openblastest openblastest.c -DINTELMKL -I$MKLROOT/include/ -L$MKLROOT/lib/intel64_win -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lgomp -lm > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_mkl" == "xno"],[
     INTELMKL="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     INTELMKL="yes"
     BLAS_CFLAGS="-m64 -I$MKLROOT/include -DINTELMKL"
     BLAS_LIBS="-L$MKLROOT/lib/intel64_win -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lgomp -lpthread -lm -ldl"
     AC_MSG_RESULT([yes, installed in $MKLROOT])
  ])
else
   AC_MSG_RESULT([no working Intel MKL detected])
   INTELMKL="no"
fi

dnl AMD BLIS library support
AC_ARG_ENABLE([amdblis],
	AS_HELP_STRING([--disable-amdblis],[Disable AMD BLIS (BLAS) library]))
 
AC_MSG_CHECKING([for AMD BLIS library])
`gcc -o openblastest openblastest.c -DAMDBLIS -I/opt/amd-blis/include -L/opt/amd-blis/lib/ -lblis-mt > /dev/null 2> /dev/null`

dnl if [[ $? -eq 0 ]]; then
dnl    AS_IF([test "x$enable_amdblis" == "xno"],[
dnl       AMDBLIS="no"
dnl       AC_MSG_RESULT([yes, but DISABLED])
dnl    ],
dnl    [
dnl       AMDBLIS="yes"
dnl       BLAS_CFLAGS="-DAMDBLIS -I/opt/amd-blis/include/"
dnl       BLAS_LIBS="-lblis-mt -L/opt/amd-blis/lib/"
dnl       AC_MSG_RESULT([yes, headers in /opt/amd-blis/include/])
dnl    ])
dnl  else
dnl    AC_MSG_RESULT([no AND detected])
dnl    if [[ "x$AMDBLIS" != "xyes" ]]; then
dnl       AMDBLIS="no"
dnl    fi
dnl fi
   AC_MSG_RESULT([AMD BLIS currently doesn't support full CBLAS!])
   if [[ "x$AMDBLIS" != "xyes" ]]; then
      AMDBLIS="no"
   fi



AC_ARG_ENABLE([cublas],
	AS_HELP_STRING([--disable-cublas],[Disable NVIDIA cuBLAS library]))

CUBLASROOT="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
cublasdir=`ls -1 -Art "$CUBLASROOT" 2> /dev/null | tail -n 1`
CUBLASROOT="$CUBLASROOT/$cublasdir"

AC_MSG_CHECKING([for NVIDIA cuBLAS library [SLOW] (Windows)])
`g++ -o cublastest cublastest.cpp -DCUBLAS "-I$CUBLASROOT/include/" "-L$CUBLASROOT/lib/x64/" -lcublas -lcudart > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_cublas" == "xno"],[
     NVIDIACUBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     NVIDIACUBLAS="yes"
     BLAS_CFLAGS="\"-I$CUBLASROOT/include/\" -DCUBLAS"
     BLAS_LIBS="\"-L$CUBLASROOT/lib/x64/\" -lcublas -lcudart"
     AC_MSG_RESULT([yes, installed in $CUBLASROOT])
  ])
else
   AC_MSG_RESULT([no NVIDIA cuBLAS detected])
   NVIDIACUBLAS="no"
fi


AC_ARG_ENABLE([nvblas],
	AS_HELP_STRING([--disable-nvblas],[Disable NVIDIA NVBLAS library]))

CUBLASROOT="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"
cublasdir=`ls -1 -Art "$CUBLASROOT" 2> /dev/null | tail -n 1`
CUBLASROOT="$CUBLASROOT/$cublasdir"

AC_MSG_CHECKING([for NVIDIA NVBLAS library with CBLAS inteface (don't exist)])
`g++ -o openblastest openblastest.c -DNVBLAS "-I$CUBLASROOT/include/" "-L$CUBLASROOT/lib/x64/" -lnvblas > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
  AS_IF([test "x$enable_nvblas" == "xno"],[
     NVIDIANVBLAS="no"
     AC_MSG_RESULT([yes, but DISABLED])
  ],
  [
     NVIDIANVBLAS="yes"
     BLAS_CFLAGS="\"-I$CUBLASROOT/include/\" -DNVBLAS"
     BLAS_LIBS="\"-L$CUBLASROOT/lib/x64/\" -lnvblas"
     AC_MSG_RESULT([yes, installed in $CUBLASROOT])
  ])
else
   AC_MSG_RESULT([no NVIDIA NVBLAS with CBLAS detected])
   NVIDIANVBLAS="no"
fi


if [[ $INTELMKL == "no" -a $OPENBLAS == "no" ]]; then
   if [[ $NVIDIACUBLAS == "no" -a $NVIDIANVBLAS == "no" ]]; then
      if [[ $AMDBLIS == "no" ]]; then
           AC_MSG_ERROR([NO BLAS DETECTED])
      fi
   fi
fi


dnl checking for ZLIB header files and library
AC_MSG_CHECKING([for ZLIB library and headers])
ZLIB_USABLE="no"

`gcc -o zlibtest zlibtest.c -I/usr/local/include/ -L/usr/local/lib/ -lz > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
   ZLIB_USABLE="yes"
   AC_MSG_RESULT([yes])
else
   AC_MSG_ERROR([no, install ZLIB and header files])
fi


dnl adds --with-atlaslibpath
dnl ATLAS support is NOT working / is broken!!!
ATLASLIBS_PATH="/usr/local/lib/"

AC_ARG_WITH([atlaslpath], AS_HELP_STRING([--with-atlaslpath=PATH],[sets search path for ATLAS libraries]),
			  [if [[ -n $withval ]]; then ATLASLIBS_PATH=$withval; fi], [])


optimization_flags="$optimization_flags -mtune=native"


AC_ARG_ENABLE([native],
	AS_HELP_STRING([--enable-native],[Enable native optimization]))

AS_IF([test "x$enable_native" = "xyes"],[
  optimization_flags="$optimization_flags -march=native"
])

AC_ARG_ENABLE([cppdebug],
	AS_HELP_STRING([--enable-cppdebug],[Enable C++ debugging flags (stdc++)]))

AS_IF([test "x$enable_cppdebug" = "xyes"],[
dnl  optimization_flags="$optimization_flags -D_GLIBCXX_DEBUG -fsanitize=address -fsanitize=undefined"
  optimization_flags="$optimization_flags -D_GLIBCXX_DEBUG -fsanitize=address"
  conf_flags="$conf_flags -D_GLIBCXX_DEBUG -fsanitize=address"
])

dnl enable newer C++ standards
dnl OLD: optimization_flags="-std=c++17 $optimization_flags"
optimization_flags="$optimization_flags"

dnl 
dnl AC_ARG_ENABLE([profiling],
dnl	AS_HELP_STRING([--enable-profiling],[Enable code profiling (speed) mode (-pg)]))
dnl 
dnl AS_IF([test "x$enable_profiling" = "xyes"],[
dnl   optimization_flags="$optimization_flags -pg"
dnl ])
dnl 


EXTRA_INCLUDEPATHS="-I/usr/include/ $EXTRA_INCLUDEPATHS $BLAS_CFLAGS $EXTRA_DEFINES"
EXTRA_LIBS="-L/usr/lib/ $EXTRA_LIBS $BLAS_LIBS -lstdc++"

AC_SUBST([EXTRA_INCLUDEPATHS],[$EXTRA_INCLUDEPATHS])
AC_SUBST([EXTRA_LIBPATHS],[$EXTRA_LIBPATHS])
AC_SUBST([EXTRA_LIBS],[$EXTRA_LIBS])
AC_SUBST([optimization_flags],[$optimization_flags])
AC_SUBST([conf_flags], [$conf_flags])
AC_SUBST([conf_libs], [$conf_libs])
AC_SUBST([SHAREDLIBSUFFIX], [$SHAREDLIBSUFFIX])

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile src/Makefile src/math/Makefile])
AC_CONFIG_FILES([src/crypto/Makefile src/neuralnetwork/Makefile])
AC_CONFIG_FILES([src/datamining/Makefile src/hmm/Makefile])
AC_CONFIG_FILES([src/reinforcement/Makefile src/rnn/Makefile])
AC_CONFIG_FILES([tools/Makefile])
AC_CONFIG_FILES([dinrhiw.pc])
AC_OUTPUT


