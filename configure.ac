dnl Process this file with autoconf to produce a configure script.

AC_INIT([whiteice::dinrhiw], [<nop@iki.fi>], [libdinrhiw])

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
AC_SUBST([EXTRA_INCLUDEPATHS],[-I/usr/local/include])

dnl checks if -L/usr/local/lib is needed to
dnl link with libraries in /usr/local/lib
AC_MSG_CHECKING([is -L/usr/local/lib command line parameter needed])

dnl not checked right now, just added
AC_MSG_RESULT([yes (not checked, write real check)])
AC_SUBST([EXTRA_LIBPATHS],[-L/usr/local/lib])


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
optimization_flags="-O3 -ffast-math"
conf_flags=""

if [[ "$cpufamily" = "i386" -o "$cpufamily" = "i486" -o "$cpufamily=i586" -o "$cpufamily" = "i686" -o "$cpufamily" = "i786" -o "$cpufamily" = "i886" -o "$cpufamily" = "i986" ]]; then
    optimization_flags="-O3 -g -DIA32 -ffast-math"
    conf_flags="-DIA32"
fi

if [[ $cpufamily = "x86_64" ]]; then
    optimization_flags="-O3 -g -DAMD64 -ffast-math"
    conf_flags="-DAMD64"
fi

if [test $conf_flags = "" ]; then
    AC_MSG_ERROR($cpufamily [, unrecognized/unsupported machine architecture.])
fi


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

`gcc -o gmp gmptest.c -I /usr/local/include/ -L/usr/local/lib/ -lgmp > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
   GMP_LIBRARY="yes"
   AC_MSG_RESULT([yes])
else
   AC_MSG_ERROR([no])
fi


dnl checking for CBLAS (BLAS) header files and (and library)
AC_MSG_CHECKING([for BLAS C headers])
CBLAS_HEADERS="no"

`gcc -c blastest.c -I/usr/local/include/ -L/usr/local/lib/ > /dev/null 2> /dev/null`

if [[ $? -eq 0 ]]; then
    CBLAS_HEADERS="yes"
    AC_MSG_RESULT([yes])
else
    AC_MSG_ERROR([no, install BLAS implementation with cblas.h])
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
ATLASLIBS_PATH="/usr/local/lib/"

AC_ARG_WITH([atlaslpath], AC_HELP_STRING([--with-atlaslpath=PATH],[sets search path for ATLAS libraries]),
			  [if [[ -n $withval ]]; then ATLASLIBS_PATH=$withval; fi], [])


dnl checking support for different instruction sets

AC_MSG_CHECKING([SSE intruction set extensions])
cpu_sse=`$FGREP -c sse $CPUINFO_FILE`
cpu_sse=`if test $cpu_sse -ge 1; then echo "yes"; else echo "no"; fi`
AC_MSG_RESULT($cpu_sse)

AC_MSG_CHECKING([SSE2 instruction set extensions])
cpu_sse2=`$FGREP -c sse2 $CPUINFO_FILE`
cpu_sse2=`if test $cpu_sse2 -ge 1; then echo "yes"; else echo "no"; fi`
AC_MSG_RESULT($cpu_sse2)

AC_MSG_CHECKING([SSE3 instruction set extensions])
cpu_sse3=`$FGREP -c sse3 $CPUINFO_FILE`
cpu_sse3=`if test $cpu_sse3 -ge 1; then echo "yes"; else echo "no"; fi`
AC_MSG_RESULT($cpu_sse3)

AC_MSG_CHECKING([3dnow! instruction set extensions])
cpu_3dnow=`$FGREP -c 3dnow $CPUINFO_FILE`
cpu_3dnow=`if test $cpu_3dnow -ge 1; then echo "yes"; else echo "no"; fi`
AC_MSG_RESULT($cpu_3dnow)


dnl activates CPU specific optimization flags


if [[ $cpu_sse3 = "yes" ]]; then
   optimization_flags="$optimization_flags -msse3"
elif [[ $cpu_sse2 = "yes" ]]; then
   optimization_flags="$optimization_flags -msse2"
elif [[ $cpu_sse = "yes" ]]; then
   optimization_flags="$optimization_flags -msse"
fi



AC_SUBST([optimization_flags],[$optimization_flags])
AC_SUBST([conf_flags], [$conf_flags])

AC_CONFIG_HEADER(config.h)
AC_CONFIG_FILES([Makefile src/Makefile src/math/Makefile])
AC_CONFIG_FILES([src/crypto/Makefile src/neuralnetwork/Makefile])
AC_CONFIG_FILES([src/datamining/Makefile])
AC_CONFIG_FILES([tools/Makefile])
AC_CONFIG_FILES([dinrhiw.pc])
AC_OUTPUT


