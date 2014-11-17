/*
 * some typedefs which are handy when writing portable code.
 * (C) Copyright Tomas Ukkonen
 * [nop@iki.fi]
 *
 * QWORD_DEFINED, DWORD_DEFINED, WORD_DEFINED, BYTE_DEFINED
 *  -> if there's 8 byte, 4 byte, 2 byte, 1 byte data types available.
 *
 * to do: endianess information
 *
 * uint is a unsigned int.
 * ushort is a unsigned short.
 * uchar is a unsigned char.
 *
 * sint is a signed int.
 * sshort is a signed short.
 * schar is a signed char.
 *
 * int32, sint32 and short32 are
 * 32bit integers.
 *
 * sshort32, char32 and schar32 are
 * signed 32bit integers.
 *
 * int16, sint16 and short16 are
 * 16bit integers.
 *
 * sshort16, char16 and schar16 are
 * signed 16bit integers.
 *
 * int8, sint8, and short8
 * 8bit integers.
 *
 * sshort8, char8 and schar8 are
 * signed 8bit integers.
 *
 * dword, uint32, ushort32 and uchar32 are unsigned 32bit integers.
 * word, uint16, ushort16 and uchar16 are unsigned 16bit integers.
 * byte, uint8, ushort8 and uchar8 are unsigned 8bit integers.
 *
 */

#ifndef _global_h_
#define _global_h_
 
// include this in "*.c" files

#include "config.h"

namespace whiteice
{

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

typedef signed int sint;
typedef signed short sshort;
typedef signed char schar;


#if SIZEOF_LONG_LONG_INT == 8
#ifndef QWORD_DEFINED
#define QWORD_DEFINED 1

typedef long long int int64;
typedef long long int short64;
typedef long long int char64;
typedef signed long long int sint64;
typedef signed long long int sshort64;
typedef signed long long int schar64;
typedef unsigned long long int uint64;
typedef unsigned long long int ushort64;
typedef unsigned long long int uchar64;
typedef unsigned long long int qword;

#endif
#endif


#if SIZEOF_LONG == 8
#ifndef QWORD_DEFINED
#define QWORD_DEFINED 1

typedef long int64;
typedef long short64;
typedef long char64;
typedef signed long sint64;
typedef signed long sshort64;
typedef signed long schar64;
typedef unsigned long uint64;
typedef unsigned long ushort64;
typedef unsigned long uchar64;
typedef unsigned long qword;

#endif
#endif


#if SIZEOF_LONG == 4
#ifndef DWORD_DEFINED
#define DWORD_DEFINED 1

typedef long int32;
typedef long short32;
typedef long char32;
typedef signed long sint32;
typedef signed long sshort32;
typedef signed long schar32;
typedef unsigned long uint32;
typedef unsigned long ushort32;
typedef unsigned long uchar32;
typedef unsigned long dword;

#endif
#endif

#if SIZEOF_LONG == 2
#ifndef WORD_DEFINED
#define WORD_DEFINED 1

typedef long int16;
typedef long short16;
typedef long char16;
typedef signed long sint16;
typedef signed long sshort16;
typedef signed long schar16;
typedef unsigned long uint16;
typedef unsigned long ushort16;
typedef unsigned long uchar16;
typedef unsigned long word;

#endif
#endif


#if SIZEOF_LONG == 1
#ifndef BYTE_DEFINED
#define BYTE_DEFINED 1

typedef long int8;
typedef long short8;
typedef long char8;
typedef signed long sint8;
typedef signed long sshort8;
typedef signed long schar8;
typedef unsigned long uint8;
typedef unsigned long ushort8;
typedef unsigned long uchar8;
typedef unsigned long byte;

#endif
#endif


#if SIZEOF_INT == 8
#ifndef QWORD_DEFINED
#define QWORD_DEFINED 1

typedef int int64;
typedef int short64;
typedef int char64;
typedef signed int sint64;
typedef signed int sshort64;
typedef signed int schar64;
typedef unsigned int uint64;
typedef unsigned int ushort64;
typedef unsigned int uchar64;
typedef unsigned int qword;

#endif
#endif


#if SIZEOF_INT == 4
#ifndef DWORD_DEFINED
#define DWORD_DEFINED 1

typedef int int32;
typedef int short32;
typedef int char32;
typedef signed int sint32;
typedef signed int sshort32;
typedef signed int schar32;
typedef unsigned int uint32;
typedef unsigned int ushort32;
typedef unsigned int uchar32;
typedef unsigned int dword;

#endif
#endif

#if SIZEOF_INT == 2
#ifndef WORD_DEFINED
#define WORD_DEFINED 1

typedef int int16;
typedef int short16;
typedef int char16;
typedef signed int sint16;
typedef signed int sshort16;
typedef signed int schar16;
typedef unsigned int uint16;
typedef unsigned int ushort16;
typedef unsigned int uchar16;
typedef unsigned int word;

#endif
#endif

#if SIZEOF_INT == 1
#ifndef BYTE_DEFINED
#define BYTE_DEFINED 1

typedef int int8;
typedef int short8;
typedef int char8;
typedef signed int sint8;
typedef signed int sshort8;
typedef signed int schar8;
typedef unsigned int uint8;
typedef unsigned int ushort8;
typedef unsigned int uchar8;
typedef unsigned int byte;

#endif
#endif


#if SIZEOF_SHORT == 8
#ifndef QWORD_DEFINED
#define QWORD_DEFINED 1

typedef short int64;
typedef short short64;
typedef short char64;
typedef signed short sint64;
typedef signed short sshort64;
typedef signed short schar64;
typedef unsigned short uint64;
typedef unsigned short ushort64;
typedef unsigned short uchar64;
typedef unsigned short qword;

#endif
#endif


#if SIZEOF_SHORT == 4
#ifndef DWORD_DEFINED
#define DWORD_DEFINED 1

typedef short int32;
typedef short short32;
typedef short char32;
typedef signed short sint32;
typedef signed short sshort32;
typedef signed short schar32;
typedef unsigned short uint32;
typedef unsigned short ushort32;
typedef unsigned short uchar32;
typedef unsigned short dword;

#endif
#endif

#if SIZEOF_SHORT == 2
#ifndef WORD_DEFINED
#define WORD_DEFINED 1

typedef short int16;
typedef short short16;
typedef short char16;
typedef signed short sint16;
typedef signed short sshort16;
typedef signed short schar16;
typedef unsigned short uint16;
typedef unsigned short ushort16;
typedef unsigned short uchar16;
typedef unsigned short word;

#endif
#endif


#if SIZEOF_SHORT == 1
#ifndef BYTE_DEFINED
#define BYTE_DEFINED 1

typedef short int8;
typedef short short8;
typedef short char8;
typedef signed short sint8;
typedef signed short sshort8;
typedef signed short schar8;
typedef unsigned short uint8;
typedef unsigned short ushort8;
typedef unsigned short uchar8;
typedef unsigned short byte;

#endif
#endif


#if SIZEOF_CHAR == 8
#ifndef QWORD_DEFINED
#define QWORD_DEFINED 1

typedef char int64;
typedef char short64;
typedef char char64;
typedef signed char sint64;
typedef signed char sshort64;
typedef signed char schar64;
typedef unsigned char uint64;
typedef unsigned char ushort64;
typedef unsigned char uchar64;


#endif
#endif


#if SIZEOF_CHAR == 4
#ifndef DWORD_DEFINED
#define DWORD_DEFINED 1

typedef char int32;
typedef char short32;
typedef char char32;
typedef signed char sint32;
typedef signed char sshort32;
typedef signed char schar32;
typedef unsigned char uint32;
typedef unsigned char ushort32;
typedef unsigned char uchar32;
typedef unsigned char dword;

#endif
#endif

#if SIZEOF_CHAR == 2
#ifndef WORD_DEFINED
#define WORD_DEFINED 1

typedef char int16;
typedef char short16;
typedef char char16;
typedef signed char sint16;
typedef signed char sshort16;
typedef signed char schar16;
typedef unsigned char uint16;
typedef unsigned char ushort16;
typedef unsigned char uchar16;
typedef unsigned char word;

#endif
#endif


#if SIZEOF_CHAR == 1
#ifndef BYTE_DEFINED
#define BYTE_DEFINED 1

typedef char int8;
typedef char short8;
typedef char char8;
typedef signed char sint8;
typedef signed char sshort8;
typedef signed char schar8;
typedef unsigned char uint8;
typedef unsigned char ushort8;
typedef unsigned char uchar8;
typedef unsigned char byte;

#endif
#endif

}

#endif
