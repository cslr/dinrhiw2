# builds dinrhow library tools/programs.
# requires that dinrhiw2 library has been installed to 
# the system (and have pkg-config installed)
#

CC = gcc
CXX= g++
BISON = bison

CP = cp
RM = rm

OPTIMIZE=-fPIC -O3 -g -DAMD64 -ffast-math -fopenmp -mtune=native

CFLAGS = -g -O2 $(OPTIMIZE) -Wno-deprecated -g -Wall `pkg-config dinrhiw --cflags`
CXXFLAGS = -g -O2  $(CFLAGS)

SOURCES = aescipher.cpp FileSource.cpp CryptoFileSource.cpp dataplot.cpp MMAP.cpp mapping.cpp \
	nntool.cpp argparser.tab.cpp cpuid_threads.cpp datatool.cpp gendata.cpp


AESCIPHER_LIBS = `pkg-config dinrhiw --libs` -lgmp
WDV_LIBS = `pkg-config dinrhiw --libs` -lgmp

AESCIPHER_OBJECTS = aescipher.o FileSource.o CryptoFileSource.o
AESCIPHER_TARGET = aescipher

DATAPLOT_OBJECTS = dataplot.o
DATAPLOT_TARGET = wdv

MMAP_OBJECTS = MMAP.o mapping.o
MMAP_TARGET  = ftest
MMAP_LIBS = 

NNTOOLS_OBJECTS = nntool.o argparser.tab.o cpuid_threads.o
NNTOOLS_TARGET  = nntool
NNTOOLS_LIBS    = `pkg-config dinrhiw --libs` -lpthread

DATATOOL_OBJECTS = datatool.o
DATATOOL_TARGET  = dstool
DATATOOL_LIBS    = `pkg-config dinrhiw --libs` -lpthread

GENDATA_OBJECTS  = gendata.o
GENDATA_TARGET   = gendata
GENDATA_LIBS     = `pkg-config dinrhiw --libs` -lpthread

##################################################

all: aescipher nntools datatool gendata

clean: aescipher_clean dataplot_clean mmap_clean nntools_clean datatool_clean gendata_clean
	$(RM) -f *~

install: all
	$(CP) aescipher /usr/local/bin/
	$(CP) dstool    /usr/local/bin/
	$(CP) nntool /usr/local/bin/

uninstall:
	$(RM) -f /usr/local/bin/aescipher
	$(RM) -f /usr/local/bin/dstool
	$(RM) -f /usr/local/bin/nntool

aescipher: $(AESCIPHER_OBJECTS)
	$(CXX) -o $(AESCIPHER_TARGET) $(CXXFLAGS) $(AESCIPHER_OBJECTS) \
	$(AESCIPHER_LIBS)

aescipher_clean:
	$(RM) -f $(AESCIPHER_TARGET)
	$(RM) -f $(AESCIPHER_OBJECTS)


dataplot: $(DATAPLOT_OBJECTS)
	$(CXX) -o $(DATAPLOT_TARGET) $(CXXFLAGS) $(DATAPLOT_OBJECTS) \
	$(WDV_LIBS)

dataplot_clean:
	$(RM) -f $(DATAPLOT_TARGET)
	$(RM) -f $(DATAPLOT_OBJECTS)

mmaptest: $(MMAP_OBJECTS)
	$(CXX) -o $(MMAP_TARGET) $(CXXFLAGS) $(MMAP_OBJECTS) $(MMAP_LIBS)

mmap_clean:
	$(RM) -f $(MMAP_TARGET)
	$(RM) -f $(MMAP_OBJECTS)

nntools: $(NNTOOLS_OBJECTS)
	$(CXX) -o $(NNTOOLS_TARGET) $(CXXFLAGS) $(NNTOOLS_OBJECTS) $(NNTOOLS_LIBS)

nntools_clean:
	$(RM) -f $(NNTOOLS_TARGET)
	$(RM) -f $(NNTOOLS_OBJECTS)

datatool: $(DATATOOL_OBJECTS)
	$(CXX) -o $(DATATOOL_TARGET) $(CXXFLAGS) $(DATATOOL_OBJECTS) $(DATATOOL_LIBS)	

datatool_clean:
	$(RM) -f $(DATATOOL_TARGET)
	$(RM) -f $(DATATOOL_OBJECTS)

gendata: $(GENDATA_OBJECTS)
	$(CXX) -o $(GENDATA_TARGET) $(CXXFLAGS) $(GENDATA_OBJECTS) $(GENDATA_LIBS)

gendata_clean:
	$(RM) -f $(GENDATA_TARGET)
	$(RM) -f $(GENDATA_OBJECTS)

depend:
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) > Makefile.depend

##################################################

# rule for building parser for command line arguments
argparser.tab.cpp: argparser.ypp
	$(BISON) -v argparser.ypp


include Makefile.depend

