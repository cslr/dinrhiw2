# CRYPTO Makefile
###################################################

CC = gcc
CXX = g++

OPTIMIZE=-fPIC -O3 -g -DAMD64 -ffast-math -fopenmp -mtune=native

CFLAGS += -g -O2 -Wall -Wno-deprecated -Wno-strict-aliasing -g -I. -I.. -I../.. -I../neuralnetwork -I../math -I/usr/local/include/ $(OPTIMIZE) -I/usr/include/ -I/usr/local/include -DOPENBLAS -I/opt/OpenBLAS/include/ -DWINOS
CXXFLAGS = $(CFLAGS)


OBJECTS = PAD.o DES.o AES.o SHA.o RSA.o DSA.o test/tst.o

EXTRA_OBJECTS = ../math/ownexception.o ../math/integer.o \
	../dynamic_bitset.o ../Log.o


SOURCES = PAD.cpp DES.cpp AES.cpp SHA.cpp RSA.cpp DSA.cpp \
	../math/ownexception.cpp ../math/integer.cpp \
	../dynamic_bitset.cpp \
	test/tst.cpp


TARGET  = tst
LIBS = -L/usr/lib/ -L/usr/local/lib -lgmp

# nbits_integer.o 
# Bitset.o

##################################################

all: $(OBJECTS) $(EXTRA_OBJECTS)
	$(CXX) -o $(TARGET) $(CXXFLAGS) $(OBJECTS) $(EXTRA_OBJECTS) $(LIBS)

clean:
	$(RM) $(OBJECTS) $(TARGET)
	$(RM) *~ test/*~

clean_rpo:
	$(RM) *.rpo
	$(RM) test/*.rpo

realclean: clean clean_rpo

depend:
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) > Makefile.depend

##################################################

include Makefile.depend
