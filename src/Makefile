# dinrhiw2 basic/generic classes
##################################################

CC = gcc
CXX= g++

OBJECTS = point.o static_array.o dynamic_array.o \
	primality_test.o binary_tree.o \
	dataset.o \
	conversion.o unique_id.o \
	conffile.o linear_ETA.o \
	dynamic_bitset.o list_source.o \
	MemoryCompressor.o timed_boolean.o \
	Log.o \
	dinrhiw.o

EXTRA_OBJECTS = math/vertex.o math/matrix.o math/ownexception.o \
	math/integer.o math/correlation.o math/matrix_rotations.o \
	math/eig.o math/blade_math.o math/real.o math/ica.o math/norms.o

TEST_OBJECTS = $(OBJECTS) $(EXTRA_OBJECTS) tst/test.o
# universal_hash.o


SOURCES = point.cpp static_array.cpp dynamic_array.cpp \
	primality_test.cpp binary_tree.cpp \
	dataset.cpp \
	conversion.cpp unique_id.cpp \
	conffile.cpp linear_ETA.cpp \
	MemoryCompressor.cpp timed_boolean.cpp \
	math/vertex.cpp math/matrix.cpp math/ownexception.cpp \
	math/norms.cpp \
	tst/test.cpp tst/conv_test.cpp \
	singleton.cpp singleton_list.cpp \
	dynamic_bitset.cpp list_source.cpp \
	Log.cpp \
	function_access_control.cpp tst/modtest.cpp \
	dinrhiw.cpp tst/test.cpp

CONV_OBJECTS = $(OBJECTS) tst/conv_test.o $(EXTRA_OBJECTS)
CONV_TARGET = convtest

MODIFIER_OBJECTS = singleton.o singleton_list.o \
	function_access_control.o tst/modtest.o \
	exceptions/ownexception.o $(OBJECTS) $(EXTRA_OBJECTS)
MOD_TARGET = modtest

OPTIMIZE=-fPIC -O3 -g -DAMD64 -ffast-math -fopenmp -mtune=native
CFLAGS=-g -O2 -Wno-deprecated -Wno-strict-aliasing -Wno-attributes -I/usr/include/ -I/usr/local/include -DOPENBLAS -I/opt/OpenBLAS/include/ -DWINOS

TARGET= testsuite
CXXFLAGS=-Wall $(CFLAGS) -g -I. -I.. -Iexceptions -Imath -Ineuralnetwork -Icrypto -Ihmm -Ireinforcement -Irnn $(OPTIMIZE)

LIBS= -L/usr/local/lib -lz -lgmp -L/usr/lib/  -L/opt/OpenBLAS/bin/ -lopenblas

CD = cd


##################################################

all: $(TEST_OBJECTS) $(MODIFIER_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TEST_OBJECTS) $(LIBS)

conv: $(CONV_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(CONV_TARGET) $(CONV_OBJECTS) $(LIBS)

mod: $(MODIFIER_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(MOD_TARGET) $(MODIFIER_OBJECTS) $(LIBS)

clean: modclean convclean
	$(RM) $(TEST_OBJECTS)
	$(RM) $(TARGET)
	$(RM) $(CONV_OBJECTS)
	$(RM) $(CONV_TARGET)
	$(RM) *~

clean_rpo:
	$(RM) *.rpo tst/*.rpo

modclean:
	$(RM) $(MODIFIER_OBJECTS)
	$(RM) $(MOD_TARGET)

convclean:
	$(RM) $(CONV_OBJECTS)
	$(RM) $(CONV_TARGET)

realclean: clean clean_rpo

depend:
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) > Makefile.depend


##################################################

include Makefile.depend





