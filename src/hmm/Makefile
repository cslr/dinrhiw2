# dinrhiw2 basic/generic classes
##################################################

CC = gcc
CXX= g++

OBJECTS = HMM.o

EXTRA_OBJECTS = ../dataset.o ../MemoryCompressor.o \
	../math/vertex.o ../math/matrix.o ../math/ownexception.o \
	../math/integer.o ../math/correlation.o ../math/matrix_rotations.o \
	../math/eig.o ../math/blade_math.o ../math/real.o ../math/ica.o \
	../linear_ETA.o ../Log.o ../math/norms.o

TEST_OBJECTS = $(OBJECTS) $(EXTRA_OBJECTS) tst/test.o


SOURCES = HMM.cpp tst/test.cpp \
	../dataset.cpp \
	../math/vertex.cpp \
	../math/integer.cpp ../math/blade_math.cpp ../math/real.cpp \
	../linear_ETA.cpp \
	../math/ownexception.cpp ../math/norms.cpp

OPTIMIZE=-fPIC -O3 -g -DAMD64 -ffast-math -fopenmp -mtune=native
CFLAGS=-g -O2 -Wno-deprecated -Wno-strict-aliasing -Wno-attributes -I/usr/include/ -I/usr/local/include -DOPENBLAS -I/opt/OpenBLAS/include/ -DWINOS

TARGET= testsuite
CXXFLAGS=-Wall $(CFLAGS) -g -I. -I.. -Iexceptions -I../math -I../neuralnetwork $(OPTIMIZE)

LIBS= -L/usr/local/lib -lz -lgmp -L/usr/lib/  -L/opt/OpenBLAS/bin/ -lopenblas

CD = cd


##################################################

all: $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(TEST_OBJECTS) $(LIBS)

clean:
	$(RM) $(TEST_OBJECTS)
	$(RM) $(TARGET)
	$(RM) *~

realclean: clean
	$(RM) hmm1.dat

depend:
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) > Makefile.depend


##################################################

include Makefile.depend





