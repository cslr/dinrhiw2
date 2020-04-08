# Datamining Makefile
###################################################

CC = gcc
CXX= g++

OPTIMIZE=-fPIC -O3 -g -DAMD64 -ffast-math -fopenmp -mtune=native
CFLAGS += $(OPTIMIZE) -Wall -g -I. -I.. -I../../ -I../neuralnetwork -I../math -I/usr/include/ -I/usr/local/include -m64 -I/opt/intel/compilers_and_libraries/linux/mkl/include -DINTELMKL 
CXXFLAGS += $(CFLAGS) -I@atlas_headerpath@


OBJECTS = AssociationRuleFinder.o test/tst.o

EXTRA_OBJECTS = ../math/ownexception.o ../math/integer.o ../dynamic_bitset.o ../timed_boolean.o ../Log.o


SOURCES = AssociationRuleFinder.cpp test/tst.cpp \
	../dynamic_bitset.cpp ../timed_boolean.cpp ../math/ownexception.cpp ../math/integer.cpp


TARGET  = tst

LIBS= -L/usr/local/lib -lz -lgmp -L/usr/lib/  -L/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64_lin -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

##################################################

all: $(OBJECTS) $(EXTRA_OBJECTS)
	$(CXX) -o $(TARGET) $(CXXFLAGS) $(OBJECTS) $(EXTRA_OBJECTS) $(LIBS)

clean:
	$(RM) $(OBJECTS) $(TARGET)
	$(RM) *~ test/*~

clean_rpo:
	$(RM) *.rpo
	$(RM) test/*.rpo

depend:
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) > Makefile.depend

##################################################

include Makefile.depend
