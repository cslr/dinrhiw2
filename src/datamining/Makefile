# Datamining Makefile
###################################################

CC = gcc
CXX= g++

OPTIMIZE=-O3 -g -DIA32 -ffast-math -DBLAS_MEMALIGN -msse3
CFLAGS += $(OPTIMIZE) -Wall -g -I. -I.. -I../neuralnetwork -I../math -I/usr/local/include -DOPENBLAS -I/include/
CXXFLAGS += $(CFLAGS) -I@atlas_headerpath@


OBJECTS = AssociationRuleFinder.o test/tst.o

EXTRA_OBJECTS = ../math/ownexception.o ../math/integer.o ../dynamic_bitset.o ../timed_boolean.o


SOURCES = AssociationRuleFinder.cpp test/tst.cpp \
	../dynamic_bitset.cpp ../timed_boolean.cpp ../math/ownexception.cpp ../math/integer.cpp


TARGET  = tst

LIBS= -L/usr/local/lib -lz -lgmp  -L/lib/ -lopenblas

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
