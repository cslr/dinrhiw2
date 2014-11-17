# Makefile
#################################################

CC  = gcc
CXX = g++

OPTIMIZE_FLAGS= -O2 -ffast-math -march=pentium

CFLAGS = -g -O2 -Wall -g  -I. -I.. -I../neuralnetwork -I../crypto \
	-I/usr/local/include `pkg-config dlib --cflags` \
	$(OPTIMIZE_FLAGS)

CXXFLAGS = $(CFLAGS)

OBJECTS = component.o io.o random.o \
	andgate.o orgate.o xorgate.o nandgate.o complement.o \
	test.o

SOURCES = component.cpp io.cpp random.cpp \
	andgate.cpp orgate.cpp xorgate.cpp nandgate.cpp complement.cpp \
	test.cpp

LIBS    = -L/usr/local/lib -lpthread

TARGET  = cputest

##################################################

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -o $(TARGET) $(CXXFLAGS) $(OBJECTS) $(LIBS)

clean:
	$(RM) $(TARGET) $(OBJECTS)
	$(RM) *~

depend:
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) > Makefile.depend


##################################################

include Makefile.depend
