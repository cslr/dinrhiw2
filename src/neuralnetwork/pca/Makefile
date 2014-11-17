
CXX= g++
OBJECTS= gha.o main.o

TARGET= testsuite

CXXFLAGS= -Wall -Werror -g -O3 -ffast-math -I.
LIBS +=

##################################################

all: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LIBS)

clean:
	$(RM) $(OBJECTS)
	$(RM) $(TARGET)
	$(RM) *~


##################################################

test.o: main.cpp gha.cpp