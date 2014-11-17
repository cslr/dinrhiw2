# GNU Makefile
#
##################################################

CC= gcc
CXX= g++

OBJECTS= Edge.o Automata.o State.o automata_tests.o 
TARGET= atest
CXXFLAGS= -Wall -g




##################################################

all: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

clean:
	$(RM) $(OBJECTS) *~
	$(RM) $(TARGET)

##################################################
