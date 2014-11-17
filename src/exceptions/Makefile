#
# exception directory makefile
# exceptions and testcode

CC = gcc
CXX = g++

CFLAGS += -Wall -Werror -g
CXXFLAGS = $(CFLAGS)
LIBS +=

OBJECTS=ownexception.o
TARGET_OBJECT=testsuite.o
TARGET=testsuite


##################################################

all: $(OBJECTS)

test: $(TARGET_OBJECT) $(OBJECTS)
	$(CXX) -o $(TARGET) $(CXXFLAGS) $(OBJECTS) $(TARGET_OBJECT) $(LIBS)

clean:
	$(RM) $(OBJECTS)
	$(RM) $(TARGET_OBJECT)
	$(RM) $(TARGET)
	$(RM) *~
