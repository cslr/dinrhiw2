
CXX=g++

CFLAGS = -Wall -g
CXXFLAGS = $(CFLAGS)

OBJECTS = unique_id.o tests.o
TARGET  = testcode


##################################################

all: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

clean:
	$(RM) $(OBJECTS)
	$(RM) $(TARGET)
	$(RM) *~


##################################################