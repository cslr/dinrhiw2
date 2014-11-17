# nenya-server ORBit C++ GNU Makefile
# (C) Tomas Ukkonen

CC = gcc
CXX = g++

ORBIT_IDL = orbit-idl-2
ORBITCPP_IDL = $(ORBIT_IDL)

OPTIMIZE = 
# -fomit-frame-pointer 
# -ffast-math -O3 -march=pentium 


CFLAGS = -Wall -Werror -g $(OPTIMIZE) \
	-I.. -I../memory -I../image -I../system -I../dos -I../../ \
	`pkg-config dinrhiw --cflags` `pkg-config ORBit-2.0-cpp --cflags`

CXXFLAGS = $(CFLAGS)


CORBA_INTERFACE=LearningSystem

include Makefile.CORBA

CORBA_SERVER_SOURCES += nenya-server.cpp
CORBA_SERVER_OBJECTS += nenya-server.o

SERVER_TARGET  = nenya-server
CLIENT_TARGET  = nenya-server-test


COMMON_OBJECTS =

HEADERS = $(CORBA_SERVER_HEADERS) $(CORBA_CLIENT_HEADERS)
SOURCES = $(CORBA_SERVER_SOURCES) $(CORBA_CLIENT_SOURCES)
OBJECTS = $(COMMON_OBJECTS)

LIBS = `pkg-config algos --libs` `pkg-config dlib --libs` \
	`pkg-config freetype2 --libs` `pkg-config ORBit-2.0-cpp --libs` \
	-lm -lpng -lz

SERVER_LIBS = $(LIBS)
CLIENT_LIBS = $(LIBS) 


###########################################################


all: $(SERVER_TARGET) $(CLIENT_TARGET)

$(SERVER_TARGET): idltargets $(CORBA_SERVER_OBJECTS)
	$(CXX) -o $(SERVER_TARGET) $(CXXFLAGS) $(LDFLAGS) \
	$(CORBA_SERVER_OBJECTS) $(SERVER_LIBS)

$(CLIENT_TARGET): $(CORBA_CLIENT_OBJECTS)
	$(CXX) -o $(CLIENT_TARGET) $(CXXFLAGS) $(LDFLAGS) \
	$(CORBA_CLIENT_OBJECTS) $(CLIENT_LIBS)

clean:
	$(RM) $(OBJECTS)
	$(RM) $(SERVER_TARGET) $(CLIENT_TARGET)
	$(RM) $(CORBA_SERVER_OBJECTS) $(CORBA_CLIENT_OBJECTS)
	$(RM) *~

idltargets: $(IDL_OBJECTS)

idlclean:
	$(RM) $(IDL_OBJECTS)

realclean: clean idlclean

depend: $(SOURCES) $(HEADERS)
	$(CXX) -MM $(SOURCES) > Make.depend


###########################################################

%.o : %.cpp ; $(CXX) -c $(CXXFLAGS)  $< -o $@

# IDL compilation rules
%-common.c %-skelimpl.c %-skels.c %-stubs.c %-stubs.h %.h: %.idl
	$(ORBIT_IDL) --lang=c --skeleton-impl $<

%-cpp-common.cc %-cpp-common.h %-cpp-skels.h %-cpp-skels.cc %-cpp-stubs.cc %-cpp.h: %.idl
	$(ORBITCPP_IDL) --lang=cpp $<


include Make.depend
