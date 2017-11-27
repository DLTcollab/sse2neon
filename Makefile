#makefile to compile multiple Cpp files

#user defined variables

#The C compiler to use 
CC = g++
#C flags
CFLAGS = -Wall -mfpu=neon
#Preprocessor flags like include paths
CPPFLAGS+=	-I. 
#Add libraries to this, e.g. -lm for the math library
LDFLAGS	:= -lm
#List of C source files
SRCS	:=	$(wildcard *.cpp)
#program name
BINARY = test.out
#header files and dependcies
DEPS = SSE2NEON.h SSE2NEONBinding.h SSE2NEONTEST.h


$(BINARY) : $(SRCS) $(DEPS) 
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $(SRCS) -o $@
# $@ is an inbuilt (automatic) variable that gives the name of the target
#$< is an inbuilt (automatic) variable that gives the name of the first (or only) prerequisite.

.PHONY: clean    
clean :
	-rm $(BINARY) *.o  *.out 2> /dev/null
