CXX = g++
ARCH_CFLAGS = -march=armv8-a+fp+simd -mtune=thunderx
CXXFLAGS = -Wall -I. $(ARCH_CFLAGS)
LDFLAGS	= -lm
OBJS = \
    tests/binding.o \
    tests/impl.o \
    tests/main.o

EXEC = tests/main

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

check: tests/main
	$^

.PHONY: clean    
clean:
	$(RM) $(OBJS) $(EXEC)
