CXX = g++
ARCH_CFLAGS = -march=armv8-a+fp+simd -mtune=thunderx
CXXFLAGS = -Wall -Wcast-qual -I. $(ARCH_CFLAGS) -MMD -std=gnu++14
LDFLAGS	= -lm
OBJS = \
    tests/binding.o \
    tests/impl.o \
    tests/main.o
deps := $(OBJS:%.o=%.d)

EXEC = tests/main

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

check: tests/main
	$^

.PHONY: clean    
clean:
	$(RM) $(OBJS) $(EXEC) $(deps)

-include $(deps)
