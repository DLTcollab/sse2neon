ifndef CXX
override CXX = g++
endif

processor := $(shell uname -p)
ifeq ($(processor),aarch64)
  ARCH_CFLAGS = -march=armv8-a+fp+simd
else ifeq ($(processor),$(filter $(processor),i386 x86_64))
  ARCH_CFLAGS = -maes -mpclmul -mssse3 -msse4.2
else
  ARCH_CFLAGS =
endif
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
