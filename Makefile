ifndef CXX
override CXX = g++
endif

ifndef CROSS_COMPILE
    processor := $(shell uname -p)
else # CROSS_COMPILE was set
    CXX = $(CROSS_COMPILE)g++
    CXXFLAGS += -static
    LDFLAGS += -static
    check_arm := $(shell echo | $(CROSS_COMPILE)cpp -dM - | grep " __ARM_ARCH " | cut -c20-)
    ifeq ($(check_arm),8)
        processor = aarch64
    else ifeq ($(check_arm),7) # detect ARMv7-A only
        processor = arm
    else
        $(error Unsupported cross-compiler)
    endif
endif

EXEC_WRAPPER =
ifdef CROSS_COMPILE
EXEC_WRAPPER = qemu-$(processor)
endif

# Follow platform-specific configurations
ifeq ($(processor),aarch64)
    ARCH_CFLAGS = -march=armv8-a+fp+simd
else ifeq ($(processor),$(filter $(processor),i386 x86_64))
    ARCH_CFLAGS = -maes -mpclmul -mssse3 -msse4.2
else ifeq ($(processor),arm)
    ARCH_CFLAGS = -mfpu=neon
else
    $(error Unsupported architecture)
endif

CXXFLAGS += -Wall -Wcast-qual -I. $(ARCH_CFLAGS) -std=gnu++14
LDFLAGS	+= -lm
OBJS = \
    tests/binding.o \
    tests/impl.o \
    tests/main.o
deps := $(OBJS:%.o=%.o.d)

.SUFFIXES: .o .cpp
.cpp.o:
	$(CXX) -o $@ $(CXXFLAGS) -c -MMD -MF $@.d $<

EXEC = tests/main

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

check: tests/main
	$(EXEC_WRAPPER) $^

indent:
	clang-format -i sse2neon.h tests/*.cpp tests/*.h

.PHONY: clean check format
clean:
	$(RM) $(OBJS) $(EXEC) $(deps)

-include $(deps)
