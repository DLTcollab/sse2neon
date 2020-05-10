ifndef CXX
override CXX = g++
endif

EXEC_WRAPPER =
ifndef CROSS_COMPILE
# Follow platform-specific configurations
processor := $(shell uname -p)
ifeq ($(processor),aarch64)
  ARCH_CFLAGS = -march=armv8-a+fp+simd
else ifeq ($(processor),$(filter $(processor),i386 x86_64))
  ARCH_CFLAGS = -maes -mpclmul -mssse3 -msse4.2
else
  ARCH_CFLAGS =
endif
else # CROSS_COMPILE was set
CXX = $(CROSS_COMPILE)g++
CXXFLAGS += -static
LDFLAGS += -static
check_arm := $(shell echo | $(CROSS_COMPILE)cpp -dM - | grep " __ARM_ARCH " | cut -c20-)
ifeq ($(check_arm),8)
  EXEC_WRAPPER = qemu-aarch64
else ifeq ($(check_arm),7)
  ARCH_CFLAGS += -mfpu=neon
  EXEC_WRAPPER = qemu-arm
else
  EXEC_WRAPPER =
endif
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

.PHONY: clean    
clean:
	$(RM) $(OBJS) $(EXEC) $(deps)

-include $(deps)
