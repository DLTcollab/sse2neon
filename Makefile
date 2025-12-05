ifndef CC
override CC = gcc
endif

ifndef CXX
override CXX = g++
endif

ifndef CROSS_COMPILE
    processor := $(shell uname -m)
else # CROSS_COMPILE was set
    CC = $(CROSS_COMPILE)gcc
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
ARCH_CFLAGS ?=
ARCH_CFLAGS_IS_SET =
ifeq ($(ARCH_CFLAGS),)
    ARCH_CFLAGS_IS_SET = true
endif
ifeq ($(ARCH_CFLAGS),none)
    ARCH_CFLAGS_IS_SET = true
endif
ifdef ARCH_CFLAGS_IS_SET
    ifeq ($(processor),$(filter $(processor),aarch64 arm64))
        override ARCH_CFLAGS := -march=armv8-a+fp+simd
    else ifeq ($(processor),$(filter $(processor),i386 x86_64))
        override ARCH_CFLAGS := -maes -mpclmul -mssse3 -msse4.2
    else ifeq ($(processor),$(filter $(processor),arm armv7 armv7l))
        override ARCH_CFLAGS := -mfpu=neon
    else
        $(error Unsupported architecture)
    endif
endif

FEATURE ?=
ifneq ($(FEATURE),)
ifneq ($(FEATURE),none)
COMMA:= ,
ARCH_CFLAGS := $(ARCH_CFLAGS)+$(subst $(COMMA),+,$(FEATURE))
endif
endif

CXXFLAGS += -Wall -Wcast-qual -Wold-style-cast -Wconversion -I. $(ARCH_CFLAGS) -std=gnu++14
LDFLAGS	+= -lm
OBJS = \
    tests/binding.o \
    tests/common.o \
    tests/impl.o \
    tests/main.o
deps := $(OBJS:%.o=%.o.d)

# Floating-point edge case test objects
FP_EDGE_OBJS = \
    tests/fp_edge_cases.o \
    tests/common.o \
    tests/binding.o
fp_edge_deps := $(FP_EDGE_OBJS:%.o=%.o.d)

.SUFFIXES: .o .cpp
.cpp.o:
	$(CXX) -o $@ $(CXXFLAGS) -c -MMD -MF $@.d $<

EXEC = tests/main
FP_EDGE_EXEC = tests/fp_edge_cases

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

$(FP_EDGE_EXEC): $(FP_EDGE_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

fp_edge_cases: $(FP_EDGE_EXEC)
ifeq ($(processor),$(filter $(processor),aarch64 arm64 arm armv7l))
	$(CC) $(ARCH_CFLAGS) -c sse2neon.h
endif
	$(EXEC_WRAPPER) $^

check: tests/main
ifeq ($(processor),$(filter $(processor),aarch64 arm64 arm armv7l))
	$(CC) $(ARCH_CFLAGS) -c sse2neon.h
endif
	$(EXEC_WRAPPER) $^

CLANG_FORMAT ?= $(shell command -v clang-format-20 2>/dev/null || \
    (command -v clang-format >/dev/null 2>&1 && \
     [ "$$(clang-format --version | sed 's/.*version \([0-9]*\).*/\1/')" -ge 20 ] 2>/dev/null && \
     echo clang-format))

indent:
	@echo "Formatting files with clang-format.."
	@if [ -z "$(CLANG_FORMAT)" ]; then \
	    echo "clang-format version 20+ is required"; \
	    echo "Install clang-format-20 or ensure 'clang-format --version' reports 20+"; \
	    exit 1; \
	fi
	$(CLANG_FORMAT) -i sse2neon.h tests/*.cpp tests/*.h

.PHONY: clean check format fp_edge_cases
clean:
	$(RM) $(OBJS) $(EXEC) $(deps) sse2neon.h.gch
	$(RM) $(FP_EDGE_OBJS) $(FP_EDGE_EXEC) $(fp_edge_deps)

-include $(deps)
-include $(fp_edge_deps)
