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

# Sanitizer support: SANITIZE=undefined for UBSan, SANITIZE=address for ASan
SANITIZE ?=
ifneq ($(SANITIZE),)
SANITIZE_FLAGS = -fsanitize=$(SANITIZE) -fno-omit-frame-pointer
# Add -fwrapv for well-defined signed overflow behavior (wraparound)
# This matches common expectations for integer overflow in SIMD code
SANITIZE_FLAGS += -fwrapv
else
SANITIZE_FLAGS =
endif

# Strict aliasing checking: STRICT_ALIASING=1 to enable
STRICT_ALIASING ?=
ifneq ($(STRICT_ALIASING),)
STRICT_ALIASING_FLAGS = -fstrict-aliasing -Wstrict-aliasing=2
else
STRICT_ALIASING_FLAGS =
endif

CXXFLAGS += -Wall -Wcast-qual -Wold-style-cast -Wconversion -I. $(ARCH_CFLAGS) -std=gnu++14 $(SANITIZE_FLAGS) $(STRICT_ALIASING_FLAGS)
LDFLAGS  += -lm $(SANITIZE_FLAGS)
OBJS = \
    tests/binding.o \
    tests/common.o \
    tests/impl.o \
    tests/main.o
deps := $(OBJS:%.o=%.o.d)

# Floating-point edge case test objects
FLOATPOINT_OBJS = \
    tests/floatpoint.o \
    tests/common.o \
    tests/binding.o
floatpoint_deps := $(FLOATPOINT_OBJS:%.o=%.o.d)

.SUFFIXES: .o .cpp
.cpp.o:
	$(CXX) -o $@ $(CXXFLAGS) -c -MMD -MF $@.d $<

EXEC = tests/main
FLOATPOINT_EXEC = tests/floatpoint

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

$(FLOATPOINT_EXEC): $(FLOATPOINT_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

floatpoint: $(FLOATPOINT_EXEC)
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

# Convenience target for running tests with UBSan
check-ubsan: clean
	$(MAKE) SANITIZE=undefined check

# Convenience target for running tests with strict aliasing checks
check-strict-aliasing: clean
	$(MAKE) STRICT_ALIASING=1 check

.PHONY: clean check check-ubsan check-strict-aliasing format floatpoint
clean:
	$(RM) $(OBJS) $(EXEC) $(deps) sse2neon.h.gch
	$(RM) $(FLOATPOINT_OBJS) $(FLOATPOINT_EXEC) $(floatpoint_deps)

-include $(deps)
-include $(floatpoint_deps)
