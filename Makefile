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
override ARCH_CFLAGS := $(ARCH_CFLAGS)+$(subst $(COMMA),+,$(FEATURE))
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

# Check if a specific compiler flag is supported by attempting a dummy compilation.
# Uses -Werror to catch unsupported warning flags (Clang warns but doesn't fail without it).
# Returns the flag if supported, empty string otherwise.
# Usage: $(call check_cxx_flag,-some-flag)
check_cxx_flag = $(shell $(CXX) -Werror $(1) -S -o /dev/null -x c++ /dev/null 2>/dev/null && echo "$(1)")

# Extra warning flags: EXTRA_WARNINGS=uninit for uninitialized variable checks
# GCC uses -Wmaybe-uninitialized (stricter), Clang uses -Wuninitialized
EXTRA_WARNINGS ?=
ifneq ($(EXTRA_WARNINGS),)
ifeq ($(EXTRA_WARNINGS),uninit)
UNINIT_FLAGS_TO_CHECK := -Wuninitialized -Wmaybe-uninitialized
EXTRA_WARNINGS_FLAGS := -Werror $(foreach flag,$(UNINIT_FLAGS_TO_CHECK),$(call check_cxx_flag,$(flag)))
endif
else
EXTRA_WARNINGS_FLAGS =
endif

CXXFLAGS += -Wall -Wcast-qual -Wold-style-cast -Wconversion -I. $(ARCH_CFLAGS) -std=gnu++14 $(SANITIZE_FLAGS) $(STRICT_ALIASING_FLAGS) $(EXTRA_WARNINGS_FLAGS)
LDFLAGS  += -lm $(SANITIZE_FLAGS)
OBJS = \
    tests/binding.o \
    tests/common.o \
    tests/impl.o \
    tests/main.o
deps := $(OBJS:%.o=%.o.d)

# IEEE-754 edge case test objects
IEEE754_OBJS = \
    tests/ieee754.o \
    tests/common.o \
    tests/binding.o
ieee754_deps := $(IEEE754_OBJS:%.o=%.o.d)

.SUFFIXES: .o .cpp
.cpp.o:
	$(CXX) -o $@ $(CXXFLAGS) -c -MMD -MF $@.d $<

EXEC = tests/main
IEEE754_EXEC = tests/ieee754

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

$(IEEE754_EXEC): $(IEEE754_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

ieee754: $(IEEE754_EXEC)
ifeq ($(processor),$(filter $(processor),aarch64 arm64 arm armv7l))
	$(CC) $(ARCH_CFLAGS) -c sse2neon.h
endif
	$(EXEC_WRAPPER) $^

check: tests/main $(IEEE754_EXEC)
ifeq ($(processor),$(filter $(processor),aarch64 arm64 arm armv7l))
	$(CC) $(ARCH_CFLAGS) -c sse2neon.h
endif
	$(EXEC_WRAPPER) tests/main
	$(EXEC_WRAPPER) $(IEEE754_EXEC)

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

# Convenience target for running only main tests (skip IEEE-754)
check-main: $(EXEC)
ifeq ($(processor),$(filter $(processor),aarch64 arm64 arm armv7l))
	$(CC) $(ARCH_CFLAGS) -c sse2neon.h
endif
	$(EXEC_WRAPPER) $(EXEC)

# Convenience target for running only IEEE-754 edge case tests (skip main)
check-ieee754: $(IEEE754_EXEC)
ifeq ($(processor),$(filter $(processor),aarch64 arm64 arm armv7l))
	$(CC) $(ARCH_CFLAGS) -c sse2neon.h
endif
	$(EXEC_WRAPPER) $(IEEE754_EXEC)

# Convenience target for running tests with UBSan
check-ubsan: clean
	$(MAKE) SANITIZE=undefined check

# Convenience target for running tests with ASan
check-asan: clean
	$(MAKE) SANITIZE=address check

# Convenience target for running tests with strict aliasing checks
check-strict-aliasing: clean
	$(MAKE) STRICT_ALIASING=1 check

# Convenience target for running tests with uninitialized variable checks
# Uses -Werror -Wuninitialized (and -Wmaybe-uninitialized on GCC)
check-uninit: clean
	$(MAKE) EXTRA_WARNINGS=uninit check

# Check preprocessor macro hygiene (typos, consistency)
# Inconsistent raw __aarch64__ checks are treated as errors because they
# cause _M_ARM64/__arm64__ builds to silently use slower fallback paths.
check-macros:
	@python3 .ci/check-macros.py sse2neon.h

# Differential testing objects
DIFFERENTIAL_OBJS = \
	tests/differential.o \
	tests/common.o \
	tests/binding.o
differential_deps := $(DIFFERENTIAL_OBJS:%.o=%.o.d)

DIFFERENTIAL_EXEC = tests/differential
GOLDEN_DIR ?= golden

$(DIFFERENTIAL_EXEC): $(DIFFERENTIAL_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Generate golden reference data on x86
# This target must be run on an x86 host to create reference outputs
generate-golden: $(DIFFERENTIAL_EXEC)
ifeq ($(processor),$(filter $(processor),aarch64 arm64 arm armv7l))
	@echo "ERROR: generate-golden must be run on x86 host"
	@echo "Use: python3 scripts/gen-golden.py on an x86 machine"
	@exit 1
endif
	@mkdir -p $(GOLDEN_DIR)
	$(DIFFERENTIAL_EXEC) --generate $(GOLDEN_DIR)
	@echo ""
	@echo "Golden data generated in $(GOLDEN_DIR)/"
	@echo "Copy this directory to ARM target for verification."

# Verify sse2neon against golden reference data on ARM
# This target compares NEON implementations against x86 reference outputs
#
# Scalar Fallback Coverage:
#   The differential harness now includes intrinsics with ARMv7 scalar fallbacks:
#   - Horizontal ops (hadd/hsub): uses shift-right-accumulate on ARMv7
#   - Dot product (dp_ps/pd): uses scalar accumulation on ARMv7
#   - String comparison (cmpistri/cmpistrm/etc): sequential loops on ARMv7
#   - CRC32: software nibble-table fallback when no __ARM_FEATURE_CRC32
#   When run on ARMv7 (via QEMU or native), these test the scalar fallback paths.
#   When run on AArch64, they test the optimized NEON paths.
check-differential: $(DIFFERENTIAL_EXEC)
	@if [ ! -d "$(GOLDEN_DIR)" ]; then \
		echo "ERROR: Golden data directory '$(GOLDEN_DIR)' not found"; \
		echo "Generate it on x86 first with: make generate-golden"; \
		echo "Or specify a custom path: make check-differential GOLDEN_DIR=/path/to/golden"; \
		exit 1; \
	fi
ifeq ($(processor),$(filter $(processor),aarch64 arm64 arm armv7l))
	$(CC) $(ARCH_CFLAGS) -c sse2neon.h
endif
	$(EXEC_WRAPPER) $(DIFFERENTIAL_EXEC) --verify $(GOLDEN_DIR)

# Coverage verification: compare implemented intrinsics against Intel reference
coverage-report:
	@python3 scripts/coverage-check.py

.PHONY: clean check check-main check-ieee754 check-ubsan check-asan check-strict-aliasing check-uninit check-macros check-differential generate-golden coverage-report indent ieee754
clean:
	$(RM) $(OBJS) $(EXEC) $(deps) sse2neon.h.gch
	$(RM) $(IEEE754_OBJS) $(IEEE754_EXEC) $(ieee754_deps)
	$(RM) $(DIFFERENTIAL_OBJS) $(DIFFERENTIAL_EXEC) $(differential_deps)

-include $(deps)
-include $(ieee754_deps)
-include $(differential_deps)
