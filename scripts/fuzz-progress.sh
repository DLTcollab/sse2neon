#!/usr/bin/env bash

# Progress bar wrapper for libFuzzer
# Usage: ./scripts/fuzz-progress.sh <fuzzer_binary> <max_time_seconds> [fuzzer_args...]
#
# Features:
# - Visual progress bar using Unicode block characters
# - Real-time metrics (coverage, corpus size, exec/s)
# - Automatic quiet mode in CI/CD environments
# - Elapsed/remaining time display
#
# CI/CD Detection:
# - Set CI=true, GITHUB_ACTIONS=true, or JENKINS_URL to enable quiet mode
# - Or run with FUZZ_QUIET=1 to force quiet mode

set -e

# Check arguments
if [ $# -lt 2 ]; then
	echo "Usage: $0 <fuzzer_binary> <max_time_seconds> [fuzzer_args...]"
	echo "Example: $0 tests/fuzz 60 -max_len=1024"
	exit 1
fi

FUZZER="$1"
MAX_TIME="$2"
shift 2

# Validate MAX_TIME is a positive integer
if ! [[ "$MAX_TIME" =~ ^[0-9]+$ ]] || [ "$MAX_TIME" -eq 0 ]; then
	echo "Error: max_time_seconds must be a positive integer (got: '$MAX_TIME')"
	exit 1
fi

# Remaining arguments are passed to fuzzer (use array to handle spaces)
FUZZER_ARGS=("$@")

# Detect CI/CD environment
is_ci() {
	[ -n "$CI" ] || [ -n "$GITHUB_ACTIONS" ] || [ -n "$JENKINS_URL" ] ||
		[ -n "$GITLAB_CI" ] || [ -n "$CIRCLECI" ] || [ -n "$TRAVIS" ] ||
		[ -n "$FUZZ_QUIET" ]
}

# ANSI color codes
if [ -t 1 ] && ! is_ci; then
	COLOR_GREEN="\033[32m"
	COLOR_YELLOW="\033[33m"
	COLOR_CYAN="\033[36m"
	COLOR_BOLD="\033[1m"
	COLOR_RESET="\033[0m"
	CLEAR_LINE="\033[2K\r"
else
	COLOR_GREEN=""
	COLOR_YELLOW=""
	COLOR_CYAN=""
	COLOR_BOLD=""
	COLOR_RESET=""
	CLEAR_LINE=""
fi

# Progress bar configuration
BAR_WIDTH=30

# Draw progress bar
draw_progress_bar() {
	local percentage=$1
	local filled=$((percentage * BAR_WIDTH / 100))
	local empty=$((BAR_WIDTH - filled))

	printf "${COLOR_CYAN}"
	for ((i = 0; i < filled; i++)); do printf "█"; done
	for ((i = 0; i < empty; i++)); do printf "░"; done
	printf "${COLOR_RESET}"
}

# Format time as MM:SS
format_time() {
	local seconds=$1
	printf "%02d:%02d" $((seconds / 60)) $((seconds % 60))
}

# Parse libFuzzer output and extract metrics
parse_metrics() {
	local line="$1"

	# Extract coverage (cov: or NEW: lines contain coverage info)
	if [[ "$line" =~ cov:\ *([0-9]+) ]]; then
		echo "cov=${BASH_REMATCH[1]}"
	fi

	# Extract corpus size
	if [[ "$line" =~ corp:\ *([0-9]+) ]]; then
		echo "corp=${BASH_REMATCH[1]}"
	fi

	# Extract exec/s
	if [[ "$line" =~ exec/s:\ *([0-9]+) ]]; then
		echo "execs=${BASH_REMATCH[1]}"
	fi

	# Detect special events
	if [[ "$line" =~ (BINGO|ERROR|CRASH|TIMEOUT|SUMMARY) ]]; then
		echo "event=${BASH_REMATCH[1]}"
	fi
}

# Main progress display function
run_with_progress() {
	local start_time=$(date +%s)
	local coverage=0
	local corpus=0
	local execs=0
	local last_event=""

	# Create temporary file for fuzzer output
	local tmpfile=$(mktemp)
	trap "rm -f $tmpfile" EXIT

	# Start fuzzer in background
	"$FUZZER" -max_total_time="$MAX_TIME" "${FUZZER_ARGS[@]}" 2>&1 | tee "$tmpfile" |
		while IFS= read -r line; do
			# Parse metrics from line
			metrics=$(parse_metrics "$line")

			for metric in $metrics; do
				case "$metric" in
				cov=*) coverage="${metric#cov=}" ;;
				corp=*) corpus="${metric#corp=}" ;;
				execs=*) execs="${metric#execs=}" ;;
				event=*) last_event="${metric#event=}" ;;
				esac
			done

			# Calculate timing
			local now=$(date +%s)
			local elapsed=$((now - start_time))
			local remaining=$((MAX_TIME - elapsed))
			if [ $remaining -lt 0 ]; then remaining=0; fi

			local percentage=$((elapsed * 100 / MAX_TIME))
			if [ $percentage -gt 100 ]; then percentage=100; fi

			# Display progress line
			printf "${CLEAR_LINE}"
			draw_progress_bar $percentage
			printf " ${COLOR_BOLD}%3d%%${COLOR_RESET}" $percentage
			printf " ${COLOR_GREEN}%s${COLOR_RESET}/%s" "$(format_time $elapsed)" "$(format_time $MAX_TIME)"
			printf " ${COLOR_YELLOW}cov:%d${COLOR_RESET}" $coverage
			printf " corp:%d" $corpus
			printf " exec/s:%d" $execs

			# Show special events
			if [ -n "$last_event" ]; then
				printf " ${COLOR_BOLD}[%s]${COLOR_RESET}" "$last_event"
			fi
		done

	# Final newline
	printf "\n"

	# Check for crashes
	if grep -q "SUMMARY.*: [1-9]" "$tmpfile" 2>/dev/null; then
		echo ""
		echo "${COLOR_BOLD}Fuzzing found issues. Check crash files in the current directory.${COLOR_RESET}"
		return 1
	fi

	return 0
}

# Run in quiet mode (for CI/CD)
run_quiet() {
	echo "Running fuzzer for ${MAX_TIME}s (quiet mode)..."
	"$FUZZER" -max_total_time="$MAX_TIME" "${FUZZER_ARGS[@]}"
	local status=$?

	if [ $status -eq 0 ]; then
		echo "Fuzzing completed successfully."
	else
		echo "Fuzzing found issues (exit code: $status)."
	fi

	return $status
}

# Main entry point
main() {
	# Verify fuzzer exists
	if [ ! -x "$FUZZER" ]; then
		echo "Error: Fuzzer binary '$FUZZER' not found or not executable."
		exit 1
	fi

	echo "sse2neon Fuzzer"
	echo "==============="
	echo "Binary: $FUZZER"
	echo "Duration: ${MAX_TIME}s"
	echo "Extra args: ${FUZZER_ARGS[*]:-none}"
	echo ""

	if is_ci; then
		run_quiet
	else
		run_with_progress
	fi
}

main
