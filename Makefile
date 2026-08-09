# Build every lecture demo that has a Makefile.
build:
	@bash ci/build.sh

# Build and run every lecture test suite that has a solution to run against.
test:
	@bash ci/test.sh

# Format all C/C++ files (clang-format) and Python files (black) in place.
format:
	@bash ci/format.sh

# Check formatting without modifying any files; fails if something isn't formatted.
format-check:
	@bash ci/format.sh --check

.PHONY: build test format format-check
