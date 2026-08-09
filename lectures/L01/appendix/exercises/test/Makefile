# Testsuite target.
TARGET := testsuite

# Root directory of the implementation under test, i.e. the exercises directory holding this
# 'test' directory. Override with 'make ML_DIR=<path>' to test a tree elsewhere.
ML_DIR := ..

# Directory holding the QAcademy Test framework, checked out as a submodule at the root of this
# repository. Run 'git submodule update --init' once if it's empty.
QACADEMY_TEST_DIR := ../../../../../libs/test
QACADEMY_TEST_LIB := $(QACADEMY_TEST_DIR)/libqacademy_test.a

# Source files under test - update this list as new source files are added to the implementation.
# Note that main.cpp is excluded: the test suite provides its own main() in testsuite.cpp.
SOURCE_FILES := $(ML_DIR)/source/ml/lin_reg/fixed.cpp \

# Test files - update this list as new test files are added to this directory.
TEST_FILES := ml/lin_reg/fixed_test.cpp \
              testsuite.cpp \

# All files.
ALL_FILES := $(SOURCE_FILES) $(TEST_FILES)

# Include directory of the implementation under test.
INC_DIR := $(ML_DIR)/include

# C++ compiler.
CXX_COMPILER := g++

# C++ compiler flags.
CXX_FLAGS := -std=c++17 -Wall -Werror -I$(INC_DIR) -I$(QACADEMY_TEST_DIR)/include

# Build and run the test suite as default.
default: build run

# Build the QAcademy Test framework.
$(QACADEMY_TEST_LIB):
	@$(MAKE) -C $(QACADEMY_TEST_DIR) lib

# Build the test suite.
build: $(QACADEMY_TEST_LIB)
	@$(CXX_COMPILER) $(ALL_FILES) -o $(TARGET) $(CXX_FLAGS) -L$(QACADEMY_TEST_DIR) -lqacademy_test

# Run the test suite.
run:
	@./$(TARGET)

# Clean the test suite.
clean:
	@rm -f $(TARGET)

.PHONY: default build run clean
