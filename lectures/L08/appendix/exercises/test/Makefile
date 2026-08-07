# Testsuite target.
TARGET := testsuite

# Root directory of the implementation under test. Unlike L01 - L05, the implementation doesn't
# live in this 'exercises' directory: from L08 onwards you work in the 'cnn_work' codebase, so
# that's what this points at. Override with 'make ML_DIR=<path>' to test a tree elsewhere.
ML_DIR := ../../../cnn_work

# Directory holding the QAcademy Test framework, checked out as a submodule at the root of this
# repository. Run 'git submodule update --init' once if it's empty.
QACADEMY_TEST_DIR := ../../../../../libs/test
QACADEMY_TEST_LIB := $(QACADEMY_TEST_DIR)/libqacademy_test.a

# Source files under test - every source file of the implementation except main.cpp, which is
# excluded because the test suite provides its own main() in testsuite.cpp.
#
# This is the whole list rather than just conv.cpp, because conv.cpp builds its activation function
# via factory::Factory, and factory.cpp in turn constructs every other layer. The files for the
# layers you haven't written yet are still TODO placeholders and compile to nothing, so the same
# list keeps working as you fill them in over L09 and L10.
SOURCE_FILES := $(ML_DIR)/source/ml/cnn/cnn.cpp \
                $(ML_DIR)/source/ml/conv_layer/conv.cpp \
                $(ML_DIR)/source/ml/conv_layer/max_pool.cpp \
                $(ML_DIR)/source/ml/dense_layer/dense.cpp \
                $(ML_DIR)/source/ml/factory/factory.cpp \
                $(ML_DIR)/source/ml/flatten_layer/flatten.cpp \
                $(ML_DIR)/source/ml/random/generator.cpp \
                $(ML_DIR)/source/ml/utils.cpp \

# Test files - update this list as new test files are added to this directory.
TEST_FILES := ml/conv_layer/conv_test.cpp \
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
