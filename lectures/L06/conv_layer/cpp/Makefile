# Application target.
TARGET := conv_demo

# C++ compiler.
CXX_COMPILER := g++

# Source files.
SRC_FILES := conv_demo.cpp \

# Compiler flags. The language standard always applies; the line below adds warnings and
# compiles the demo body once your implementation is in place.
CXX_FLAGS := -std=c++17
# TODO: Uncomment this line once the implementation is finished.
#CXX_FLAGS := -std=c++17 -Wall -Werror -DCONV_LAYER_IMPLEMENTED

# Build and run the application as default.
default: build run

# Build the application.
build:
	@$(CXX_COMPILER) $(SRC_FILES) -o $(TARGET) $(CXX_FLAGS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
clean:
	@rm -f $(TARGET)

.PHONY: default build run clean
