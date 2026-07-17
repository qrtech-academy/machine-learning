# Target application.
TARGET := app

# C++ compiler.
CXX_COMPILER := g++

# Source files.
SRC_FILES := source/main.cpp \
             source/ml/cnn/cnn.cpp \
             source/ml/conv_layer/conv.cpp \
             source/ml/conv_layer/max_pool.cpp \
             source/ml/dense_layer/dense.cpp \
             source/ml/factory/factory.cpp \
             source/ml/flatten_layer/flatten.cpp \
             source/ml/random/generator.cpp \
             source/ml/utils.cpp \

# Include directory.
INC_DIR := -I include

# Compiler flags.
# Whether each layer is a stub or the real implementation is chosen per-layer inside
# source/ml/factory/factory.cpp, not by a compiler flag.
COMPILER_FLAGS := -Wall -Werror -std=c++17

# Build and run the target as default.
default: build run

# Build the target.
build:
	@$(CXX_COMPILER) $(SRC_FILES) -o $(TARGET) $(COMPILER_FLAGS) $(INC_DIR)

# Run the target.
run:
	@./$(TARGET)

# Clean the target.
clean:
	@rm -f $(TARGET)
