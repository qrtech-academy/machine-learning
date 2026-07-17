# Application target.
TARGET := flatten_demo

# C compiler.
C_COMPILER := gcc

# Source files.
SRC_FILES := source/flatten_demo.c \
				source/ml/flatten_layer.c \
                source/ml/matrix.c \

# Main include directory.
INC_DIR := -I include/

# Compiler flags.
C_FLAGS := -std=c11 -Wall -Werror

# Linked libraries.
LINK_LIBS := -lm

# Build and run the application as default.
default: build run

# Build the application.
build:
	@$(C_COMPILER) $(SRC_FILES) -o $(TARGET) $(C_FLAGS) $(INC_DIR) $(LINK_LIBS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
clean:
	@rm -f $(TARGET)
