# Compiler settings
CC = gcc
CFLAGS = -fPIC -g

# Executable and libraries
TARGET = main
LIB1 = lib1.so
LIB2 = lib2.so

# Default target: compile everything
all: $(TARGET)

# Compile main.c and link with dynamic loading library
$(TARGET): main.c $(LIB1) $(LIB2)
	$(CC) $(CFLAGS) -o $(TARGET) main.c -ldl
	rm -f $(LIB1) $(LIB2)  # Remove shared libraries after linking

# Build lib1.so as an intermediate step
$(LIB1): lib1.c
	$(CC) $(CFLAGS) -shared -o $(LIB1) lib1.c

# Build lib2.so as an intermediate step
$(LIB2): lib2.c
	$(CC) $(CFLAGS) -shared -o $(LIB2) lib2.c

# Clean up compiled files
clean:
	rm -f $(TARGET)
