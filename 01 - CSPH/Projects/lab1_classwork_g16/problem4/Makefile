CC = g++
CFLAGS = -std=c++20

# Define the name of the output executable
EXECUTABLE = program4

all: problem4.cpp fake_intrinsics.h
	$(CC) $(CFLAGS) problem4.cpp -o $(EXECUTABLE) fake_intrinsics.h

run:
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

.PHONY: all run clean
