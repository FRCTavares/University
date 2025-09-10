CC = g++
CFLAGS = -std=c++20

# Define the name of the output executable
EXECUTABLE = my_program

all: main.cpp fake_intrinsics.h stats.h
	$(CC) $(CFLAGS) main.cpp -o $(EXECUTABLE)

run:
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)

.PHONY: all run clean
