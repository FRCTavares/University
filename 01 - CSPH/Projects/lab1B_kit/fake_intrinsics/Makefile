CC = g++
CFLAGS = -std=c++20

# Define the name of the output executable
EXECUTABLE = my_program

all: main.cpp fake_intrinsics.h stats.h
	$(CC) $(CFLAGS) main.cpp -o $(EXECUTABLE)  

run:
	./$(EXECUTABLE)

run-vl4:
	./$(EXECUTABLE) --vl 4

run-vl8:
	./$(EXECUTABLE) --vl 8

run-vl16:
	./$(EXECUTABLE) --vl 16

help:
	@echo "Usage:"
	@echo "  make all      - Compile the program"
	@echo "  make run      - Run with default vector length (8)"
	@echo "  make run-vl4  - Run with vector length 4"
	@echo "  make run-vl8  - Run with vector length 8"
	@echo "  make run-vl16 - Run with vector length 16"
	@echo "  make clean    - Remove executable"
	@echo ""
	@echo "Direct usage:"
	@echo "  ./$(EXECUTABLE) [--vl <4|8|16>]"

clean:
	rm -f $(EXECUTABLE)

.PHONY: all run run-vl4 run-vl8 run-vl16 help clean
