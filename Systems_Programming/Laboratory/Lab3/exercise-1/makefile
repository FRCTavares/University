# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -g

# ncurses library
LDFLAGS = -lncurses

# Source files
SERVER_SRC = server.c
HUMAN_CLIENT_SRC = human-control-client.c
MACHINE_CLIENT_SRC = machine-control-client.c

# Header files
HEADERS = remote-char.h

# Output executables
SERVER_EXE = server
HUMAN_CLIENT_EXE = human_client
MACHINE_CLIENT_EXE = machine_client

# Default target
all: $(SERVER_EXE) $(HUMAN_CLIENT_EXE) $(MACHINE_CLIENT_EXE)

# Server target
$(SERVER_EXE): $(SERVER_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(SERVER_SRC) -o $(SERVER_EXE) $(LDFLAGS)

# Human client target
$(HUMAN_CLIENT_EXE): $(HUMAN_CLIENT_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(HUMAN_CLIENT_SRC) -o $(HUMAN_CLIENT_EXE) $(LDFLAGS)

# Machine client target
$(MACHINE_CLIENT_EXE): $(MACHINE_CLIENT_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(MACHINE_CLIENT_SRC) -o $(MACHINE_CLIENT_EXE)

# Clean target
clean:
	rm -f $(SERVER_EXE) $(HUMAN_CLIENT_EXE) $(MACHINE_CLIENT_EXE)
	rm -f *.o

# Run server
run_server: $(SERVER_EXE)
	./$(SERVER_EXE)

# Run human client
run_human_client: $(HUMAN_CLIENT_EXE)
	./$(HUMAN_CLIENT_EXE)

# Run machine client
run_machine_client: $(MACHINE_CLIENT_EXE)
	./$(MACHINE_CLIENT_EXE)
