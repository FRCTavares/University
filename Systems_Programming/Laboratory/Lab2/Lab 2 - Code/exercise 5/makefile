CC = gcc
CFLAGS = -fPIC -g

# Executables
SERVER = server
CLIENT = client
LIBRARY = funcs-ex5.so

all: $(SERVER) $(CLIENT) $(LIBRARY)

# Build the server executable
$(SERVER): server.c
	$(CC) $(CFLAGS) -o $(SERVER) server.c -ldl

# Build the client executable
$(CLIENT): client.c
	$(CC) $(CFLAGS) -o $(CLIENT) client.c

# Build the dynamic library
$(LIBRARY): funcs-ex5.c
	$(CC) $(CFLAGS) -shared -o $(LIBRARY) funcs-ex5.c

clean:
	rm -f $(SERVER) $(CLIENT) $(LIBRARY) /tmp/fifo_execution
