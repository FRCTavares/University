CC = gcc
CFLAGS = -fPIC -g

# Executables
SERVER = server
CLIENT = client
LIBRARY = libfunctions.so

all: $(SERVER) $(CLIENT) $(LIBRARY)

$(SERVER): server.c
	$(CC) $(CFLAGS) -o $(SERVER) server.c -ldl

$(CLIENT): client.c
	$(CC) $(CFLAGS) -o $(CLIENT) client.c

$(LIBRARY): libfunctions.c
	$(CC) $(CFLAGS) -shared -o $(LIBRARY) libfunctions.c

clean:
	rm -f $(SERVER) $(CLIENT) $(LIBRARY) /tmp/fifo_execution
