CC = gcc
CFLAGS = -g
LDFLAGS = -lncurses -lzmq

SERVER_SRC = server-exercise-3.c
MACHINE_CLIENT_SRC = machine-client.c
HUMAN_CLIENT_SRC = human-client.c
REMOTE_DISPLAY_CLIENT_SRC = remote-display-client.c
COMMON_SRC = remote-char.h

all: server machine-client human-client remote-display-client

server: $(SERVER_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) -o server $(SERVER_SRC) $(LDFLAGS)

machine-client: $(MACHINE_CLIENT_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) -o machine-client $(MACHINE_CLIENT_SRC) $(LDFLAGS)

human-client: $(HUMAN_CLIENT_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) -o human-client $(HUMAN_CLIENT_SRC) $(LDFLAGS)

remote-display-client: $(REMOTE_DISPLAY_CLIENT_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) -o remote-display-client $(REMOTE_DISPLAY_CLIENT_SRC) $(LDFLAGS)

clean:
	rm -f server machine-client human-client remote-display-client