CC = gcc
CFLAGS = -g
LDFLAGS = -lncurses -lzmq

SERVER_SRC = game-server.c
ASTRONAUT_CLIENT_SRC = astronaut-client.c
OUTER_SPACE_DISPLAY_SRC = outer-space-display.c
COMMON_SRC = protocol.h

all: server astronaut-client outer-space-display

server: $(SERVER_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) -o server $(SERVER_SRC) $(LDFLAGS)

astronaut-client: $(ASTRONAUT_CLIENT_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) -o astronaut-client $(ASTRONAUT_CLIENT_SRC) $(LDFLAGS)

outer-space-display: $(OUTER_SPACE_DISPLAY_SRC) $(COMMON_SRC)
	$(CC) $(CFLAGS) -o outer-space-display $(OUTER_SPACE_DISPLAY_SRC) $(LDFLAGS)

clean:
	rm -f server astronaut-client outer-space-display