CC = gcc
CFLAGS = -g -I.
LDFLAGS = -lncurses -lzmq

# Source files
SERVER_SRC = server/game-server.c
ASTRONAUT_CLIENT_SRC = astronaut-client/astronaut-client.c
OUTER_SPACE_DISPLAY_SRC = outer-space-display/outer-space-display.c
COMMON_HDR = protocol.h

# Targets
all: server/server astronaut-client/astronaut-client outer-space-display/outer-space-display

server/server: $(SERVER_SRC) $(COMMON_HDR)
	mkdir -p server
	$(CC) $(CFLAGS) -o $@ $(SERVER_SRC) $(LDFLAGS)

astronaut-client/astronaut-client: $(ASTRONAUT_CLIENT_SRC) $(COMMON_HDR)
	mkdir -p astronaut-client
	$(CC) $(CFLAGS) -o $@ $(ASTRONAUT_CLIENT_SRC) $(LDFLAGS)

outer-space-display/outer-space-display: $(OUTER_SPACE_DISPLAY_SRC) $(COMMON_HDR)
	mkdir -p outer-space-display
	$(CC) $(CFLAGS) -o $@ $(OUTER_SPACE_DISPLAY_SRC) $(LDFLAGS)

clean:
	rm -f server/server astronaut-client/astronaut-client outer-space-display/outer-space-display
