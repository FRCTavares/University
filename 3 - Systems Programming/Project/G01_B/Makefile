CC = gcc
C++ = g++
CFLAGS = -g -I.
LDFLAGS = -lncurses -lzmq -lpthread -lprotobuf-c -lprotobuf

# Source files
SERVER_SRC = server/game-server.c
ASTRONAUT_CLIENT_SRC = astronaut-client/astronaut-client.c
OUTER_SPACE_DISPLAY_SRC = outer-space-display/outer-space-display.c
ASTRONAUT_CLIENT_DISPLAY_SRC = astronaut-display-client/astronaut-display-client.c
SPACE_HIGH_SCORES_SRC = space-high-scores/space-high-scores.C
COMMON_HDR = protocol.h
PROT_BUFF_HDR_CPP = message.pb.h
MESSAGE_SRC_CPP = message.pb.cc
PROT_BUFF_HDR_C = message.pb-c.h
MESSAGE_SRC_C = message.pb-c.c


# Targets
all: server/server astronaut-client/astronaut-client outer-space-display/outer-space-display astronaut-display-client/astronaut-display-client space-high-scores/space-high-scores

server/server: $(SERVER_SRC) $(COMMON_HDR) $(PROT_BUFF_HDR_C) $(MESSAGE_SRC_C)
	mkdir -p server
	$(CC) $(CFLAGS) -o $@ $(SERVER_SRC) $(MESSAGE_SRC_C) $(LDFLAGS)

astronaut-client/astronaut-client: $(ASTRONAUT_CLIENT_SRC) $(COMMON_HDR)
	mkdir -p astronaut-client
	$(CC) $(CFLAGS) -o $@ $(ASTRONAUT_CLIENT_SRC) $(LDFLAGS)

outer-space-display/outer-space-display: $(OUTER_SPACE_DISPLAY_SRC) $(COMMON_HDR)
	mkdir -p outer-space-display
	$(CC) $(CFLAGS) -o $@ $(OUTER_SPACE_DISPLAY_SRC) $(LDFLAGS)

astronaut-display-client/astronaut-display-client: $(ASTRONAUT_CLIENT_DISPLAY_SRC) $(COMMON_HDR)
	mkdir -p astronaut-display-client
	$(CC) $(CFLAGS) -o $@ $(ASTRONAUT_CLIENT_DISPLAY_SRC) $(LDFLAGS)

space-high-scores/space-high-scores: $(SPACE_HIGH_SCORES_SRC) $(COMMON_HDR) $(PROT_BUFF_HDR_CPP) $(MESSAGE_SRC_CPP)
	mkdir -p space-high-scores
	$(C++) $(CFLAGS) -o $@ $(SPACE_HIGH_SCORES_SRC) $(MESSAGE_SRC_CPP) $(LDFLAGS)


clean:
	rm -f server/server astronaut-client/astronaut-client outer-space-display/outer-space-display astronaut-display-client/astronaut-display-client space-high-scores/space-high-scores
