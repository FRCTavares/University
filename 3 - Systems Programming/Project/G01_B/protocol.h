// protocol.h
#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <ncurses.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <zmq.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <pthread.h>

// Constants
#define WINDOW_SIZE 22
#define MAX_PLAYERS 8
#define MAX_ALIENS 256
#define ALIEN_COUNT 16 * 16 / 3

// Server Addresses
#define SERVER_IP "tcp://0.0.0.0"
#define SERVER_REQUEST_PORT "5555"
#define SERVER_PUBLISH_PORT "5556"
#define SERVER_REQUEST_ADDRESS SERVER_IP ":" SERVER_REQUEST_PORT
#define SERVER_PUBLISH_ADDRESS SERVER_IP ":" SERVER_PUBLISH_PORT

// Client Addresses
#define CLIENT_SUBSCRIBE_IP "tcp://localhost"
#define CLIENT_SUBSCRIBE_PORT "5557"
#define CLIENT_REQUEST_IP "tcp://localhost"
#define CLIENT_REQUEST_PORT "5558"
#define CLIENT_SUBSCRIBE_ADDRESS CLIENT_SUBSCRIBE_IP ":" CLIENT_SUBSCRIBE_PORT
#define CLIENT_REQUEST_ADDRESS CLIENT_REQUEST_IP ":" CLIENT_REQUEST_PORT

// Message Types
typedef enum
{
    MSG_TYPE_CONNECT = 0,
    MSG_TYPE_MOVE = 1,
    MSG_TYPE_ZAP = 2,
    MSG_TYPE_DISCONNECT = 3,
    MSG_TYPE_GAME_END = 4,
    MSG_TYPE_ALIEN_DIRECTION = 5
} msg_type_t;

// Directions
typedef enum direction_t
{
    UP,
    DOWN,
    LEFT,
    RIGHT
} direction_t;

// Astronaut Information
typedef struct ch_info_t
{
    int ch;
    int pos_x, pos_y;
    int dir; // Whether character moves vertically(1) or horizontally(0)
    int score;
    time_t last_fire_time;
    time_t stunned;
    int GAME_TOKEN;
} ch_info_t;

// Message Structure
typedef struct
{
    msg_type_t msg_type;
    int ch;
    direction_t direction;
    int GAME_TOKEN;
} remote_char_t;

#endif // PROTOCOL_H