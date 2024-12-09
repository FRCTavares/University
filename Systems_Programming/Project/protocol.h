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

typedef enum
{
    MSG_TYPE_CONNECT = 0,
    MSG_TYPE_MOVE = 1,
    MSG_TYPE_ZAP = 2,
    MSG_TYPE_DISCONNECT = 3,
    MSG_TYPE_GAME_END = 4
} msg_type_t;

typedef enum direction_t
{
    UP,
    DOWN,
    LEFT,
    RIGHT
} direction_t;

typedef struct ch_info_t
{
    int ch;
    int pos_x, pos_y;
    int dir; // Whether character moves vertically(1) or horizontally(0)
    int score;
    time_t last_fire_time;
    int stunned;
} ch_info_t;

typedef struct
{
    msg_type_t msg_type;
    int ch;
    direction_t direction;
} remote_char_t;

#endif // PROTOCOL_H