#ifndef REMOTE_CHAR_H
#define REMOTE_CHAR_H

typedef enum { UP, DOWN, LEFT, RIGHT } direction_t;

typedef struct {
    int msg_type;       // 1 for connection, 2 for movement
    char character;     // Character to display
    direction_t direction; // Movement direction
} message_t;

#define FIFO_PATH "/tmp/remote_char_fifo"

#endif // REMOTE_CHAR_H
