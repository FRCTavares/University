#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <stdlib.h>
#include <zmq.h>
#include <string.h>

#define WINDOW_SIZE 22

typedef struct screen_update_t
{
    int pos_x;
    int pos_y;
    char ch;
} screen_update_t;

int main()
{

    void *context = zmq_ctx_new();

    // Initialize subscriber socket
    void *subscriber = zmq_socket(context, ZMQ_SUB);
    zmq_connect(subscriber, "tcp://localhost:5556");

    // Subscribe to all messages
    zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);

    // Initialize ncurses
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();

    WINDOW *my_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, 0);
    box(my_win, 0, 0);
    wrefresh(my_win);

    screen_update_t update;

    while (1)
    {
        zmq_recv(subscriber, &update, sizeof(screen_update_t), 0);
        wmove(my_win, update.pos_x, update.pos_y);
        waddch(my_win, update.ch | A_BOLD);
        wrefresh(my_win);
    }

    // Cleanup
    endwin();
    zmq_close(subscriber);
    zmq_ctx_destroy(context);
    return 0;
}