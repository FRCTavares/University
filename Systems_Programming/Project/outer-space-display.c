#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <stdlib.h>
#include <zmq.h>
#include <string.h>

#define WINDOW_SIZE 22
#define MAX_PLAYERS 8

typedef struct screen_update_t
{
    int pos_x;
    int pos_y;
    char ch;
    char players[MAX_PLAYERS];
    int scores[MAX_PLAYERS];
    int player_count;
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

    WINDOW *score_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, WINDOW_SIZE + 1);
    box(score_win, 0, 0);
    wrefresh(score_win);

    screen_update_t update;

    while (1)
    {
        zmq_recv(subscriber, &update, sizeof(screen_update_t), 0);

        //Screen Updates
        if(update.ch != 's'){
            wmove(my_win, update.pos_x, update.pos_y);
            waddch(my_win, update.ch | A_BOLD);
            wrefresh(my_win);
        }

        //Score updates
        else{
            werase(score_win); // Clear the scoreboard window

            // Display header
            mvwprintw(score_win, 1, 3, "SCORE");

            // Display each player's score
            for (int i = 0; i < update.player_count; i++)
            {
                mvwprintw(score_win, 2 + i, 3, "%c - %d", update.players[i], update.scores[i]);
            }

            box(score_win, 0, 0); // Draw the border
            wrefresh(score_win);  // Refresh to show changes
        }
    }

    // Cleanup
    endwin();
    zmq_close(subscriber);
    zmq_ctx_destroy(context);
    return 0;
}