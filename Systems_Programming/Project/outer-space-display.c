#include <ncurses.h>
#include <unistd.h>
#include <stdlib.h>
#include <zmq.h>
#include <string.h>
#include "protocol.h"

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
    if (context == NULL)
    {
        perror("Outer space display failed to create ZeroMQ context");
        return -1;
    }

    // Initialize subscriber socket
    void *subscriber = zmq_socket(context, ZMQ_SUB);
    if (subscriber == NULL)
    {
        perror("Outer space display failed to create subscriber socket");
        zmq_ctx_term(context);
        return -1;
    }

    if (zmq_connect(subscriber, SERVER_PUBLISH_ADDRESS) != 0)
    {
        perror("Outer space display zmq_connect failed");
        if (subscriber)
            zmq_close(subscriber);
        if (context)
            zmq_ctx_destroy(context);
        exit(-1);
    }

    // Subscribe to all messages
    if (zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0) != 0)
    {
        perror("Outer space display zmq_setsockopt failed");
        if (subscriber)
            zmq_close(subscriber);
        if (context)
            zmq_ctx_destroy(context);
        exit(-1);
    }

    // Initialize ncurses
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();

    WINDOW *my_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, 0);
    if (my_win == NULL)
    {
        endwin();
        perror("Failed to create window");
        return -1;
    }

    box(my_win, 0, 0);
    if (wrefresh(my_win) == ERR)
    {
        fprintf(stderr, "wrefresh failed\n");
        return -1;
    }

    WINDOW *score_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, WINDOW_SIZE + 1);
    if (score_win == NULL)
    {
        endwin();
        perror("Failed to create window");
        return -1;
    }

    box(score_win, 0, 0);
    if (wrefresh(score_win) == ERR)
    {
        fprintf(stderr, "wrefresh failed\n");
        return -1;
    }

    screen_update_t update;

    while (1)
    {
        int rc = zmq_recv(subscriber, &update, sizeof(screen_update_t), 0);

        if (rc == -1)
        {
            perror("Outer space display zmq_recv failed");
            endwin();
            if (subscriber)
                zmq_close(subscriber);
            if (context)
                zmq_ctx_destroy(context);
            return -1;
        }

        // Screen Updates
        if (update.ch != 's')
        {
            if (wmove(my_win, update.pos_x, update.pos_y) == ERR)
            {
                fprintf(stderr, "wmove failed\n");
                return -1;
            }

            if (waddch(my_win, update.ch | A_BOLD) == ERR)
            {
                fprintf(stderr, "waddch failed\n");
                return -1;
            }

            if (wrefresh(my_win) == ERR)
            {
                fprintf(stderr, "wrefresh failed\n");
                return -1;
            }
        }

        // Score updates
        else
        {
            if (werase(score_win) == ERR)
            {
                endwin();
                perror("werase failed");
                return -1;
            }

            // Display header
            mvwprintw(score_win, 1, 3, "SCORE");

            // Display each player's score
            for (int i = 0; i < update.player_count; i++)
            {
                mvwprintw(score_win, 2 + i, 3, "%c - %d", update.players[i], update.scores[i]);
            }

            box(score_win, 0, 0); // Draw the border

            if (wrefresh(score_win) == ERR)
            {
                fprintf(stderr, "wrefresh failed\n");
                return -1;
            }
        }
    }

    // Cleanup
    endwin();
    zmq_close(subscriber);
    zmq_ctx_destroy(context);
    return 0;
}