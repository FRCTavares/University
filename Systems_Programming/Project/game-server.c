#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <stdlib.h>
#include <zmq.h>
#include <string.h>
#include "zhelpers.h"

#define WINDOW_SIZE 22

int STAR_POS[8][2] = {{2, 10}, {10, 19}, {19, 10}, {10, 2}, {1, 10}, {10, 20}, {20, 10}, {10, 1}};

typedef struct ch_info_t
{
    int ch;
    int pos_x, pos_y;
    int dir; // Wether character moves vertically(1) or horizontally(0)
    int score;
} ch_info_t;

typedef struct screen_update_t
{
    int pos_x;
    int pos_y;
    char ch;
} screen_update_t;

void new_position(int *x, int *y, direction_t direction)
{
    switch (direction)
    {
    case UP:
        (*x)--;
        if (*x == 2)
            *x = 3;
        break;
    case DOWN:
        (*x)++;
        if (*x == WINDOW_SIZE - 3)
            *x = WINDOW_SIZE - 4;
        break;
    case LEFT:
        (*y)--;
        if (*y == 2)
            *y = 3;
        break;
    case RIGHT:
        (*y)++;
        if (*y == WINDOW_SIZE - 3)
            *y = WINDOW_SIZE - 4;
        break;
    default:
        break;
    }
}

int find_ch_info(ch_info_t char_data[], int n_char, int ch)
{
    for (int i = 0; i < n_char; i++)
    {
        if (ch == char_data[i].ch)
        {
            return i;
        }
    }
    return -1;
}

int main()
{

    // Initialize the list for player info - Half move vertically, half move horizontally

    // Even indices move horizontaly, odd ones move vertically

    // TREAT DISCONNECT CASE, will probably need a vector to save which indices are busy!
    ch_info_t char_data[8];
    int n_chars = 0;

    /* NEW CONTEXT AND SOCKETS*/
    void *context = zmq_ctx_new();
    void *responder = zmq_socket(context, ZMQ_REP);
    int rc = zmq_bind(responder, "tcp://0.0.0.0:5555");
    if (rc != 0)
    {
        perror("Server zmq_bind failed");
        return -1;
    }

    void *publisher = zmq_socket(context, ZMQ_PUB);
    rc = zmq_bind(publisher, "tcp://0.0.0.0:5556");
    if (rc != 0)
    {
        perror("Publisher zmq_bind failed");
        return -1;
    }

    // Initialize ncurses
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();

    WINDOW *my_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, 0);
    box(my_win, 0, 0);
    wrefresh(my_win);

    int ch;
    int pos_x;
    int pos_y;
    direction_t direction;
    remote_char_t m;
    screen_update_t update;

    /*GAME IS ONGOING*/
    /*
     4 OPTIONS of mesage type:

     -connect (0)
     -movement (1)
     -zap (2)
     -disconnect (3)




    */
    while (1)
    {

        rc = zmq_recv(responder, &m, sizeof(remote_char_t), 0);
        if (rc == -1)
        {
            perror("Server zmq_recv failed");
            return -1;
        }

        // CONNECTION

        if (m.msg_type == 0)
        {
            ch = 65 + n_chars;
            pos_x = STAR_POS[n_chars][0];
            pos_y = STAR_POS[n_chars][1];

            char_data[n_chars].ch = ch;
            char_data[n_chars].pos_x = pos_x;
            char_data[n_chars].pos_y = pos_y;
            char_data[n_chars].dir = n_chars % 2;
            char_data[n_chars].score = 0;

            rc = zmq_send(responder, &char_data[n_chars], sizeof(ch_info_t), 0);
            if (rc == -1)
            {
                perror("Server connection response failed");
                return -1;
            }

            n_chars++;
        }
        if (m.msg_type == 1)
        {
            int ch_pos = find_ch_info(char_data, n_chars, m.ch);
            if (ch_pos != -1)
            {
                pos_x = char_data[ch_pos].pos_x;
                pos_y = char_data[ch_pos].pos_y;
                ch = char_data[ch_pos].ch;

                // Erase character from old position
                wmove(my_win, pos_x, pos_y);
                waddch(my_win, ' ');
                wrefresh(my_win);

                // Send update to clear the old position
                update.pos_x = pos_x;
                update.pos_y = pos_y;
                update.ch = ' ';
                zmq_send(publisher, &update, sizeof(screen_update_t), 0);

                direction = m.direction;

                new_position(&pos_x, &pos_y, direction);
                char_data[ch_pos].pos_x = pos_x;
                char_data[ch_pos].pos_y = pos_y;
            }

            rc = zmq_send(responder, NULL, 0, 0);
            if (rc == -1)
            {
                perror("Server zmq_send failed 111");
                return -1;
            }
        }
        wmove(my_win, pos_x, pos_y);
        waddch(my_win, ch | A_BOLD);
        wrefresh(my_win);

        // Send update for the new position
        update.pos_x = pos_x;
        update.pos_y = pos_y;
        update.ch = ch;
        zmq_send(publisher, &update, sizeof(screen_update_t), 0);
    }

    endwin();
    zmq_close(responder);
    zmq_close(publisher);
    zmq_ctx_destroy(context);
    return 0;
}