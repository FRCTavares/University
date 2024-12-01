#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include <zmq.h>

typedef struct ch_info_t
{
    int ch;
    int pos_x, pos_y;
    int dir; // Wether character moves vertically(1) or horizontally(0)
    int score;
} ch_info_t;

int main()
{

    void *context = zmq_ctx_new();
    void *requester = zmq_socket(context, ZMQ_REQ);

    int rc = zmq_connect(requester, "tcp://localhost:5555");
    if (rc != 0)
    {
        perror("Client zmq_connect failed");
        return -1;
    }

    remote_char_t m;
    m.msg_type = 0;

    printf("Client: Sending initial message...\n");
    rc = zmq_send(requester, &m, sizeof(remote_char_t), 0);
    if (rc == -1)
    {
        perror("Client zmq_send failed");
        return -1;
    }

    ch_info_t r;

    rc = zmq_recv(requester, &r, sizeof(ch_info_t), 0);
    if (rc == -1)
    {
        perror("Client zmq_recv failed");
        return -1;
    }
    printf("You will control character %c\n", r.ch);
    m.ch = r.ch;

    


    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();

    mvprintw(0,0, "You are controlling character %c", r.ch);

    int n = 0;
    m.msg_type = 1;

    int key;

    if (r.dir == 0)
    {
        do
        {
            key = getch();
            n++;
            switch (key)
            {
            case KEY_LEFT:
                mvprintw(2, 0, "%d Left arrow is pressed", n);
                m.direction = LEFT;
                break;
            case KEY_RIGHT:
                mvprintw(2, 0, "%d Right arrow is pressed", n);
                m.direction = RIGHT;
                break;
            default:
                key = 'x';
                break;
            }
            if (key != 'x')
            {
                // Send the direction message to the server
                rc = zmq_send(requester, &m, sizeof(m), 0);
                if (rc == -1)
                {
                    perror("Client zmq_send failed");
                    return -1;
                }

                // Wait for the server response
                rc = zmq_recv(requester, NULL, 0, 0);
                if (rc == -1)
                {
                    perror("Client zmq_recv failed");
                    return -1;
                }
            }
            refresh();
        } while (key != 27); // Exit on ESC key
    }

    else if (r.dir == 1)
    {

        do
        {
            key = getch();
            n++;
            switch (key)
            {
            case KEY_DOWN:
                mvprintw(2, 0, "%d Down arrow is pressed", n);
                m.direction = DOWN;
                break;
            case KEY_UP:
                mvprintw(2, 0, "%d Up arrow is pressed", n);
                m.direction = UP;
                break;
            default:
                key = 'x';
                break;
            }
            if (key != 'x')
            {
                // Send the direction message to the server
                rc = zmq_send(requester, &m, sizeof(m), 0);
                if (rc == -1)
                {
                    perror("Client zmq_send failed");
                    return -1;
                }

                // Wait for the server response
                rc = zmq_recv(requester, NULL, 0, 0);
                if (rc == -1)
                {
                    perror("Client zmq_recv failed");
                    return -1;
                }
            }
            refresh();
        } while (key != 27); // Exit on ESC key
    }

    endwin();
    zmq_close(requester);
    zmq_ctx_destroy(context);
    return 0;
}