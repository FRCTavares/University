#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include <zmq.h>
#include <stdio.h>

typedef struct ch_info_t
{
    int ch;
    int pos_x, pos_y;
    int dir; // Whether character moves vertically(1) or horizontally(0)
    int score;
} ch_info_t;

// Add a new message type for game end
#define MSG_TYPE_GAME_END 4

int main()
{

    void *context = zmq_ctx_new();
    void *requester = zmq_socket(context, ZMQ_REQ);

    // Add this after initializing the requester socket
    void *subscriber = zmq_socket(context, ZMQ_SUB);
    zmq_connect(subscriber, "tcp://localhost:5557");
    zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);

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

    mvprintw(0, 0, "You are controlling character %c", r.ch);

    int n = 0;
    m.msg_type = 1;

    int key;

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
        case KEY_DOWN:
            mvprintw(2, 0, "%d Down arrow is pressed", n);
            m.direction = DOWN;
            break;
        case KEY_UP:
            mvprintw(2, 0, "%d Up arrow is pressed", n);
            m.direction = UP;
            break;
        case ' ':
            mvprintw(2, 0, "%d Space key is pressed", n);
            m.msg_type = 2; // Special value to indicate zap activation
            break;
        case 'q':
        case 'Q':
            mvprintw(2, 0, "%d Quit key is pressed", n);
            m.msg_type = 3; // Special value to indicate disconnect
            rc = zmq_send(requester, &m, sizeof(m), 0);
            if (rc == -1)
            {
                perror("Client zmq_send failed");
                return -1;
            }
            // Wait for the server response
            int response;
            rc = zmq_recv(requester, &response, sizeof(int), 0);
            if (rc == -1)
            {
                perror("Client zmq_recv failed");
                return -1;
            }
            // Print final score and exit
            mvprintw(3, 0, "Client disconnected, Final Score: %d", r.score);
            refresh();
            sleep(2); // Wait for 2 seconds to display the message
            endwin();
            zmq_close(requester);
            zmq_ctx_destroy(context);
            return 0;
        default:
            key = 'x';
            break;
        }
        if (key != 'x')
        {
            // Send the direction, zap activation, or disconnect message to the server
            rc = zmq_send(requester, &m, sizeof(m), 0);
            if (rc == -1)
            {
                perror("Client zmq_send failed");
                return -1;
            }

            // Wait for the server response
            int score;
            rc = zmq_recv(requester, &score, sizeof(int), 0);
            if (rc == -1)
            {
                perror("Client zmq_recv failed");
                return -1;
            }

            mvprintw(1, 0, "Your score: %d", score);

            // Reset message type to movement after zap activation
            if (m.msg_type == 2)
            {
                m.msg_type = 1;
            }
        }

        // Inside the main loop, after processing any message
        zmq_pollitem_t items[] = {
            {requester, 0, ZMQ_POLLIN, 0},
            {subscriber, 0, ZMQ_POLLIN, 0}};
        zmq_poll(items, 2, 0);

        if (items[0].revents & ZMQ_POLLIN)
        {
            // Handle replies from server as before
        }

        if (items[1].revents & ZMQ_POLLIN)
        {
            remote_char_t end_msg;
            rc = zmq_recv(subscriber, &end_msg, sizeof(remote_char_t), 0);
            if (end_msg.msg_type == MSG_TYPE_GAME_END)
            {
                mvprintw(3, 0, "The game ended, your final score is: %d", r.score);
                refresh();
                sleep(2); // Wait for 2 seconds to display the message
                endwin();
                zmq_close(requester);
                zmq_close(subscriber);
                zmq_ctx_destroy(context);
                return 0;
            }
        }

        refresh();
    } while (key != 27); // Exit on ESC key

    endwin();
    zmq_close(requester);
    zmq_ctx_destroy(context);
    return 0;
}