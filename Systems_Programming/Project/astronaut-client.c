#include <ncurses.h>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include <zmq.h>
#include <stdio.h>
#include "protocol.h"

void initialize_zmq(void **context, void **requester, void **subscriber);
void initialize_ncurses();
void cleanup(void *context, void *requester, void *subscriber);
int handle_input(remote_char_t *message);
int process_server_messages(void *requester, remote_char_t *message);

int main()
{
    void *context = NULL;
    void *requester = NULL;
    void *subscriber = NULL;
    remote_char_t message = {0};
    ch_info_t info = {0};
    int running = 1;

    initialize_zmq(&context, &requester, &subscriber);
    initialize_ncurses();

    // Send initial message to server
    printf("Client: Sending initial message...\n");
    if (zmq_send(requester, &message, sizeof(remote_char_t), 0) == -1)
    {
        perror("Client zmq_send failed");
        cleanup(context, requester, subscriber);
        return -1;
    }

    // Receive character info from server
    if (zmq_recv(requester, &info, sizeof(ch_info_t), 0) == -1)
    {
        perror("Client zmq_recv failed");
        cleanup(context, requester, subscriber);
        return -1;
    }
    message.ch = info.ch;
    mvprintw(0, 0, "You are controlling character %c", info.ch);

    while (running)
    {
        int input_result = handle_input(&message);
        if (input_result == 0)
        {
            if (process_server_messages(requester, &message) == -1)
            {
                cleanup(context, requester, subscriber);
                return -1;
            }
        }
        else if (input_result == 1)
        {
            if (process_server_messages(requester, &message) == -1)
            {
                cleanup(context, requester, subscriber);
                return -1;
            }
            running = 0;
        }

        refresh();
    }

    cleanup(context, requester, subscriber);
    return 0;
}

void initialize_zmq(void **context, void **requester, void **subscriber)
{
    *context = zmq_ctx_new();
    *requester = zmq_socket(*context, ZMQ_REQ);
    *subscriber = zmq_socket(*context, ZMQ_SUB);

    zmq_connect(*subscriber, "tcp://localhost:5557");
    zmq_setsockopt(*subscriber, ZMQ_SUBSCRIBE, "", 0);

    if (zmq_connect(*requester, "tcp://localhost:5555") != 0)
    {
        perror("Client zmq_connect failed");
        cleanup(*context, *requester, *subscriber);
        exit(-1);
    }
}

void initialize_ncurses()
{
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();
}

void cleanup(void *context, void *requester, void *subscriber)
{
    endwin();
    if (requester)
        zmq_close(requester);
    if (subscriber)
        zmq_close(subscriber);
    if (context)
        zmq_ctx_destroy(context);
}

int handle_input(remote_char_t *message)
{
    int key = getch();
    static int n = 0;
    n++;

    // Set message type
    message->msg_type = MSG_TYPE_MOVE;

    switch (key)
    {
    case KEY_LEFT:
        mvprintw(2, 0, "%d Left arrow is pressed", n);
        message->direction = LEFT;
        break;
    case KEY_RIGHT:
        mvprintw(2, 0, "%d Right arrow is pressed", n);
        message->direction = RIGHT;
        break;
    case KEY_DOWN:
        mvprintw(2, 0, "%d Down arrow is pressed", n);
        message->direction = DOWN;
        break;
    case KEY_UP:
        mvprintw(2, 0, "%d Up arrow is pressed", n);
        message->direction = UP;
        break;
    case ' ':
        mvprintw(2, 0, "%d Space key is pressed", n);
        message->msg_type = MSG_TYPE_ZAP; // Zap activation
        break;
    case 'q':
    case 'Q':
        mvprintw(2, 0, "%d Quit key is pressed", n);
        message->msg_type = MSG_TYPE_DISCONNECT; // Disconnect
        return 1;
    default:
        // Unrecognized key
        return -1;
    }
    return 0;
}

int process_server_messages(void *requester, remote_char_t *message)
{
    if (zmq_send(requester, message, sizeof(remote_char_t), 0) == -1)
    {
        perror("Client zmq_send failed");
        return -1;
    }

    int reply;
    if (zmq_recv(requester, &reply, sizeof(int), 0) == -1)
    {
        perror("Client zmq_recv failed");
        return -1;
    }

    if (reply == -2) // Disconnection confirmed
    {
        mvprintw(3, 0, "You have been disconnected from the server");
        return -1;
    }
    int score = reply;
    mvprintw(1, 0, "Your score: %d", score);
    return 0;
}
