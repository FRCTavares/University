#include "protocol.h"

/*
Function to delete the necessary zmq variables

param:
    void *context: pointer to the zmq context
    void *requester: pointer to the zmq requester socket

return:
    void
*/
void cleanup(void *context, void *requester)
{
    endwin();
    if (requester)
        zmq_close(requester);
    if (context)
        zmq_ctx_destroy(context);
}

/*
Function to intialize zmq context and socket
Connects the requester socket to the server

param:
    void **context: pointer to the zmq context
    void **requester: pointer to the zmq requester socket

return:
    void
*/
void initialize_zmq(void **context, void **requester)
{
    *context = zmq_ctx_new();
    if (context == NULL)
    {
        perror("Client failed to create ZeroMQ context");
        exit(-1);
    }

    *requester = zmq_socket(*context, ZMQ_REQ);
    if (requester == NULL)
    {
        perror("Client failed to create requester socket");
        zmq_ctx_term(context);
        exit(-1);
    }

    if (zmq_connect(*requester, SERVER_REQUEST_ADDRESS) != 0)
    {
        perror("Client zmq_connect failed");
        cleanup(*context, *requester);
        exit(-1);
    }
}

/*
Function to initialize the ncurses environment

param:
    void

return:
    void
*/
void initialize_ncurses()
{
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();
}

/*
Function to handke the input from the keyboard
It detects which key is pressed and puts the necessary inormation in the message variable

param:
    remote_char_t *message: pointer to the message variable

return:
    int: 0 if the game is still running, 1 if the user wants to quit, -1 if the key is not recognized
*/
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

/*
Function which sends message to the server and
    processes the replies sent back by the server

param:
    void *requester: pointer to the zmq requester socket
    remote_char_t *message: pointer to the message variable

return:
    int: 0 if the game is still running, -1 if the user wants to quit
*/
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

int main()
{
    void *context = NULL;
    void *requester = NULL;

    remote_char_t message = {0};
    ch_info_t info = {0};
    int running = 1;

    initialize_zmq(&context, &requester);
    initialize_ncurses();

    // Send initial message to server
    printf("Client: Sending initial message...\n");
    if (zmq_send(requester, &message, sizeof(remote_char_t), 0) == -1)
    {
        perror("Client zmq_send failed");
        cleanup(context, requester);
        return -1;
    }

    // Receive character info from server
    if (zmq_recv(requester, &info, sizeof(ch_info_t), 0) == -1)
    {
        perror("Client zmq_recv failed");
        cleanup(context, requester);
        return -1;
    }

    // Update the messsage variable with the character information
    // This information will be the same throughout the whole game
    message.ch = info.ch;
    message.GAME_TOKEN = info.GAME_TOKEN;

    // if char is 0 then the server is full
    if (info.ch == 0)
    {
        mvprintw(0, 0, "Server is full. Please try again later.");
        cleanup(context, requester);
        return -1;
    }

    mvprintw(0, 0, "You are controlling character %c", info.ch);

    while (running)
    {
        int input_result = handle_input(&message);
        if (input_result == 0)
        {
            if (process_server_messages(requester, &message) == -1)
            {
                cleanup(context, requester);
                return -1;
            }
        }
        else if (input_result == 1)
        {
            if (process_server_messages(requester, &message) == -1)
            {
                cleanup(context, requester);
                return -1;
            }
            running = 0;
        }

        refresh();
    }

    cleanup(context, requester);
    return 0;
}
