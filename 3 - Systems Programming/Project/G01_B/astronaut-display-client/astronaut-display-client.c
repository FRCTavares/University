#include "protocol.h"

#define WINDOW_SIZE 22
#define MAX_PLAYERS 8

void initialize_zmq(void **context, void **requester);
void initialize_ncurses();
void cleanup(void *context, void *requester);
int handle_input(remote_char_t *message);
int process_server_messages(void *requester, remote_char_t *message);
void *display_thread(void *context);


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

    mvprintw(30, 0, "You are controlling character %c", info.ch);

    sleep(1);

    pthread_t thread;
    pthread_create(&thread, NULL, display_thread, (void*) context);

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
        // mvprintw(2, 0, "%d Left arrow is pressed", n);
        message->direction = LEFT;
        break;
    case KEY_RIGHT:
        // mvprintw(2, 0, "%d Right arrow is pressed", n);
        message->direction = RIGHT;
        break;
    case KEY_DOWN:
        // mvprintw(2, 0, "%d Down arrow is pressed", n);
        message->direction = DOWN;
        break;
    case KEY_UP:
        // mvprintw(2, 0, "%d Up arrow is pressed", n);
        message->direction = UP;
        break;
    case ' ':
        // mvprintw(2, 0, "%d Space key is pressed", n);
        message->msg_type = MSG_TYPE_ZAP; // Zap activation
        break;
    case 'q':
    case 'Q':
        // mvprintw(2, 0, "%d Quit key is pressed", n);
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
    mvprintw(32, 0, "Your score: %d", score);
    return 0;
}

/*
Function for the thread
Contains the code for which outer-space-display was previously responsible,
arg is the ZMQ context
 */
void *display_thread(void *context)
{

    // Initialize subscriber socket
    void *subscriber = zmq_socket(context, ZMQ_SUB);
    if (subscriber == NULL)
    {
        perror("Outer space display failed to create subscriber socket");
        zmq_ctx_term(context);
        exit(-1);
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

    /*
        Initialize the Game and Score windows side by side
        */
    WINDOW *my_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, 0);
    if (my_win == NULL)
    {
        endwin();
        perror("Failed to create window");
        exit(-1);
    }
    box(my_win, 0, 0);
    if (wrefresh(my_win) == ERR)
    {
        fprintf(stderr, "wrefresh failed\n");
        exit(-1);
    }

    WINDOW *score_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, WINDOW_SIZE + 1);
    if (score_win == NULL)
    {
        endwin();
        perror("Failed to create window");
        exit(-1);
    }
    box(score_win, 0, 0);
    if (wrefresh(score_win) == ERR)
    {
        fprintf(stderr, "wrefresh failed\n");
        exit(-1);
    }

    screen_update_t update;

    /*
    Main loop
    Wait to receive the published messages from the server and process them
    */
    while (1)
    {
        //Receive published update
        int rc = zmq_recv(subscriber, &update, sizeof(screen_update_t), 0);
        if (rc == -1)
        {
            perror("Outer space display zmq_recv failed");
            endwin();
            if (subscriber)
                zmq_close(subscriber);
            if (context)
                zmq_ctx_destroy(context);
            exit(-1);
        }

  
        // Score updates
        if(update.ch == 's')
        {
            if (werase(score_win) == ERR)
            {
                endwin();
                perror("werase failed");
                exit(-1);
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
                exit(-1);
            }
        }

        //Game over
        else if(update.ch == 'o'){
            
            werase(my_win);
            werase(score_win);
            mvwprintw(my_win, 10, 10, "GAME OVER");
            // Print final scores
            for (int i = 0; i < update.player_count; i++)
            {
                mvwprintw(my_win, 12 + i, 10, "%c - %d", update.players[i], update.scores[i]);
            }

            

            if (wrefresh(my_win) == ERR)
            {
                fprintf(stderr, "wrefresh failed\n");
                exit(-1);
            }

            if (wrefresh(score_win) == ERR)
            {
                fprintf(stderr, "wrefresh failed\n");
                exit(-1);
            }

            sleep(5);
            break;
        }

        //game window change
        else
        {
            if (wmove(my_win, update.pos_x, update.pos_y) == ERR)
            {
                fprintf(stderr, "wmove failed\n");
                exit(-1);
            }

            if (waddch(my_win, update.ch | A_BOLD) == ERR)
            {
                fprintf(stderr, "waddch failed\n");
                exit(-1);
            }

            if (wrefresh(my_win) == ERR)
            {
                fprintf(stderr, "wrefresh failed\n");
                exit(-1);
            }
        }
    }




}