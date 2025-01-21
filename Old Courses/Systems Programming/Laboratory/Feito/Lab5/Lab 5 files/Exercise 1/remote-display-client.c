// remote-display-client.c
#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <stdlib.h>
#include <zmq.h>
#include <string.h>

#define WINDOW_SIZE 15
#define AUTH_TOKEN "Pass" // Define the authorization token

typedef struct screen_update_t {
    int pos_x;
    int pos_y;
    char ch;
} screen_update_t;

int main() {
    // Initialize ZeroMQ context and authentication socket
    void *context = zmq_ctx_new();
    void *auth_requester = zmq_socket(context, ZMQ_REQ);
    zmq_connect(auth_requester, "tcp://localhost:5557");

    // Send the authorization token
    zmq_send(auth_requester, AUTH_TOKEN, strlen(AUTH_TOKEN), 0);

    // Receive the authentication response
    char reply[256];
    int rc = zmq_recv(auth_requester, reply, 256, 0);
    if (rc == -1) {
        perror("Auth zmq_recv failed");
        return -1;
    }
    reply[rc] = '\0'; // Null-terminate the received reply

    // Check if authentication was successful
    if (strcmp(reply, "OK") != 0) {
        printf("Authorization failed\n");
        zmq_close(auth_requester);
        zmq_ctx_destroy(context);
        return -1;
    }

    zmq_close(auth_requester);

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
    box(my_win, 0 , 0);	
    wrefresh(my_win);

    screen_update_t update;

    // Main loop to receive and display updates
    while (1) {
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