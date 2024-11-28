#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include <zmq.h>

int main() {
    // Initialize ZeroMQ context and requester socket
    void *context = zmq_ctx_new();
    void *requester = zmq_socket(context, ZMQ_REQ);

    int rc = zmq_connect(requester, "tcp://localhost:5555");
    if (rc != 0) {
        perror("Client zmq_connect failed");
        return -1;
    }

    // Get the character from the user
    char ch;
    do {
        printf("What is your character (a..z)?: ");
        ch = getchar();
        ch = tolower(ch);  
    } while (!isalpha(ch));

    remote_char_t m;
    m.msg_type = 0;
    m.ch = ch;
    
    // Send the initial message to the server
    printf("Client: Sending initial message...\n");
    rc = zmq_send(requester, &m, sizeof(remote_char_t), 0);
    if (rc == -1) {
        perror("Client zmq_send failed");
        return -1;
    }

    // Wait for the server response
    printf("Client: Waiting for server response...\n");
    rc = zmq_recv(requester, NULL, 0, 0);
    printf("Client: Received response from server.\n");
    if (rc == -1) {
        perror("Client zmq_recv failed");
        return -1;
    }

    // Initialize ncurses
    initscr();
    cbreak();
    keypad(stdscr, TRUE);
    noecho();

    int n = 0;
    m.msg_type = 1;
    m.ch = ch;
        
    int key;
    do {
        key = getch();		
        n++;
        switch (key) {
            case KEY_LEFT:
                mvprintw(0, 0, "%d Left arrow is pressed", n);
                m.direction = LEFT;
                break;
            case KEY_RIGHT:
                mvprintw(0, 0, "%d Right arrow is pressed", n);
                m.direction = RIGHT;
                break;
            case KEY_DOWN:
                mvprintw(0, 0, "%d Down arrow is pressed", n);
                m.direction = DOWN;
                break;
            case KEY_UP:
                mvprintw(0, 0, "%d Up arrow is pressed", n);
                m.direction = UP;
                break;
            default:
                key = 'x'; 
                break;
        }
        if (key != 'x') {
            // Send the direction message to the server
            rc = zmq_send(requester, &m, sizeof(m), 0);
            if (rc == -1) {
                perror("Client zmq_send failed");
                return -1;
            }

            // Wait for the server response
            rc = zmq_recv(requester, NULL, 0, 0);
            if (rc == -1) {
                perror("Client zmq_recv failed");
                return -1;
            }
        }
        refresh();
    } while (key != 27); // Exit on ESC key

    // Cleanup
    endwin();
    zmq_close(requester);
    zmq_ctx_destroy(context);
    return 0;
}