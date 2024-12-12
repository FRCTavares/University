#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <stdlib.h>
#include <zmq.h>
#include <string.h>

#define WINDOW_SIZE 15 
#define AUTH_TOKEN "Pass" // Define the authorization token

typedef struct ch_info_t {
    int ch;
    int pos_x, pos_y;
} ch_info_t;

typedef struct screen_update_t {
    int pos_x;
    int pos_y;
    char ch;
} screen_update_t;

// Function to generate a random direction
direction_t random_direction() {
    return random() % 4;
}

// Function to update the position based on the direction
void new_position(int* x, int *y, direction_t direction) {
    switch (direction) {
        case UP:
            (*x)--;
            if (*x == 0)
                *x = 2;
            break;
        case DOWN:
            (*x)++;
            if (*x == WINDOW_SIZE - 1)
                *x = WINDOW_SIZE - 3;
            break;
        case LEFT:
            (*y)--;
            if (*y == 0)
                *y = 2;
            break;
        case RIGHT:
            (*y)++;
            if (*y == WINDOW_SIZE - 1)
                *y = WINDOW_SIZE - 3;
            break;
        default:
            break;
    }
}

// Function to find character information
int find_ch_info(ch_info_t char_data[], int n_char, int ch) {
    for (int i = 0; i < n_char; i++) {
        if (ch == char_data[i].ch) {
            return i;
        }
    }
    return -1;
}

int main() {	
    ch_info_t char_data[100];
    int n_chars = 0;

    // Initialize ZeroMQ context and sockets
    void *context = zmq_ctx_new();
    void *responder = zmq_socket(context, ZMQ_REP);
    int rc = zmq_bind(responder, "tcp://0.0.0.0:5555");
    if (rc != 0) {
        perror("Server zmq_bind failed");
        return -1;
    }

    void *publisher = zmq_socket(context, ZMQ_PUB);
    rc = zmq_bind(publisher, "tcp://0.0.0.0:5556");
    if (rc != 0) {
        perror("Publisher zmq_bind failed");
        return -1;
    }

    void *auth_responder = zmq_socket(context, ZMQ_REP);
    rc = zmq_bind(auth_responder, "tcp://0.0.0.0:5557");
    if (rc != 0) {
        perror("Auth zmq_bind failed");
        return -1;
    }

    // Initialize ncurses
    initscr();		    	
    cbreak();				
    keypad(stdscr, TRUE);   
    noecho();			    

    WINDOW *my_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, 0);
    box(my_win, 0 , 0);	
    wrefresh(my_win);

    int ch;
    int pos_x;
    int pos_y;
    direction_t direction;
    remote_char_t m;
    screen_update_t update;

    while (1) {
        // Poll for incoming messages
        zmq_pollitem_t items[] = {
            { responder, 0, ZMQ_POLLIN, 0 },
            { auth_responder, 0, ZMQ_POLLIN, 0 }
        };
        zmq_poll(items, 2, -1);

        // Handle game messages
        if (items[0].revents & ZMQ_POLLIN) {
            rc = zmq_recv(responder, &m, sizeof(remote_char_t), 0);
            if (rc == -1) {
                perror("Server zmq_recv failed");
                return -1;
            }
            if (m.msg_type == 0) {
                ch = m.ch;
                pos_x = WINDOW_SIZE / 2;
                pos_y = WINDOW_SIZE / 2;

                char_data[n_chars].ch = ch;
                char_data[n_chars].pos_x = pos_x;
                char_data[n_chars].pos_y = pos_y;
                n_chars++;
            }
            if (m.msg_type == 1) {
                int ch_pos = find_ch_info(char_data, n_chars, m.ch);
                if (ch_pos != -1) {
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
            }
            wmove(my_win, pos_x, pos_y);
            waddch(my_win, ch | A_BOLD);
            wrefresh(my_win);

            // Send update for the new position
            update.pos_x = pos_x;
            update.pos_y = pos_y;
            update.ch = ch;
            zmq_send(publisher, &update, sizeof(screen_update_t), 0);

            rc = zmq_send(responder, NULL, 0, 0);
            if (rc == -1) {
                perror("Server zmq_send failed");
                return -1;
            }
        }

        // Handle authentication messages
        if (items[1].revents & ZMQ_POLLIN) {
            char token[256];
            rc = zmq_recv(auth_responder, token, 256, 0);
            if (rc == -1) {
                perror("Auth zmq_recv failed");
                return -1;
            }
            token[rc] = '\0'; // Null-terminate the received token

            if (strcmp(token, AUTH_TOKEN) == 0) {
                zmq_send(auth_responder, "OK", 2, 0);
            } else {
                zmq_send(auth_responder, "FAIL", 4, 0);
            }
        }
    }

    // Cleanup
    endwin();
    zmq_close(responder);
    zmq_close(publisher);
    zmq_close(auth_responder);
    zmq_ctx_destroy(context);
    return 0;
}