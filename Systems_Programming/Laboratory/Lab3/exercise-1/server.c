#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#define WINDOW_SIZE 15
#define MAX_CLIENTS 10

typedef struct {
    char character;
    int pos_x;
    int pos_y;
} client_t;

client_t clients[MAX_CLIENTS];
int client_count = 0;

void new_position(int *x, int *y, direction_t direction) {
    switch (direction) {
        case UP:
            (*x)--;
            if (*x < 1) *x = 1;
            break;
        case DOWN:
            (*x)++;
            if (*x >= WINDOW_SIZE - 1) *x = WINDOW_SIZE - 2;
            break;
        case LEFT:
            (*y)--;
            if (*y < 1) *y = 1;
            break;
        case RIGHT:
            (*y)++;
            if (*y >= WINDOW_SIZE - 1) *y = WINDOW_SIZE - 2;
            break;
    }
}

int main() {
    int fd;
    if (mkfifo(FIFO_PATH, 0666) == -1 && errno != EEXIST) {
        perror("mkfifo");
        exit(1);
    }

    fd = open(FIFO_PATH, O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(1);
    }

    initscr();
    cbreak();
    noecho();

    WINDOW *my_win = newwin(WINDOW_SIZE, WINDOW_SIZE, 0, 0);
    box(my_win, 0, 0);
    wrefresh(my_win);

    message_t msg;

    while (1) {
        if (read(fd, &msg, sizeof(msg)) > 0) {
            if (msg.msg_type == 1) { // Connection message
                if (client_count < MAX_CLIENTS) {
                    clients[client_count].character = msg.character;
                    clients[client_count].pos_x = WINDOW_SIZE / 2;
                    clients[client_count].pos_y = WINDOW_SIZE / 2;
                    client_count++;
                } else {
                    fprintf(stderr, "Max clients reached. Cannot add more clients.\n");
                }
            } else if (msg.msg_type == 2) { // Movement message
                for (int i = 0; i < client_count; i++) {
                    if (clients[i].character == msg.character) {
                        new_position(&clients[i].pos_x, &clients[i].pos_y, msg.direction);
                        break;
                    }
                }
            }

            werase(my_win);
            box(my_win, 0, 0);
            for (int i = 0; i < client_count; i++) {
                mvwaddch(my_win, clients[i].pos_x, clients[i].pos_y, clients[i].character | A_BOLD);
            }
            wrefresh(my_win);
        }
    }

    endwin();
    close(fd);
    unlink(FIFO_PATH);
    return 0;
}
