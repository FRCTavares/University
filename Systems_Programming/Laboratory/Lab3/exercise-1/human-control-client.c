#include <ncurses.h>
#include "remote-char.h"
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
    /*-------------------------------------------------------- To Do 4 --------------------------------------------------------*/
    int fd = open(FIFO_PATH, O_WRONLY);
    if (fd == -1) {
        perror("open");
        exit(1);
    }
    /*-------------------------------------------------------------------------------------------------------------------------*/

    /*-------------------------------------------------------- To Do 5 --------------------------------------------------------*/
    char character;
    printf("Enter the character to control: ");
    scanf(" %c", &character);
    /*-------------------------------------------------------------------------------------------------------------------------*/
    /*-------------------------------------------------------- To Do 6 --------------------------------------------------------*/
    message_t msg;
    msg.msg_type = 1;  // Connection message
    msg.character = character;

    if (write(fd, &msg, sizeof(msg)) == -1) {
        perror("write");
        exit(1);
    }
    /*-------------------------------------------------------------------------------------------------------------------------*/

    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);

    while (1) {
        int ch = getch();
        if (ch == 27) break; // Exit on ESC key
        /*-------------------------------------------------------- To Do 9 ----------------------------------------------------*/
        msg.msg_type = 2; // Movement message
        switch (ch) {
        case KEY_UP: msg.direction = UP; break;
        case KEY_DOWN: msg.direction = DOWN; break;
        case KEY_LEFT: msg.direction = LEFT; break;
        case KEY_RIGHT: msg.direction = RIGHT; break;
        default: continue; // Ignore other keys
        }
        /*---------------------------------------------------------------------------------------------------------------------*/
        /*-------------------------------------------------------- To Do 10 ----------------------------------------------------*/
        if (write(fd, &msg, sizeof(msg)) == -1) {
            perror("write");
            exit(1);
        }
        /*---------------------------------------------------------------------------------------------------------------------*/

    }

    endwin();
    close(fd);
    return 0;
}
