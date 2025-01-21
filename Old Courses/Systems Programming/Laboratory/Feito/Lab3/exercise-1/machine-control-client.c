#include "remote-char.h"
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main() {
    srand(time(NULL)); // Seed the random number generator

    /*-------------------------------------------------------- To Do 4 --------------------------------------------------------*/
    // Open the FIFO for writing
    int fd = open(FIFO_PATH, O_WRONLY);
    if (fd == -1) {
        perror("open");
        exit(1);
    }
    /*-------------------------------------------------------------------------------------------------------------------------*/

    /*-------------------------------------------------------- To Do 6 --------------------------------------------------------*/
    // Send the connection message
    message_t msg;
    msg.msg_type = 1;  // Connection message
    msg.character = 'M';  // Machine-controlled character

    if (write(fd, &msg, sizeof(msg)) == -1) {
        perror("write");
        exit(1);
    }
    /*-------------------------------------------------------------------------------------------------------------------------*/

    // Loop to send random movement messages
    while (1) {
        usleep(500000); // 0.5 second delay between movements
        msg.msg_type = 2;        // Movement message type
        msg.direction = rand() % 4; // Random direction (UP, DOWN, LEFT, RIGHT)

        // Send the movement message to the server
        if (write(fd, &msg, sizeof(msg)) == -1) {
            perror("write");
            exit(1);
        }
    }

    close(fd); // Close the FIFO when done
    return 0;
}
