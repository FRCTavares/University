#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#define FIFO_NAME "/tmp/fifo_execution"
#define BUFFER_SIZE 100

int main() {
    int fd;
    char func_name[BUFFER_SIZE];

    // Open the FIFO for writing
    if ((fd = open(FIFO_NAME, O_WRONLY)) == -1) {
        perror("open");
        exit(1);
    }

    printf("Client ready. Enter function names to execute:\n");

    // Read function names from the user and send them to the server
    while (1) {
        printf("Enter function name: ");
        if (fgets(func_name, BUFFER_SIZE, stdin) == NULL) {
            break;
        }

        if (write(fd, func_name, strlen(func_name) + 1) == -1) {
            perror("write");
            break;
        }
    }

    close(fd);
    return 0;
}
