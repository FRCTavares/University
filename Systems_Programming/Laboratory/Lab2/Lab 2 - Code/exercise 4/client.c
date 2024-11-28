#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#define FIFO_NAME "/tmp/fifo_execution"
#define RESULT_FIFO_NAME "/tmp/fifo_result"
#define BUFFER_SIZE 100

int main() {
    int fd, result_fd;
    char func_name[BUFFER_SIZE];
    int result;

    // Open the FIFO for writing function names to the server
    if ((fd = open(FIFO_NAME, O_WRONLY)) == -1) {
        perror("open");
        exit(1);
    }

    // Open the FIFO for reading the results from the server
    if ((result_fd = open(RESULT_FIFO_NAME, O_RDONLY)) == -1) {
        perror("open result FIFO");
        exit(1);
    }

    printf("Client ready. Type function names to execute (e.g., f1, f2):\n");

    // Send function names to the server and receive results
    while (1) {
        printf("Enter function name: ");
        if (!fgets(func_name, BUFFER_SIZE, stdin)) {
            perror("fgets");
            break;
        }

        // Write the function name to the FIFO
        if (write(fd, func_name, strlen(func_name)) == -1) {
            perror("write");
            break;
        }

        // Read the result from the server
        if (read(result_fd, &result, sizeof(result)) == -1) {
            perror("read result");
            break;
        }

        // Print the result
        printf("Received result: %d\n", result);
    }

    close(fd);
    close(result_fd);
    return 0;
}
