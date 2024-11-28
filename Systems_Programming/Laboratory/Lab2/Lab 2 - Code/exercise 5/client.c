#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#define FIFO_NAME "/tmp/fifo_execution"
#define RESULT_FIFO_NAME "/tmp/fifo_result"
#define BUFFER_SIZE 100

int main() {
    int fd, result_fd;
    char func_name[BUFFER_SIZE];
    int arg, result;

    // Open the FIFOs
    if ((fd = open(FIFO_NAME, O_WRONLY)) == -1) {
        perror("open");
        exit(1);
    }
    if ((result_fd = open(RESULT_FIFO_NAME, O_RDONLY)) == -1) {
        perror("open result FIFO");
        exit(1);
    }

    printf("Client ready. Type function names and arguments (if required) to execute:\n");

    while (1) {
        // Get the function name
        printf("Enter function name (f1, f2, f3): ");
        if (!fgets(func_name, BUFFER_SIZE, stdin)) {
            perror("fgets");
            break;
        }
        func_name[strcspn(func_name, "\n")] = '\0'; // Remove newline character

        // Send the function name
        if (write(fd, func_name, strlen(func_name)) == -1) {
            perror("write");
            break;
        }

        // If function is f3, send an argument
        if (strcmp(func_name, "f3") == 0) {
            printf("Enter an integer argument: ");
            while (scanf("%d", &arg) != 1) {
                printf("Invalid input. Please enter an integer: ");
                while (getchar() != '\n');
            }
            getchar(); // Consume newline
            write(fd, &arg, sizeof(arg));
        }

        // Read the result from the server
        if (read(result_fd, &result, sizeof(result)) > 0) {
            printf("Result from server: %d\n", result);
        } else {
            perror("Error reading result");
            break;
        }
    }

    close(fd);
    close(result_fd);
    return 0;
}
