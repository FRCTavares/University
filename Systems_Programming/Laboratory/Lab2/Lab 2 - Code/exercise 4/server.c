#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>

#define FIFO_NAME "/tmp/fifo_execution"
#define RESULT_FIFO_NAME "/tmp/fifo_result"  // New FIFO for sending results back to the client
#define LIBRARY_NAME "./libfuncs.so"
#define BUFFER_SIZE 100

int main() {
    int fd, result_fd;
    char func_name[BUFFER_SIZE];

    // Create the FIFOs if they don't exist
    if (mkfifo(FIFO_NAME, 0666) == -1) {
        perror("mkfifo");
    }

    if (mkfifo(RESULT_FIFO_NAME, 0666) == -1) {
        perror("mkfifo");
    }

    // Open the FIFO for reading the function names from the client
    if ((fd = open(FIFO_NAME, O_RDONLY)) == -1) {
        perror("open");
        exit(1);
    }

    // Open the FIFO for sending results to the client
    if ((result_fd = open(RESULT_FIFO_NAME, O_WRONLY)) == -1) {
        perror("open result FIFO");
        exit(1);
    }

    // Load the dynamic library
    void *handle = dlopen(LIBRARY_NAME, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Failed to load library: %s\n", dlerror());
        exit(1);
    }

    printf("Server ready. Waiting for function names...\n");

    // Read function names from the FIFO and execute them
    while (read(fd, func_name, BUFFER_SIZE) > 0) {
        func_name[strcspn(func_name, "\n")] = '\0'; // Remove newline character
        printf("Received request for function: %s\n", func_name);

        // Find the function in the library
        int (*func)() = dlsym(handle, func_name);
        if (!func) {
            fprintf(stderr, "Function %s not found in library: %s\n", func_name, dlerror());
            continue;
        }

        // Execute the function
        int result = func();
        printf("Function %s executed. Result: %d\n", func_name, result);

        // Send the result back to the client
        if (write(result_fd, &result, sizeof(result)) == -1) {
            perror("write result");
            continue;
        }
    }

    // Cleanup
    dlclose(handle);
    close(fd);
    close(result_fd);
    unlink(FIFO_NAME);
    unlink(RESULT_FIFO_NAME);

    return 0;
}
