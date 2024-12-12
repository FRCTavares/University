#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>

#define FIFO_NAME "/tmp/fifo_execution"
#define RESULT_FIFO_NAME "/tmp/fifo_result"
#define LIBRARY_NAME "./funcs-ex5.so"
#define BUFFER_SIZE 100

int main() {
    int fd, result_fd;
    char func_name[BUFFER_SIZE];
    int arg;

    // Create FIFOs
    if (mkfifo(FIFO_NAME, 0666) == -1) perror("mkfifo");
    if (mkfifo(RESULT_FIFO_NAME, 0666) == -1) perror("mkfifo");

    // Open FIFOs
    if ((fd = open(FIFO_NAME, O_RDONLY)) == -1) {
        perror("open");
        exit(1);
    }
    if ((result_fd = open(RESULT_FIFO_NAME, O_WRONLY)) == -1) {
        perror("open result FIFO");
        exit(1);
    }

    // Load library
    void *handle = dlopen(LIBRARY_NAME, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Failed to load library: %s\n", dlerror());
        exit(1);
    }

    printf("Server ready. Waiting for function names...\n");

    while (read(fd, func_name, BUFFER_SIZE) > 0) {
        func_name[strcspn(func_name, "\n")] = '\0';
        printf("Received request for function: %s\n", func_name);

        int (*func)() = dlsym(handle, func_name);
        if (!func) {
            fprintf(stderr, "Function %s not found: %s\n", func_name, dlerror());
            continue;
        }

        if (strcmp(func_name, "f3") == 0) {
            read(fd, &arg, sizeof(arg));
            printf("Executing f3 with argument %d...\n", arg);
            int result = func(arg);
            write(result_fd, &result, sizeof(result));
        } else {
            int result = func();
            printf("Executing %s...\n", func_name);
            write(result_fd, &result, sizeof(result));
        }
    }

    dlclose(handle);
    close(fd);
    close(result_fd);
    unlink(FIFO_NAME);
    unlink(RESULT_FIFO_NAME);
    return 0;
}
