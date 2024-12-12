#include <zmq.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

int main (void)
{
    printf("Connecting to hello world server…\n");

    // Create a new ZeroMQ context
    void *context = zmq_ctx_new();
    void *requester = zmq_socket(context, ZMQ_REQ);

    // Connect to the server
    int rc = zmq_connect(requester, "ipc:///tmp/s1");
    if (rc != 0) {
        perror("zmq_connect failed");
        return 1;
    }

    while (1) {
        int n;
        char buffer[10];

        // Get input from the user
        printf("Type an integer: ");
        scanf("%d", &n);
        printf("Sending number %d...\n", n);

        // Send the number to the server
        int sent = zmq_send(requester, &n, sizeof(n), 0);
        if (sent == -1) {
            perror("zmq_send failed");
            return 1;
        }

        printf("Sent %d bytes\n", sent);

        // Receive the reply from the server
        int received = zmq_recv(requester, &n, sizeof(n), 0);
        if (received == -1) {
            perror("zmq_recv failed");
            return 1;
        }

        printf("Received number %d\n", n);
    }

    zmq_close(requester);
    zmq_ctx_destroy(context);
    return 0;
}
