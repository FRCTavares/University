#include <zmq.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

int main (void)
{
    //  Socket to talk to clients
    void *context = zmq_ctx_new ();
    void *responder = zmq_socket (context, ZMQ_REP);
    
    // Bind to the IPC endpoint
    int rc = zmq_bind (responder, "ipc:///tmp/s1");
    if (rc != 0) {
        perror("zmq_bind failed");
        return 1;
    }

    while (1) {
        int n;
        
        // Receive a message from the client
        int received = zmq_recv(responder, &n, sizeof(n), 0);
        if (received == -1) {
            perror("zmq_recv failed");
            return 1;
        }

        printf("Received %d\n", n);

        n = n * 2;  // Process the data
        sleep(5);   // Simulate some processing time
        printf("Sending Reply %d\n", n);

        // Send the modified data back to the client
        int sent = zmq_send(responder, &n, sizeof(n), 0);
        if (sent == -1) {
            perror("zmq_send failed");
            return 1;
        }

        printf("Sent %d bytes\n", sent);
    }

    zmq_close(responder);
    zmq_ctx_destroy(context);
    return 0;
}
