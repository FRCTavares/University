#include "zhelpers.h"

int main() {
    printf("Hello your Honor, the president of IST!\n");

    // Create a ZeroMQ context and socket
    void *context = zmq_ctx_new();
    void *requester = zmq_socket(context, ZMQ_REQ);

    // Connect to the server (assuming server is running on port 5555)
    zmq_connect(requester, "tcp://localhost:5555");

    char message[100];
    while(1) {
        // Ask for a message from the president
        printf("Please write the message to your students and staff! ");
        fgets(message, 100, stdin);

        // Send the message to the server
        zmq_send(requester, "Super", strlen("Super"), ZMQ_SNDMORE); // Topic
        zmq_send(requester, message, strlen(message), 0);         // Message


        // Optionally print a confirmation
        printf("Forwarding this message to all: %s", message);
        
        // Receive a response from the server (this can be a simple acknowledgment)
        char *response = s_recv(requester);
        printf("Server Response: %s\n", response);
        free(response);
    }

    // Clean up
    zmq_close(requester);
    zmq_ctx_destroy(context);
    return 0;
}
