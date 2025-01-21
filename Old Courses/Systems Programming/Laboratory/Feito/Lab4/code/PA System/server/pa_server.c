#include "zhelpers.h"

int main (void) {
    // Initialize the context
    void *context = zmq_ctx_new();
    
    // Create the REP socket to receive messages from the microphones
    void *receiver = zmq_socket(context, ZMQ_REP);
    zmq_bind(receiver, "tcp://*:5555");

    // Create the PUB socket to send messages to the speakers
    void *publisher = zmq_socket(context, ZMQ_PUB);
    zmq_bind(publisher, "tcp://*:5556");

    printf("Server is ready...\n");

    while (1) {
        // Receive message from a microphone (super or department)
        char *dpt_name = s_recv(receiver);  // Department name (topic)
        char *message = s_recv(receiver);    // Message content
        
        // Print the received message (for debugging purposes)
        printf("Received message from department %s: %s\n", dpt_name, message);

        // Forward the message to all speakers (if the message is from the super microphone, no filtering)
        if (strcmp(dpt_name, "Super") == 0) {
            zmq_send(publisher, "Super", strlen("Super"), ZMQ_SNDMORE); // Send topic 'super'
        } else {
            zmq_send(publisher, dpt_name, strlen(dpt_name), ZMQ_SNDMORE); // Send topic (department)
        }
        zmq_send(publisher, message, strlen(message), 0);  // Send the actual message
        
        // Optionally, send a reply back to the microphone (e.g., acknowledgment)
        zmq_send(receiver, "Message received", 16, 0);

        // Free memory allocated for strings
        free(dpt_name);
        free(message);
    }

    // Cleanup
    zmq_close(receiver);
    zmq_close(publisher);
    zmq_ctx_destroy(context);
    return 0;
}
