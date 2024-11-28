#include "zhelpers.h"

int main(){
    char line[100];
    char dpt_name[100];

    // Prompt for department name
    printf("What is the department of this building? (DEEC, DEI, ...)");
    fgets(line, 100, stdin);
    sscanf(line, "%s", dpt_name);  // Read department name

    printf("We will broadcast all messages from the president of IST and %s\n", dpt_name);

    // Initialize ZeroMQ context and socket
    void *context = zmq_ctx_new();
    void *subscriber = zmq_socket(context, ZMQ_SUB);
    
    // Connect to the server's publishing socket
    zmq_connect(subscriber, "tcp://localhost:5556");
    
    // Subscribe to messages for the department and 'super'
    zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, dpt_name, strlen(dpt_name));
    zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "Super", strlen("Super"));
    
    char message[100];
    while (1) {
        // Receive the topic
        char *topic = s_recv(subscriber);
        // Receive the message
        char *message = s_recv(subscriber);

        // Print the message
        printf("Message from %s: %s\n", topic, message);

        // Free the received strings
        free(topic);
        free(message);
    }

    // Clean up
    zmq_close(subscriber);
    zmq_ctx_destroy(context);
    return 0;
}
