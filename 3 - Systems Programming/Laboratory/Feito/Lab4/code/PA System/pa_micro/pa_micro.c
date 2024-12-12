#include "zhelpers.h"

int main(){
    char line[100];
    char dpt_name[100];

    // Prompt for department name
    printf("What is your department? (DEEC, DEI, ...)");
    fgets(line, 100, stdin);
    sscanf(line, "%s", dpt_name);

    printf("Hello your Honor, the President of %s\n", dpt_name);

    // Connect to the server using ZMQ_REQ
    void *context = zmq_ctx_new ();
    void *subscriber = zmq_socket (context, ZMQ_REQ);

    // Connect to the server (assuming server is running on port 5555)
    zmq_connect (subscriber, "tcp://localhost:5555");

    char message[100];
    while(1){
        // Ask for a message from the president
        printf("Please write the message to the students and staff on your buildings! ");
        fgets(message, 100, stdin);

        //send message to server
        zmq_send(subscriber, dpt_name, strlen(dpt_name), ZMQ_SNDMORE);
        zmq_send(subscriber, message, strlen(message), 0);

        printf("Forwarding this message to all: %s", message);

        //Extra: Receive and print a response from the server (ACK)
        char *response = s_recv(subscriber);
        printf("Server Response: %s\n", response);
        free(response);

    }

    zmq_close (subscriber);
    zmq_ctx_destroy (context);
    return 0;
}