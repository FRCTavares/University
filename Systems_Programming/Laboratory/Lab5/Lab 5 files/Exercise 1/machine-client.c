// machine-client.c
#include "remote-char.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <zmq.h>

int main() {	 
    // Initialize ZeroMQ context and requester socket
    void *context = zmq_ctx_new();
    void *requester = zmq_socket(context, ZMQ_REQ);

    int rc = zmq_connect(requester, "tcp://localhost:5555");
    if (rc != 0) {
        perror("Client zmq_connect failed");
        return -1;
    }

    // Get the character from the user
    char ch;
    do {
        printf("What is your symbol (a..z)?: ");
        ch = getchar();
        ch = tolower(ch);  
    } while (!isalpha(ch));

    remote_char_t m;
    m.msg_type = 0;
    m.ch = ch;
    
    // Send the initial message to the server
    printf("Client: Sending initial message...\n");
    rc = zmq_send(requester, &m, sizeof(remote_char_t), 0);
    if (rc == -1) {
        perror("Client zmq_send failed");
        return -1;
    }

    // Wait for the server response
    printf("Client: Waiting for server response...\n");
    rc = zmq_recv(requester, NULL, 0, 0);
    printf("Client: Received response from server.\n");
    if (rc == -1) {
        perror("Client zmq_recv failed");
        return -1;
    }

    int sleep_delay;
    direction_t direction;
    int n = 0;
    while (1) {
        n++;
        sleep_delay = random() % 700000;
        usleep(sleep_delay);
        direction = random() % 4;
        switch (direction) {
            case LEFT:
                printf("%d Going Left\n", n);
                break;
            case RIGHT:
                printf("%d Going Right\n", n);
                break;
            case DOWN:
                printf("%d Going Down\n", n);
                break;
            case UP:
                printf("%d Going Up\n", n);
                break;
        }
        m.direction = direction;
        m.msg_type = 1;

        // Send the direction message to the server
        rc = zmq_send(requester, &m, sizeof(m), 0);
        if (rc == -1) {
            perror("Client zmq_send failed");
            return -1;
        }

        // Wait for the server response
        rc = zmq_recv(requester, NULL, 0, 0);
        if (rc == -1) {
            perror("Client zmq_recv failed");
            return -1;
        }
    }

    // Cleanup
    zmq_close(requester);
    zmq_ctx_destroy(context);
    return 0;
}