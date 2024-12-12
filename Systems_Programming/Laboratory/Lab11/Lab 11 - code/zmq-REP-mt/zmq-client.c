//  Hello World client
#include <zmq.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>



int main (){

    void *context = zmq_ctx_new ();

    void *requester = zmq_socket (context, ZMQ_REQ);
    zmq_connect (requester, "ipc:///tmp/fifo-pipe-front-end");
    int n;
    int resp;
    while(1) {
        n = rand();
        printf ("Sending %d\n", n);
        zmq_send(requester, &n, sizeof(n), 0);        

        zmq_recv(requester, &resp, sizeof(resp), 0);
        if (resp){
            printf ("%d is prime\n", n);
        }else{
            printf ("%d is NOT prime\n", n);

        }
        sleep(1);
    }
    return 0;
}