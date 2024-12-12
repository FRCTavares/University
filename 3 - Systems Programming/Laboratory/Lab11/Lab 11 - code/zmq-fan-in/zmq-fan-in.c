#include <zmq.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>


//void *context;
void *  context;
void *thread_pusher( void *ptr ){

    long int thread_number = (long int)ptr;

    void *pusher = zmq_socket (context, ZMQ_PUSH);
    int rc = zmq_connect (pusher, "inproc://xxxx");
    assert (rc == 0);

    printf("Thread ready\n");
    while (1) {
        printf ("Pusher %ld sending %ld\n", thread_number, thread_number);
        sleep(1);
        zmq_send(pusher, &thread_number, sizeof(thread_number), 0);
    }
    return 0;
}



int main (void){

    context = zmq_ctx_new ();


    //  Socket facing subscribers
    void *sock_recv = zmq_socket (context, ZMQ_PULL);
    int rc = zmq_bind (sock_recv, "inproc://xxxx");
    assert (rc == 0);



    
    long int n_pub;
    for (n_pub = 0; n_pub < 2; n_pub++) {
        pthread_t worker;
        pthread_create( &worker, NULL, thread_pusher, (void *) n_pub);
    }


    long int n;
    printf("Thread ready\n");
    while (1) {
        zmq_recv(sock_recv, &n, sizeof(n), 0);
        printf ("\tMain received %ld\n", n);
    }
 

    //  We never get here...
    return 0;
}