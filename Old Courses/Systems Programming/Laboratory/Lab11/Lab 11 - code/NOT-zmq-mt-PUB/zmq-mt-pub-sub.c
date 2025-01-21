#include <zmq.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>


//void *context;
void *  context;
void *thread_publisher( void *ptr ){

    long int thread_number = (long int)ptr;

    void *publisher = zmq_socket (context, ZMQ_PUB);
    int rc = zmq_connect (publisher, "tcp://127.0.0.1:5555");
    assert (rc == 0);

    printf("Thread ready\n");
    while (1) {
        printf ("Publisher %ld sending %ld\n", thread_number, thread_number);
        sleep(1);
        zmq_send(publisher, &thread_number, sizeof(thread_number), 0);
    }
    return 0;

}


void *thread_subscriber( void *ptr ){

    long int thread_number = (long int)ptr;

    void *subscriber = zmq_socket (context, ZMQ_SUB);
    int rc = zmq_connect (subscriber, "tcp://127.0.0.1:6666");
    zmq_setsockopt (subscriber, ZMQ_SUBSCRIBE, "", 0);
    assert (rc == 0);

    long int n;
    printf("Thread ready\n"); 
    while (1) {
        zmq_recv(subscriber, &n, sizeof(n), 0);
        printf ("\tSubscriber %ld received %ld\n", thread_number, n);
    }
    return 0;
}


int main (void){

    context = zmq_ctx_new ();

    //  Socket facing publishers
    void *sock_publishers = zmq_socket (context, ZMQ_XSUB);
    int rc = zmq_bind (sock_publishers, "tcp://127.0.0.1:5555");
    assert (rc == 0);

    //  Socket facing subscribers
    void *sock_subcribers = zmq_socket (context, ZMQ_XPUB);
    rc = zmq_bind (sock_subcribers, "tcp://127.0.0.1:6666");
    assert (rc == 0);



    
    long int n_pub;
    for (n_pub = 0; n_pub < 2; n_pub++) {
        pthread_t worker;
        pthread_create( &worker, NULL, thread_publisher, (void *) n_pub);
    }


   long int n_sub;
    for (n_sub = 0; n_sub < 2; n_sub++) {
        pthread_t worker;
        pthread_create( &worker, NULL, thread_subscriber, (void *) n_sub);
    }


    //  Start the proxy
    zmq_proxy (sock_publishers, sock_subcribers, NULL);

    //  We never get here...
    return 0;
}