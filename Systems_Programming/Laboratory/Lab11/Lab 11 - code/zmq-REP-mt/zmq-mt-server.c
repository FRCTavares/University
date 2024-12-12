#include <zmq.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>

//void *context;
void *  context;
void *thread_function( void *ptr ){

    long int thread_number = (long int)ptr;

    void *responder = zmq_socket (context, ZMQ_REP);
    int rc = zmq_connect (responder, "inproc://back-end");
    assert (rc == 0);

    int n;
    printf("Thread ready\n");
    while (1) {
        
        zmq_recv(responder, &n, sizeof(n), 0);
        printf ("Thread %ld received %d\n", thread_number, n);
        sleep(1);
        zmq_send(responder, &n, sizeof(n), 0);
    }
    return 0;

}


int main (void)
{

    context = zmq_ctx_new ();

    //  Socket facing clients
    void *frontend = zmq_socket (context, ZMQ_ROUTER);
    int rc = zmq_bind (frontend, "ipc:///tmp/fifo-pipe-front-end");
    assert (rc == 0);

    //  Socket facing services
    void *backend = zmq_socket (context, ZMQ_DEALER);
    rc = zmq_bind (backend, "inproc://back-end");
    assert (rc == 0);



    
    long int worker_nbr;
    for (worker_nbr = 0; worker_nbr < 2; worker_nbr++) {
        pthread_t worker;
        pthread_create( &worker, NULL, thread_function, (void *) worker_nbr);
    }



    //  Start the proxy
    zmq_proxy (frontend, backend, NULL);

    //  We never get here...
    return 0;
}