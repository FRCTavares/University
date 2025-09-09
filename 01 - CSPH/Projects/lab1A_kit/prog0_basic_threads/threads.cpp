#include <stdio.h>
#include <thread>


void thread_run_me(int thread_id) {
    for (int i=0; i<5; i++) {
        printf("Hello %d from thread %d\n", i, thread_id);
    }
}


int main(int argc, char** argv) {

    const int num_threads = 4;

    std::thread my_threads[num_threads];

    // "spawn" num_threads new threads.  Each thread is given a unique
    // thread id.

    for (int i=0; i<num_threads; i++) {
        my_threads[i] = std::thread(thread_run_me, i);
    }

    // the main thread now "joins" with all the "spawned" threads. A
    // call to join() will block until the requested thread has
    // completed. In this program, that means the "spawed" thread that
    // is being joined has returned from the function thread_run_me()

    printf("The main thread is about to join spawned threads.\n");

    for (int i=0; i<num_threads; i++) {
        my_threads[i].join();
    }

    return 0;
}
