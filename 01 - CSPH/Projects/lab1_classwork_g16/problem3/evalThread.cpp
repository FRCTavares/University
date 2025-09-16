#include <stdio.h>
#include <thread>

#include "CycleTimer.h"

typedef struct {
    unsigned int width;
    unsigned int height;
    int* data;
    int* output;
    int threadId;
    int numThreads;
} WorkerArgs;


extern void evalSerial(int width, int height, int startRow, int totalRows, int* data, int output[]);


//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs * const args) {

    // TODO FOR STUDENTS: Implement the body of the worker
    // thread here. Each thread should make a call to evalSerial()
    // to compute a part of the output image.  For example, in a
    // program that uses two threads, thread 0 could compute the top
    // half of the image and thread 1 could compute the bottom half.

    int rowsPerThread = args->height / args->numThreads;
    int startRow = args->threadId * rowsPerThread;
    int totalRows = (args->threadId == args->numThreads - 1)
                        ? (args->height - startRow)  // Last thread takes the remainder
                        : rowsPerThread;
    evalSerial(args->width, args->height, startRow, totalRows, args->data, args->output);

    printf("Hello world from thread %d\n", args->threadId);
}

//
// evalThread
//
// Threads of execution are created by spawning std::threads.
void evalThread(int numThreads, int width, int height, int* data, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS];

    for (int i=0; i<numThreads; i++) {
      
        // TODO FOR STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].width = width;
        args[i].height = height;
        args[i].data = data;
        args[i].numThreads = numThreads;
        args[i].output = output;
      
        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i=1; i<numThreads; i++) {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }
    
    workerThreadStart(&args[0]);

    // join worker threads
    for (int i=1; i<numThreads; i++) {
        workers[i].join();
    }
}
