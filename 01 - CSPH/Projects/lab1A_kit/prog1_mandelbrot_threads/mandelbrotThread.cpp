#include <stdio.h>
#include <thread>

#include "CycleTimer.h"

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

extern void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int numRows,
    int maxIterations,
    int output[]);

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{
    // Record start time for this thread
    double startTime = CycleTimer::currentSeconds();

    // OLD IMPLEMENTATION (COMMENTED OUT): Contiguous block assignment
    // This approach assigns contiguous blocks of rows to each thread,
    // but leads to load imbalance due to varying computational complexity
    // across different regions of the Mandelbrot set.
    /*
    int rowsPerThread = args->height / args->numThreads;
    int startRow = args->threadId * rowsPerThread;

    // Treat the last thread specially to cover all rows
    int numRows = rowsPerThread;
    if (args->threadId == args->numThreads - 1)
    {
        numRows = args->height - startRow;
    }

    // Call the serial function to compute this thread's portion
    mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                     args->width, args->height,
                     startRow, numRows,
                     args->maxIterations, args->output);
    */

    // NEW IMPLEMENTATION: Interleaved row assignment
    // Each thread processes rows in round-robin fashion for better load balancing
    // Thread 0: rows 0, 4, 8, 12, ...
    // Thread 1: rows 1, 5, 9, 13, ...
    // Thread 2: rows 2, 6, 10, 14, ...
    // etc.
    int rowsProcessed = 0;
    for (int row = args->threadId; row < args->height; row += args->numThreads)
    {
        // Call the serial function to compute one row at a time
        mandelbrotSerial(args->x0, args->y0, args->x1, args->y1,
                         args->width, args->height,
                         row, 1, // startRow = row, numRows = 1
                         args->maxIterations, args->output);
        rowsProcessed++;
    }

    // Record end time and calculate elapsed time for this thread
    double endTime = CycleTimer::currentSeconds();
    double elapsedTime = (endTime - startTime) * 1000; // Convert to milliseconds

    // Print timing information for this thread
    printf("Thread %d: computed %d rows (interleaved) in %.3f ms\n",
           args->threadId, rowsProcessed, elapsedTime);
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
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

    for (int i = 0; i < numThreads; i++)
    {

        // TODO FOR STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread

        // Explanation of each field:
        // x0, y0, x1, y1:  Coordinates of the image
        // width, height:   Width and height of the image
        // maxIterations:   Number of iterations for each pixel
        // output:          Output array
        // threadId:       ID of the thread (0 to numThreads-1)
        // numThreads:     Total number of threads

        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]);

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
}
