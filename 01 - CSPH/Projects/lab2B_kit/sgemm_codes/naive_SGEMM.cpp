#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include "dbi_carm_roi.h"

// #define ROI_ENABLED //Enable this flag for CARM profilling

void matrix_multiply(float** a, float** b, float** c, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}


int main(int argc, char* argv[]) {
    int msize = 512;  // Default size of the matrix
    int num_runs = 15;  // Default number of runs

    // Check if enough arguments are provided
    if (argc > 3) {
        fprintf(stderr, "Usage: %s <matrix_size> <num_runs>\n", argv[0]);
        return 1;  // Return an error code
    }

    // Parse matrix size from the first argument
    if (argc > 1){
        msize = atoi(argv[1]);
        if (msize <= 0) {
            fprintf(stderr, "Invalid matrix size: %s\n", argv[1]);
            return 1;  // Return an error code
        }
    }

    // Parse number of runs from the second argument
    if (argc > 2){
        num_runs = atoi(argv[2]);
        if (num_runs <= 0) {
            fprintf(stderr, "Invalid number of runs: %s\n", argv[2]);
            return 1;  // Return an error code
        }
    }

    int i = 0;

    // Allocate matrices
    float** a = new float*[msize];
    float** b = new float*[msize];
    float** c = new float*[msize];

    for (int i = 0; i < msize; i++) {
        a[i] = new float[msize];
        b[i] = new float[msize];

        for (int j = 0; j < msize; j++) {
            a[i][j] = 5.0f; // Initialize all elements to 5
            b[i][j] = 5.0f;
        }

	    c[i] = new float[msize]();
    }

    // Perform matrix multiplication
    #ifdef ROI_ENABLED
    CARM_roi_begin();
    #else
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    __SSC_MARK(0xFACE);
    #endif
    for (i=0; i<num_runs; i++){
        matrix_multiply(a, b, c, msize);
    }
    #ifdef ROI_ENABLED
    CARM_roi_end();
    #else
    __SSC_MARK(0xDEAD);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double timing_duration = ((t_end.tv_sec + (double) t_end.tv_nsec / 1000000000) - (t_start.tv_sec + (double) t_start.tv_nsec / 1000000000));
    printf("Time Taken:\t%0.9lf seconds\n", timing_duration);
    #endif

    // Free matrices
    for (int i = 0; i < msize; i++) {
        delete[] a[i];
        delete[] b[i];
        delete[] c[i];
    }

    delete[] a;
    delete[] b;
    delete[] c;
    
    return 0;
}
