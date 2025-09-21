#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "dbi_carm_roi.h"

#define CHUNK_SIZE 16
// #define ROI_ENABLED  //Enable this flag for CARM profilling

void matrix_multiply(float** a, float** b, float** c, float** t, int size) {
    int i = 0, j = 0, k = 0, ichunk = 0, jchunk = 0, ci = 0, cj = 0;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            t[i][j] = b[j][i];
        }
    }

    for (ichunk = 0; ichunk < size; ichunk += CHUNK_SIZE) {
        for (jchunk = 0; jchunk < size; jchunk += CHUNK_SIZE) {
            for (i = 0; i < CHUNK_SIZE; i++) {
                ci = ichunk + i;
                for (j = 0; j < CHUNK_SIZE; j++) {
                    cj = jchunk + j;
                    for (k = 0; k < size; k++) {
                        c[ci][cj] += a[ci][k] * t[cj][k];
                    }
                }
            }
        }
    }
}


int main(int argc, char* argv[]) {
    int msize = 512;  // Default size of the matrix
    int num_runs = 15;  // Default number of runs
    struct timespec t_start, t_end;

    // Check if enough arguments are provided
    if (argc > 2) {
        fprintf(stderr, "Usage: %s <num_runs>\n", argv[0]);
        return 1;  // Return an error code
    }

    // Parse number of runs from the second argument
    if (argc > 1){
        num_runs = atoi(argv[1]);
        if (num_runs <= 0) {
            fprintf(stderr, "Invalid number of runs: %s\n", argv[1]);
            return 1;  // Return an error code
        }
    }
    
    int i = 0;

    // Allocate matrices
    float** a = new float*[msize];
    float** b = new float*[msize];
    float** c = new float*[msize];
    float** t = new float*[msize];
    
    for (int i = 0; i < msize; i++) {
        a[i] = new float[msize];
        b[i] = new float[msize];

	    for (int j = 0; j < msize; j++) {
            a[i][j] = 5.0f; // Initialize all elements to 5
	        b[i][j] = 5.0f;
	    }

	    c[i] = new float[msize]();
        t[i] = new float[msize]();
    }

    // Perform matrix multiplication
    #ifdef ROI_ENABLED
    CARM_roi_begin();
    #else
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    __SSC_MARK(0xFACE);
    #endif
    for (i=0; i<num_runs; i++){
        matrix_multiply(a, b, c, t, msize);
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
	    delete[] t[i];
    }

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] t;

    return 0;
}
