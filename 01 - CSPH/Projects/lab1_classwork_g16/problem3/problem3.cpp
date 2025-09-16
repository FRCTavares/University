#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <fstream>
#include <getopt.h>

#include "CycleTimer.h"

extern void evalSerial(int width, int height, int startRow, int totalRows, int* data, int output[]);

extern void evalThread(int numThreads, int width, int height, int* data, int output[]);

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -t  --threads <N>  Use N threads\n");
    printf("  -?  --help         This message\n");
}

bool verifyResult (int *gold, int *result, int width, int height) {

    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (gold[i * width + j] != result[i * width + j]) {
                printf ("ERROR :: Mismatch : [%d][%d], Expected : %d, Actual : %d\n",
                            i, j, gold[i * width + j], result[i * width + j]);
                return 0;
            }
        }
    }

    printf("Correctness verification: PASSED!");

    return 1;
}

void readImage(char* filename, int* serial, int width, int height){
    std::ifstream ppmFile(filename, std::ios::binary);
    std::string magicNumber;
    ppmFile >> magicNumber;

    char ch;
    ppmFile.get(ch);
    while (ch == '#') {
        ppmFile.ignore(10000, '\n');
        ppmFile.get(ch);
    }
    ppmFile.unget();  // Put back the last read character

    int width_, height_, maxColor;
    ppmFile >> width_ >> height_ >> maxColor;
    ppmFile.get();

    // Read binary pixel data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            unsigned char r, g, b;
            int index = (i * width + j);
            ppmFile.read(reinterpret_cast<char*>(&r), 1);
            ppmFile.read(reinterpret_cast<char*>(&g), 1);
            ppmFile.read(reinterpret_cast<char*>(&b), 1);
            serial[index] = static_cast<int>(b);
        }
    }

    ppmFile.close();
}

int main(int argc, char** argv) {

    const unsigned int width = 1600;
    const unsigned int height = 1200;
    int numThreads = 2;

    if(argc != 3){
        printf("Usage: ./problem2 imagename numthreads");
    }

    int* serial = new int[width*height];
    memset(serial, 0, width * height * sizeof(int));

    readImage(argv[1], serial, width, height);
    numThreads = atoi(argv[2]);

    int* output_serial = new int[width*height];
    int* output_thread = new int[width*height];
    
    //
    // Run the serial implementation.  
    // Run the code three times and take the minimum to 
    // get a good estimate.
    //

    double minSerial = 1e30;
    for (int i = 0; i < 5; ++i) {
        memset(output_serial, 0, width * height * sizeof(int));
        double startTime = CycleTimer::currentSeconds();
        evalSerial(width, height, 0, height, serial, output_serial);
        double endTime = CycleTimer::currentSeconds();
        minSerial = std::min(minSerial, endTime - startTime);
    }

    printf("[Serial]:\t\t[%.3f] ms\n", minSerial * 1000);

    //
    // Run the threaded version for the first image
    //

    double minThread = 1e30;
    for (int i = 0; i < 5; ++i) {
      memset(output_thread, 0, width * height * sizeof(int));
        double startTime = CycleTimer::currentSeconds();
        evalThread(numThreads, width, height, serial, output_thread);
        double endTime = CycleTimer::currentSeconds();
        minThread = std::min(minThread, endTime - startTime);
    }

    printf("[Threads]:\t\t[%.3f] ms\n", minThread * 1000);

    // compute speedup
    printf("\t\t\t\t(%.2fx speedup from %d threads)\n", minSerial/minThread, numThreads);

    verifyResult(output_serial,output_thread, width, height);

    delete[] serial;
    delete[] output_serial;
    delete[] output_thread;

    return 0;
}
