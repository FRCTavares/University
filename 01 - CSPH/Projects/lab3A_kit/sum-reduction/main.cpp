#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

int reductionCuda(int N, int* array);
void printCudaInfo();


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}


int main(int argc, char** argv)
{

    // default: arrays of 500M numbers
    long int N = 500 * 1000 * 1000;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"arraysize",  1, 0, 'n'},
        {"help",       0, 0, '?'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?n:", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'n':
            N = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    int *array = new int[N];
    int result=0;

    for (int i=0; i<N; i++) {
        array[i] = rand()%11-5;
        result+=array[i];
    }
    int cudaResult;

    printCudaInfo();
    
    printf("Running 3 timing tests:\n");
    for (int i=0; i<3; i++) {
      cudaResult = reductionCuda(N, array);
    }

    delete [] array;
    
    printf("CPU result:%d\n",result);
    printf("CUDA result:%d\n",cudaResult);
    if(result!=cudaResult){
        printf("Results don't match! :(\n");
    }
    else{
        printf("Results match! :)\n");
    }

    return 0;
}
