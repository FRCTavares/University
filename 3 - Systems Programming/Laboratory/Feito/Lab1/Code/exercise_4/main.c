#include "lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

int main(){
    int a;
    char line[100];
    char library_name[100];
    void *handle;
    void (*func_1)();
    void (*func_2)();

    printf("What version of the functions do you want to use?\n");
    printf("\t1 - Normal (lib1)\n");
    printf("\t2 - Optimized (lib2)\n");
    fgets(line, 100, stdin);
    sscanf(line, "%d", &a);

    // Set the library path based on user input
    if (a == 1) {
        strcpy(library_name, "./lib1.so");
        printf("Running the normal version from %s\n", library_name);
    } else if (a == 2) {
        strcpy(library_name, "./lib2.so");
        printf("Running the optimized version from %s\n", library_name);
    } else {
        printf("Invalid selection. Exiting.\n");
        exit(-1);
    }

    // Load the library
    handle = dlopen(library_name, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Error: %s\n", dlerror());
        exit(1);
    }

    // Load the functions from the library
    func_1 = dlsym(handle, "func_1");
    func_2 = dlsym(handle, "func_2");

    // Call the functions
    if (func_1) func_1();
    if (func_2) func_2();

    // Close the library
    dlclose(handle);

    return 0;
}
